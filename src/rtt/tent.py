"""Independent, episodic Tent proposals for the last four ResNet-50 blocks.

This module implements entropy minimization of batch-normalization affine
parameters, as in Wang et al., *Tent: Fully Test-Time Adaptation by Entropy
Minimization* (ICLR 2021), https://github.com/DequanWang/tent. The RTT image
experiment identifies the last four bottlenecks as its adaptation region
(RTT Appendix F / Table 15). Accordingly, the earlier image encoder, all
convolutions, and the classifier remain frozen. This restricted region and
the fresh Adam optimizer on every call are explicit reconstruction choices;
upstream Tent supports adaptation of batch norms throughout a model and
continual updates across calls.

``prepare`` must run before RTT admission. It receives no labels, adapts an
independent copy, then exports a complete post-update trajectory. Returned
states do not depend on any RTT acceptance decision or incoming repaired
state. A caller may form its proposal direction as ``cache[depth] - Hbar``.
The original incumbent is never modified, and no adapted model or optimizer
is retained between calls. Entropies in the report use current-batch
statistics in the selected batch norms, both before and after optimization;
``entropy_before`` is not the frozen incumbent's running-statistic entropy.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn


def tail_batch_norms(model: nn.Module) -> list[tuple[str, nn.BatchNorm2d]]:
    """Read-only selection of BN modules in ResNet's four committed blocks.

    The selection includes the projection-shortcut batch norm in layer4.0.
    On a standard ResNet-50 this selects 13 modules / 26 affine tensors /
    25,600 scalar parameters, with no batch norm from the source prefix.
    """
    if getattr(model, "family", None) != "resnet50" or getattr(model, "T", None) != 4:
        raise ValueError("Tent proposals require the ResNet-50 reference with a four-block tail")
    backbone = getattr(model, "backbone", None)
    stage3, stage4 = getattr(backbone, "layer3", None), getattr(backbone, "layer4", None)
    if not isinstance(stage3, nn.Sequential) or not isinstance(stage4, nn.Sequential):
        raise ValueError("Tent requires backbone.layer3 and backbone.layer4 block sequences")
    if not len(stage3) or len(stage4) != 3:
        raise ValueError("Tent expects the last layer3 block followed by three layer4 blocks")
    regions = [(f"backbone.layer3.{len(stage3) - 1}", stage3[-1])]
    regions.extend((f"backbone.layer4.{i}", block) for i, block in enumerate(stage4))
    selected = []
    for prefix, block in regions:
        for suffix, module in block.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if not module.affine or module.weight is None or module.bias is None:
                    raise ValueError("Tent needs affine weight and bias in every selected batch norm")
                selected.append((f"{prefix}.{suffix}" if suffix else prefix, module))
    if not selected:
        raise ValueError("The committed ResNet tail contains no adaptable batch normalization")
    return selected


def _configure_copy(model: nn.Module) -> tuple[nn.Module, list[tuple[str, nn.Parameter]]]:
    """Copy first, so even training modes and running-statistic buffers are isolated."""
    adapted = deepcopy(model)
    adapted.eval()
    adapted.requires_grad_(False)
    parameters = []
    for name, module in tail_batch_norms(adapted):
        module.train()
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None
        module.num_batches_tracked = None
        for affine in ("weight", "bias"):
            parameter = getattr(module, affine)
            parameter.requires_grad_(True)
            parameters.append((f"{name}.{affine}", parameter))
    return adapted, parameters


def _entropy(logits: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Image Tent requires [batch, classes] logits with at least two classes")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Non-finite logits during Tent preparation")
    # Float32 entropy avoids underflow for mixed-precision image networks.
    values = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    log_probabilities = values.log_softmax(dim=-1)
    return -(log_probabilities.exp() * log_probabilities).sum(dim=-1).mean()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TentProposal:
    """Reset-per-call, label-free proposal producer; it is not an RTT gate.

    ``steps`` positive Adam steps use the same input call throughout. Adam's
    defaults are beta=(0.9, 0.999), epsilon=1e-8, and zero weight decay. The
    frozen source prefix is evaluated once; the four-block tail is evaluated
    once per optimization step and once more to export post-update states.
    All of that work, including the isolated model copy, is timed and must be
    included in the caller's end-to-end latency. The report is JSON-serializable.
    """

    def __init__(self, model: nn.Module, learning_rate: float = 0.001, steps: int = 1):
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("Tent steps must be a positive integer")
        if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
            raise ValueError("Tent learning_rate must be finite and positive")
        tail_batch_norms(model)  # Fail unsupported families before any expensive work.
        self.model = model
        self.learning_rate, self.steps = float(learning_rate), int(steps)
        self.last_report: dict[str, Any] = {}

    @property
    def report(self) -> dict[str, Any]:
        """Latest preparation diagnostics; alias used by the generic RTT engine."""
        return self.last_report

    @torch.inference_mode(False)
    def prepare(self, batch: Any) -> dict[int, Tensor]:
        """Adapt on a ``Batch.unlabeled()`` and return detached states at depths 0..3.

        No mask, ground truth, posterior calibration, RTT trace, or acceptance
        feedback enters the entropy objective. Labels must be explicitly absent.
        ``torch.enable_grad`` permits use inside a caller's evaluation context.
        """
        self.last_report = {}
        if not hasattr(batch, "labels") or batch.labels is not None:
            raise ValueError("Tent requires an explicitly unlabeled batch; call batch.unlabeled()")
        if getattr(batch, "multilabel", False):
            raise ValueError("Image Tent expects mutually exclusive classes, not multilabel targets")
        x = batch.x
        if x.ndim != 4 or x.shape[0] < 1 or x.shape[1] != 3 or not x.is_floating_point():
            raise ValueError("Tent expects a nonempty floating-point [batch,3,height,width] image tensor")
        _synchronize(x.device)
        started = perf_counter()
        adapted, named_parameters = _configure_copy(self.model)
        parameters = [parameter for _, parameter in named_parameters]
        initial = [parameter.detach().clone() for parameter in parameters]
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate, betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=0)
        context = dict(batch.context)
        # Prefix modules remain in eval mode, and do not participate in adaptation.
        with torch.no_grad():
            start_state = adapted.encode(x.detach(), context).detach()
        entropy_before = None
        with torch.enable_grad():
            for _ in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                state = start_state
                for depth in range(adapted.T):
                    state = adapted.transition(state, depth, context)
                loss = _entropy(adapted.logits(state, context))
                if entropy_before is None:
                    entropy_before = float(loss.detach())
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                       for parameter in parameters):
                    raise FloatingPointError("Non-finite batch-normalization gradient during Tent preparation")
                optimizer.step()
        # Returning a pre-update forward would silently discard a one-step adaptation.
        candidates = {}
        with torch.no_grad():
            state = start_state
            for depth in range(adapted.T):
                state = adapted.transition(state, depth, context)
                candidates[depth] = state.detach().clone()
            entropy_after = float(_entropy(adapted.logits(state, context)))
            changed = [int(torch.count_nonzero(parameter.detach() != before))
                       for parameter, before in zip(parameters, initial)]
        # Release the per-call copy and optimizer; no state persists to a later call.
        parameter_count = sum(parameter.numel() for parameter in parameters)
        names = [name for name, _ in named_parameters]
        del adapted, optimizer, parameters, named_parameters, initial, state, loss, start_state, context
        _synchronize(x.device)
        self.last_report = {
            "kind": "tent",
            "scope": "last_four_resnet50_bottlenecks_bn_affine",
            "episodic": True,
            "reset_each_call": True,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "optimizer": "Adam",
            "selected_parameter_names": names,
            "parameter_tensors_adaptable": len(names),
            "parameter_values_adaptable": parameter_count,
            "parameter_tensors_updated": sum(count > 0 for count in changed),
            "parameter_values_updated": sum(changed),
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "preparation_seconds": perf_counter() - started,
            "prefix_forward_evaluations": 1,
            "tail_forward_evaluations": self.steps + 1,
            "candidate_state_semantics": "post_update_independent_trajectory",
        }
        return candidates
