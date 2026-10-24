"""Tent adapts isolated BN affine parameters and cannot consume labels or RTT feedback."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def modules():
    root = Path(__file__).resolve().parents[1] / "src" / "rtt"
    result = {}
    for name in ("models", "tent"):
        spec = importlib.util.spec_from_file_location(f"rtt_{name}_tent_test", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result[name] = module
    torch.set_num_threads(1)
    return SimpleNamespace(**result)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return torch.tanh(self.bn(self.conv(x)) + x)


class TinyImageReference(nn.Module):
    """A small real convolution/BN network with the same four-tail-block interface."""
    family, task_level, T = "resnet50", "image", 4

    def __init__(self):
        super().__init__()
        self.prefix = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Tanh(), nn.Dropout(0.5))
        self.backbone = nn.Module()
        self.backbone.layer3 = nn.Sequential(TinyBlock(), TinyBlock())
        self.backbone.layer4 = nn.Sequential(TinyBlock(), TinyBlock(), TinyBlock())
        self.backbone.fc = nn.Linear(4, 3)

    def encode(self, x, context):
        state = self.backbone.layer3[0](self.prefix(x))
        context["h0"] = state
        return state

    def transition(self, state, depth, context):
        block = self.backbone.layer3[-1] if depth == 0 else self.backbone.layer4[depth - 1]
        return block(state)

    def logits(self, state, context):
        return self.backbone.fc(state.mean(dim=(-2, -1)))


def tiny_case():
    torch.manual_seed(71)
    source = TinyImageReference().eval().requires_grad_(False)
    x = torch.randn(5, 3, 4, 4) + torch.arange(5)[:, None, None, None] * 0.7
    batch = SimpleNamespace(x=x, context={"marker": torch.ones(1)}, labels=None, multilabel=False)
    return source, batch


def test_actual_resnet_selects_every_bn_in_exactly_last_four_blocks(modules):
    source = modules.models.build_model("resnet50", 3, 1000, {"pretrained": False}).eval()
    selected = modules.tent.tail_batch_norms(source)
    names = [name for name, _ in selected]
    assert len(names) == 13
    assert "backbone.layer3.5.bn1" in names
    assert "backbone.layer4.0.downsample.1" in names
    assert all(name.startswith(("backbone.layer3.5.", "backbone.layer4.")) for name in names)
    assert sum(module.weight.numel() + module.bias.numel() for _, module in selected) == 25_600
    assert all(not module.training and module.track_running_stats for _, module in selected)


def test_only_copied_tail_bn_affine_parameters_are_adaptable(modules):
    source, _ = tiny_case()
    adapted, selected = modules.tent._configure_copy(source)
    selected_names = {name for name, _ in selected}
    assert {name for name, parameter in adapted.named_parameters() if parameter.requires_grad} == selected_names
    assert all(name.startswith(("backbone.layer3.1.", "backbone.layer4.")) for name in selected_names)
    selected_modules = {name for name, _ in modules.tent.tail_batch_norms(adapted)}
    for name, module in adapted.named_modules():
        assert module.training == (name in selected_modules)
        if name in selected_modules:
            assert not module.track_running_stats
            assert module.running_mean is module.running_var is module.num_batches_tracked is None
    assert adapted.prefix[1].running_mean is not None
    assert all(not parameter.requires_grad for parameter in source.parameters())
    assert all(not module.training for module in source.modules())


def test_preparation_really_updates_entropy_and_preserves_source_and_input(modules):
    source, batch = tiny_case()
    original_state = deepcopy(source.state_dict())
    original_modes = {name: module.training for name, module in source.named_modules()}
    original_requires = {name: parameter.requires_grad for name, parameter in source.named_parameters()}
    original_x, original_context = batch.x.clone(), dict(batch.context)
    proposal = modules.tent.TentProposal(source, learning_rate=0.005, steps=3)
    with torch.no_grad():
        candidates = proposal.prepare(batch)
    assert set(candidates) == {0, 1, 2, 3}
    assert all(state.shape == (5, 4, 4, 4) for state in candidates.values())
    assert all(not state.requires_grad and state.grad_fn is None for state in candidates.values())
    report = proposal.report
    assert report is proposal.last_report
    assert report["parameter_tensors_updated"] > 0
    assert report["parameter_values_updated"] > 0
    assert report["entropy_after"] < report["entropy_before"]
    assert report["preparation_seconds"] > 0 and report["tail_forward_evaluations"] == 4
    json.dumps(report, allow_nan=False)
    # A final output reconstructed from the returned state has the reported entropy.
    logits = source.logits(candidates[3], {})
    observed = -(logits.log_softmax(-1).exp() * logits.log_softmax(-1)).sum(-1).mean()
    assert report["entropy_after"] == float(observed)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
    assert original_modes == {name: module.training for name, module in source.named_modules()}
    assert original_requires == {name: parameter.requires_grad for name, parameter in source.named_parameters()}
    assert all(parameter.grad is None for parameter in source.parameters())
    torch.testing.assert_close(batch.x, original_x, rtol=0, atol=0)
    assert set(batch.context) == set(original_context)
    assert batch.context["marker"] is original_context["marker"]


def test_every_call_resets_model_and_optimizer_even_after_another_batch(modules):
    source, batch = tiny_case()
    proposal = modules.tent.TentProposal(source, steps=2)
    first = proposal.prepare(batch)
    first_report = dict(proposal.report)
    different = SimpleNamespace(x=-batch.x * 2, context={}, labels=None, multilabel=False)
    proposal.prepare(different)
    # Inference mode is common in an evaluation runner; adaptation still needs gradients.
    with torch.inference_mode():
        repeated = proposal.prepare(batch)
    for depth in first:
        torch.testing.assert_close(repeated[depth], first[depth], rtol=0, atol=0)
    assert proposal.report["entropy_before"] == first_report["entropy_before"]
    assert proposal.report["entropy_after"] == first_report["entropy_after"]


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_step_counts_are_rejected(modules, steps):
    source, _ = tiny_case()
    with pytest.raises(ValueError, match="positive integer"):
        modules.tent.TentProposal(source, steps=steps)


@pytest.mark.parametrize("learning_rate", [0, -1, float("nan"), float("inf")])
def test_invalid_learning_rates_are_rejected(modules, learning_rate):
    source, _ = tiny_case()
    with pytest.raises(ValueError, match="finite and positive"):
        modules.tent.TentProposal(source, learning_rate=learning_rate)


def test_labels_multilabel_tasks_and_nonimage_families_are_rejected(modules):
    source, batch = tiny_case()
    proposal = modules.tent.TentProposal(source)
    batch.labels = torch.tensor([0, 1, 0, 1, 2])
    with pytest.raises(ValueError, match="unlabeled"):
        proposal.prepare(batch)
    assert proposal.report == {}
    batch.labels = None
    batch.multilabel = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        proposal.prepare(batch)
    graph = modules.models.build_model("appnp", 3, 2, {"hidden": 4, "T": 4})
    with pytest.raises(ValueError, match="ResNet-50"):
        modules.tent.TentProposal(graph)
