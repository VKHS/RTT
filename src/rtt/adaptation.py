"""General reference-tail admission for graph, image, molecular and tabular models.

This backend evaluates the actual frozen continuation before and after a trial
step. It computes an exact all-class price in real arithmetic, and a label-free
finite-step remainder around a tail JVP. It trades additional tail evaluations
for applicability to arbitrary supplied model primitives. The native arxiv tube
backend remains in rtt.core; their runtime claims must not be conflated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .batching import Batch
from .losses import floor_damage, log_probabilities, supervised_loss, unit_weights
from .runtime import NUMERICAL_STATUS, Timer


def row_features(state):
    return state if state.ndim == 2 else state.mean(dim=tuple(range(2, state.ndim)))


def broadcast_rows(features, state):
    return features if state.ndim == 2 else features[(...,) + (None,) * (state.ndim - 2)].expand_as(state)


def row_norm(value):
    return value.flatten(1).double().norm(dim=1)


def trim_rows(direction, limit):
    factor = (float(limit) / row_norm(direction).clamp_min(1e-30)).clamp_max(1).to(direction)
    return direction * factor.reshape((-1,) + (1,) * (direction.ndim - 1))


class DepthAdapter(nn.Module):
    """Two-candidate hull and field/covariance heads at one intervention depth."""
    def __init__(self, channels, config, seed):
        super().__init__()
        hidden, count = config["head_hidden"], config["bank_size"]
        rank = min(16, channels)
        self.bank = nn.Parameter(torch.randn(count, channels) * 0.02)
        self.bank_weights = nn.Sequential(nn.Linear(channels, hidden), nn.Tanh(), nn.Linear(hidden, count))
        self.value = nn.Sequential(nn.Linear(channels, rank), nn.Tanh(), nn.Linear(rank, channels))
        self.query = nn.Linear(channels, min(8, channels), bias=False)
        self.key = nn.Linear(channels, min(8, channels), bias=False)
        self.sign = nn.Linear(channels, channels)
        generator = torch.Generator().manual_seed(seed)
        probes = torch.randn(config["probes"], channels, generator=generator)
        probes = F.normalize(probes, dim=1)
        self.register_buffer("probes", torch.cat((probes, -probes), dim=0))
        input_features = channels + 2 * config["probes"]
        self.field = nn.Sequential(nn.Linear(input_features, hidden), nn.Tanh(), nn.Linear(hidden, channels))
        self.variance = nn.Sequential(nn.Linear(input_features, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.node_weight = nn.Sequential(nn.Linear(input_features, hidden), nn.Tanh(), nn.Linear(hidden, 1), nn.Sigmoid())
        self.covariance_floor = config["covariance_floor"]

    def _signed_message(self, state, context):
        features = row_features(state)
        value = self.value(features)
        edge = context.get("edge_index")
        if state.ndim > 2:
            # P=I at the example level; the proposal may use spatial neighbours.
            local = F.avg_pool2d(state, 3, stride=1, padding=1) - state
            return local * broadcast_rows(torch.tanh(self.sign(features)), state)
        if edge is None or edge.numel() == 0:
            return torch.tanh(self.sign(features)) * value
        source, target = edge
        q, k = self.query(features), self.key(features)
        score = torch.tanh((q[target] * k[source]).sum(1) / math.sqrt(q.shape[1]))
        denominator = score.new_zeros(len(state)).index_add_(0, target, score.abs()).clamp_min(1e-6)
        operator = torch.sparse_coo_tensor(torch.stack((target, source)), score / denominator[target],
                                           (len(state), len(state))).coalesce()
        return torch.sparse.mm(operator, value)

    def support_features(self, candidates):
        vectors = torch.stack([row_features(candidate) for candidate in candidates])
        support = torch.einsum("mnc,pc->mnp", vectors, self.probes).amax(dim=0).clamp_min(0)
        return support

    def candidates(self, state, context, row_trim, exported_state=None):
        features = row_features(state)
        first = broadcast_rows(torch.softmax(self.bank_weights(features), dim=1) @ self.bank, state)
        if exported_state is not None:
            if exported_state.shape != state.shape:
                raise ValueError("Exported proposal state has incompatible shape")
            first = exported_state.to(state) - state
        second = self._signed_message(state, context)
        candidates = [trim_rows(first, row_trim), trim_rows(second, row_trim)]
        inputs = torch.cat((features, self.support_features(candidates)), dim=1)
        weight = self.node_weight(inputs)
        return torch.stack([candidate * weight.reshape((-1,) + (1,) * (state.ndim - 1)) for candidate in candidates])

    def estimate(self, state, candidates, units):
        features = torch.cat((row_features(state), self.support_features(candidates)), dim=1)
        spatial_size = state[0, 0].numel() if state.ndim > 2 else 1
        field = broadcast_rows(self.field(features), state) / (max(1, units) * spatial_size)
        variance = (F.softplus(self.variance(features)).squeeze(1) + self.covariance_floor)
        variance = variance / (max(1, units) ** 2 * spatial_size ** 2)
        return field, variance


class AdaptationModules(nn.Module):
    def __init__(self, model, config, *, seed=1):
        super().__init__()
        self.config = dict(config)
        self.open_depths = list(range(min(config["open_first"], model.T), model.T))
        self.depths = nn.ModuleDict({str(depth): DepthAdapter(model.state_channels(depth), config, seed + depth)
                                    for depth in self.open_depths})
        head_parameters = sum(p.numel() for module in self.depths.values()
                              for part in (module.field, module.variance, module.node_weight)
                              for p in part.parameters())
        # A two-layer output corrector matched to the admission-head parameter count.
        width = max(1, round((head_parameters - model.output_dim) / (model.embedding_dim + model.output_dim + 1)))
        self.corrector = nn.Sequential(nn.Linear(model.embedding_dim, width), nn.Tanh(), nn.Linear(width, model.output_dim))
        self.capacity = {"admission_head_parameters": head_parameters,
                         "corrector_parameters": sum(p.numel() for p in self.corrector.parameters()),
                         "corrector_hidden": width}

    def proposal_parameters(self):
        for module in self.depths.values():
            for name in ("bank", "bank_weights", "value", "query", "key", "sign", "node_weight"):
                part = getattr(module, name)
                if isinstance(part, nn.Parameter):
                    yield part
                else:
                    yield from part.parameters()

    def head_parameters(self):
        for module in self.depths.values():
            yield from module.field.parameters()
            yield from module.variance.parameters()


def role_ids(state, context, roles):
    # Fixed structural partition, independent of labels and candidate ordering.
    membership = context.get("batch")
    if membership is not None and len(membership) <= len(state):
        extra = torch.arange(len(state) - len(membership), device=state.device)
        return torch.cat((membership, extra)) % roles
    return torch.arange(len(state), device=state.device) % roles


def coefficient_geometry(candidates, field, variance, role, roles):
    count, rows = candidates.shape[:2]
    flat = candidates.flatten(2).double()
    f = field.flatten(1).double()
    row_gram = torch.einsum("mnd,pnd->nmp", flat, flat)
    row_slope = torch.einsum("mnd,nd->nm", flat, f)
    slope = field.new_zeros((roles, count), dtype=torch.float64).index_add_(0, role, row_slope)
    gram_blocks = field.new_zeros((roles, count, count), dtype=torch.float64).index_add_(0, role, row_gram)
    covariance_blocks = torch.zeros_like(gram_blocks).index_add_(0, role, row_gram * variance.double()[:, None, None])
    gram = torch.block_diag(*gram_blocks.unbind(0))
    covariance = torch.block_diag(*covariance_blocks.unbind(0))
    return slope.flatten(), covariance, gram


def physical_step(coefficients, candidates, role):
    factors = coefficients.reshape(-1, candidates.shape[0])[role].T
    factors = factors.reshape(factors.shape + (1,) * (candidates.ndim - 2))
    return (candidates * factors.to(candidates)).sum(dim=0)


def joint_score(error, covariance):
    error, covariance = error.double(), covariance.double()
    eigenvalues, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    threshold = max(float(eigenvalues.max()) * 1e-10, 1e-30)
    active = eigenvalues > threshold
    coordinates = vectors.T @ error
    if bool((coordinates[~active].abs() > 1e-7 * error.norm().clamp_min(1e-12)).any()):
        return float("inf")
    if not bool(active.any()):
        return 0.0 if float(error.norm()) <= 1e-12 else float("inf")
    return float(torch.sqrt((coordinates[active].square() / eigenvalues[active]).sum()))


def projected_direction(slope, covariance, gram, kappa, config):
    """Projected convex surrogate solve; exact nonlinear predicates follow it."""
    if math.isnan(kappa) or kappa < 0:
        raise ValueError("Uncertainty radii must be nonnegative; positive infinity closes admission")
    if not math.isfinite(kappa) or slope.numel() == 0 or not bool((slope < 0).any()):
        return torch.zeros_like(slope), "zero_or_infinite_radius"
    t = torch.zeros_like(slope)
    moment, squared = torch.zeros_like(t), torch.zeros_like(t)
    best, best_value = t.clone(), 0.0
    regularizer = config["rho_x"] / config["omega"]
    for iteration in range(1, config["solver_steps"] + 1):
        norm = torch.sqrt((t @ covariance @ t).clamp_min(0) + 1e-24)
        gradient = slope + kappa * (covariance @ t) / norm + regularizer * (gram @ t)
        moment = 0.9 * moment + 0.1 * gradient
        squared = 0.999 * squared + 0.001 * gradient.square()
        t = (t - 0.1 * (moment / (1 - 0.9 ** iteration)) /
             ((squared / (1 - 0.999 ** iteration)).sqrt() + 1e-12)).clamp_min(0)
        blocks = t.reshape(config["roles"], 2)
        blocks = blocks / (blocks.sum(1, keepdim=True) / config["coefficient_cap"]).clamp_min(1)
        t = blocks.flatten()
        objective = slope @ t + kappa * torch.sqrt((t @ covariance @ t).clamp_min(0)) + 0.5 * regularizer * (t @ gram @ t)
        if torch.isfinite(objective) and float(objective) < best_value:
            best, best_value = t.clone(), float(objective)
    return best, "approximate_projected_surrogate" if best_value < 0 else "zero_no_robust_descent"


def tail_logits(model, batch, state, next_depth, context):
    return batch.select(model.tail(state, next_depth, context))


def true_tail_gradient(model, batch, state, next_depth, context):
    with torch.enable_grad():
        variable = state.detach().requires_grad_(True)
        logits = tail_logits(model, batch, variable, next_depth, context)
        loss = supervised_loss(logits, batch.labels, batch.multilabel, batch.mask)
        return torch.autograd.grad(loss, variable)[0].detach()


@dataclass
class CallResult:
    logits: torch.Tensor
    reference_logits: torch.Tensor
    state: torch.Tensor
    reference_state: torch.Tensor
    context: dict
    report: dict


class ReferenceTailEngine:
    def __init__(self, model, modules, config):
        self.model, self.modules, self.config = model, modules, dict(config)

    def _step(self, batch, state, depth, context, kappa, remaining_call, remaining_rows,
              open_left, exported=None):
        config, model = self.config, self.model
        module = self.modules.depths[str(depth)]
        with torch.no_grad():
            candidates = module.candidates(state, context, config["row_trim"], exported)
            field, variance = module.estimate(state, candidates, batch.units)
            role = role_ids(state, context, config["roles"])
            slope, covariance, gram = coefficient_geometry(candidates, field, variance, role, config["roles"])
            coefficients, solver_status = projected_direction(slope, covariance, gram, kappa, config)
        if not bool(coefficients.any()):
            return state, None, {"depth": depth, "status": solver_status}
        weights = unit_weights(batch.mask, state.device)
        price_budget = max(0.0, remaining_call) / max(1, open_left)
        margin = config["budget_margin"]
        with torch.no_grad():
            before = tail_logits(model, batch, state, depth + 1, context)
            before_log = log_probabilities(before, batch.multilabel)
        for trial in range(config["line_search_steps"]):
            t = coefficients * (0.5 ** trial)
            displacement = physical_step(t, candidates, role)
            ideal_displacement = displacement
            proposed = state + displacement
            # The actual rounded displacement is used in the finite-step JVP.
            displacement = proposed - state
            with torch.no_grad():
                after = tail_logits(model, batch, proposed, depth + 1, context)
                after_log = log_probabilities(after, batch.multilabel)
                per_unit = (before_log - after_log).amax(-1).clamp_min(0)
                charge = float((per_unit * weights).sum())
            if not math.isfinite(charge) or charge > max(0.0, price_budget - margin):
                continue
            if bool((per_unit[batch.mask] > remaining_rows[batch.mask] - margin).any()):
                continue
            def log_tail(variable):
                return log_probabilities(tail_logits(model, batch, variable, depth + 1, context), batch.multilabel)
            with torch.enable_grad():
                _, tangent = torch.autograd.functional.jvp(log_tail, state.detach(), displacement.detach(),
                                                          create_graph=False, strict=False)
            # For every possible label: loss change <= true linear slope + this
            # measured, label-independent remainder. No smoothness assumption is
            # substituted for a crossed ReLU kink.
            remainder = float((((before_log - after_log + tangent).amax(-1)).clamp_min(0) * weights).sum())
            uncertainty = kappa * math.sqrt(max(0.0, float(t @ covariance @ t)))
            regularizer = config["rho_x"] * float(displacement.double().square().sum()) / 2
            phi = float(slope @ t) + uncertainty + (remainder + regularizer) / config["omega"]
            if math.isfinite(phi) and phi <= -margin:
                return proposed.detach(), per_unit.detach(), {
                    "depth": depth, "status": "ADMITTED", "charge": charge, "remainder": remainder,
                    "robust_model": phi, "uncertainty": uncertainty, "scale": 0.5 ** trial,
                    "solver": solver_status, "row_step_max": float(row_norm(displacement).max()),
                    "addition_rounding_residual_norm": float((displacement - ideal_displacement).double().norm()),
                }
        return state, None, {"depth": depth, "status": "NO_FEASIBLE_DESCENT", "solver": solver_status}

    def serve(self, batch: Batch, radii, *, budget=None, stop_before=None, observer=None):
        if batch.labels is not None:
            raise ValueError("Online admission accepts only batch.unlabeled(); labels belong to offline training/audit")
        model, config = self.model, self.config
        H_plus = config["H_plus"] if budget is None else float(budget)
        if not math.isfinite(H_plus) or H_plus < 0:
            raise ValueError("The call budget must be finite and nonnegative")
        model.eval()
        self.modules.eval()
        accepted, steps = 0, []
        row_spend = torch.zeros_like(batch.mask, dtype=torch.float64)
        with Timer(batch.device) as timer:
            exported, proposal_report = {}, None
            if config.get("proposal_kind", "native") == "tent":
                from .tent import TentProposal
                proposal = TentProposal(model, learning_rate=config.get("tent_lr", 0.001), steps=config.get("tent_steps", 1))
                exported = proposal.prepare(batch)
                proposal_report = proposal.report
            context = dict(batch.context)
            with torch.no_grad():
                state = model.encode(batch.x, context)
            context["rtt_exported_states"] = exported
            spent = 0.0
            for depth in range(model.T):
                with torch.no_grad():
                    state = model.transition(state, depth, context).detach()
                if depth == stop_before:
                    if observer is not None:
                        observer(state, depth, context)
                    return state, context
                if depth not in self.modules.open_depths or H_plus <= 0:
                    continue
                if observer is not None:
                    observer(state, depth, context)
                kappa = float(radii.get(depth, radii.get(str(depth), float("inf"))))
                if math.isnan(kappa) or kappa < 0:
                    raise ValueError("Uncertainty radii must be nonnegative")
                state, per_unit, record = self._step(
                    batch, state, depth, context, kappa, H_plus - spent,
                    config["H_row"] - row_spend,
                    sum(d >= depth for d in self.modules.open_depths), exported.get(depth))
                steps.append(record)
                if per_unit is not None:
                    row_spend += per_unit
                    spent += record["charge"]
                    accepted += 1
            with torch.no_grad():
                logits = batch.select(model.logits(state, context))
                reference_context = dict(batch.context)
                reference_state = model.encode(batch.x, reference_context)
                for depth in range(model.T):
                    reference_state = model.transition(reference_state, depth, reference_context)
                reference = batch.select(model.logits(reference_state, reference_context))
                if not bool(torch.isfinite(reference).all()):
                    raise FloatingPointError("The incumbent produced nonfinite logits; fallback is unavailable")
                damage, row_damage, _, _ = floor_damage(reference, logits, batch.mask, batch.multilabel)
                consistent = bool(torch.isfinite(logits).all()) and bool(torch.isfinite(reference).all())
                passed = consistent and float(damage) <= H_plus and bool((row_damage[batch.mask] <= config["H_row"]).all())
                passed = passed and spent <= H_plus and bool((row_spend[batch.mask] <= config["H_row"]).all())
                passed = passed and all(row.get("robust_model", -1) <= 0 for row in steps)
                if not passed:
                    logits, state, mode = reference.clone(), reference_state, "F"
                    damage = damage.new_zeros(())
                    row_damage = torch.zeros_like(row_damage)
                else:
                    mode = "R1" if accepted else "NO_STEP"
        report = {"mode": mode, "accepted_steps": accepted if passed else 0,
                  "attempted_accepted_steps": accepted, "steps": steps,
                  "call_damage_bound": float(damage), "row_damage_max": float(row_damage[batch.mask].max()),
                  "sum_tail_charges": spent, "H_plus": H_plus, "H_row": config["H_row"],
                  "checker_passed_before_fallback": passed, "labels_supplied": False,
                  "endpoint_checked": True,
                  "backend": "exact_reference_tail", "separate_full_reference_count": 1,
                  "latency_seconds": timer.seconds, "numerical_status": NUMERICAL_STATUS,
                  "proposal": proposal_report}
        return CallResult(logits.detach(), reference.detach(), state.detach(), reference_state.detach(), context, report)
