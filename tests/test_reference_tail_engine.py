"""Behavioral tests for internal admission and the independent endpoint checker.

These are numerical implementation checks, not benchmark experiments or a
directed-rounding proof. The positive-admission fixture fits an exact field
offline; the engine is subsequently given an explicitly unlabeled batch.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rtt import adaptation
from rtt.adaptation import AdaptationModules, ReferenceTailEngine, true_tail_gradient
from rtt.batching import Batch
from rtt.configuration import DEFAULT_ADAPTATION, DEFAULT_PROTOCOL, DEFAULT_TRAINING, validate_config
from rtt.losses import supervised_loss
from rtt.models import build_model


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def settings(**overrides):
    return {**DEFAULT_ADAPTATION, "roles": 2, "bank_size": 2, "head_hidden": 4,
            "probes": 2, "solver_steps": 24, "line_search_steps": 16,
            "H_plus": 0.025, "H_row": 0.035, "row_trim": 0.8,
            "budget_margin": 1e-8, **overrides}


def make_batch(x, context=None, *, labels=None, mask=None, multilabel=False, query=None):
    count = len(x) if query is None else len(query)
    if labels is not None:
        count = len(labels)
    if mask is not None:
        count = len(mask)
    if mask is None:
        mask = torch.ones(count, dtype=torch.bool, device=x.device)
    return Batch(x=x, context={} if context is None else context,
                 item_ids=np.arange(count), mask=mask, multilabel=multilabel,
                 query=query, labels=labels)


class FrozenFixtureHead(nn.Module):
    """All values are fixed offline; there is no access to labels during serving."""
    def __init__(self, first, field):
        super().__init__()
        self.register_buffer("directions", torch.stack((first, torch.zeros_like(first))))
        self.register_buffer("field", field)
        self.register_buffer("variance", field.new_zeros(len(field)))

    def candidates(self, state, context, row_trim, exported=None):
        return self.directions

    def estimate(self, state, candidates, units):
        return self.field, self.variance


class FixtureModules(nn.Module):
    def __init__(self, head, depth=0):
        super().__init__()
        self.open_depths = [depth]
        self.depths = nn.ModuleDict({str(depth): head})


class AnalyticReference(nn.Module):
    """A controlled two-transition reference for exact floor and kink examples."""
    T = 2

    def __init__(self, *, relu_tail=False):
        super().__init__()
        self.relu_tail = relu_tail

    def encode(self, x, context):
        context["h0"] = x
        return x

    def transition(self, state, depth, context):
        return F.relu(state) if self.relu_tail and depth == 1 else state

    def logits(self, state, context):
        return torch.cat((torch.zeros_like(state), state), dim=1) if self.relu_tail else state

    def tail(self, state, next_depth, context):
        for depth in range(next_depth, self.T):
            state = self.transition(state, depth, context)
        return self.logits(state, context)


def perfect_appnp_case():
    torch.manual_seed(211)
    model = build_model("appnp", 3, 3, {"hidden": 4, "enc_hidden": 5, "T": 3, "alpha": 0.15}).double().freeze()
    x = torch.tensor([[0.2, 0.5, -0.3], [0.8, -0.1, 0.6], [-0.4, 0.7, 0.9], [1.0, 0.3, 0.2]], dtype=torch.float64)
    edge = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    labeled = make_batch(x, {"edge_index": edge}, labels=torch.tensor([0, 1, 2, 0]))
    context = dict(labeled.context)
    with torch.no_grad():
        state = model.transition(model.encode(x, context), 0, context)
    gradient = true_tail_gradient(model, labeled, state, 1, context)
    assert gradient.norm() > 0
    direction = -0.8 * gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    modules = FixtureModules(FrozenFixtureHead(direction, gradient))
    return model, modules, labeled, context


def assert_all_class_floors(result, batch, config):
    """Independently enumerate possible labels for this small multiclass call."""
    reference = result.reference_logits.double().log_softmax(-1)
    delivered = result.logits.double().log_softmax(-1)
    per_class = reference - delivered
    assert bool((per_class[batch.mask] <= config["H_row"] + 1e-12).all())
    possibilities = torch.cartesian_prod(*[torch.arange(reference.shape[1])] * len(reference))
    rows = torch.arange(len(reference)).expand_as(possibilities)
    every_label_loss_change = per_class[rows, possibilities].mean(1)
    worst = float(every_label_loss_change.max())
    assert worst <= config["H_plus"] + 1e-12
    assert result.report["call_damage_bound"] == pytest.approx(worst, abs=1e-12)
    return worst


def test_nonzero_internal_admission_uses_reference_tail_and_preserves_every_label_floor(monkeypatch):
    model, modules, labeled, context = perfect_appnp_case()
    config = settings()
    engine = ReferenceTailEngine(model, modules, config)
    original_step, interventions = engine._step, []

    def capture_step(*args, **kwargs):
        changed, charges, report = original_step(*args, **kwargs)
        if charges is not None:
            interventions.append(changed.detach().clone())
        return changed, charges, report

    monkeypatch.setattr(engine, "_step", capture_step)
    result = engine.serve(labeled.unlabeled(), {0: 0.0})
    assert result.report["mode"] == "R1"
    assert result.report["accepted_steps"] == 1 and len(interventions) == 1
    assert result.report["steps"][0]["robust_model"] < 0
    assert result.report["steps"][0]["charge"] > 0
    assert result.report["checker_passed_before_fallback"]
    assert not torch.equal(result.state, result.reference_state)
    assert not torch.equal(result.logits, result.reference_logits)
    # The released logits must be produced by the true remaining graph updates.
    continued = model.tail(interventions[0], 1, context)
    torch.testing.assert_close(result.logits, continued, rtol=0, atol=0)
    torch.testing.assert_close(result.reference_logits, model(labeled.x, dict(labeled.context)), rtol=0, atol=0)
    before = supervised_loss(result.reference_logits, labeled.labels)
    after = supervised_loss(result.logits, labeled.labels)
    assert after < before
    assert_all_class_floors(result, labeled, config)
    assert 0 < result.report["sum_tail_charges"] <= config["H_plus"]


def test_line_search_respects_tighter_call_and_row_budgets():
    model, modules, labeled, _ = perfect_appnp_case()
    generous = settings(H_plus=2.0, H_row=2.0)
    tight = settings(H_plus=0.001, H_row=0.0015)
    large = ReferenceTailEngine(model, modules, generous).serve(labeled.unlabeled(), {0: 0.0})
    small = ReferenceTailEngine(model, modules, tight).serve(labeled.unlabeled(), {0: 0.0})
    assert large.report["mode"] == small.report["mode"] == "R1"
    assert small.report["steps"][0]["scale"] < large.report["steps"][0]["scale"]
    assert small.report["steps"][0]["row_step_max"] < large.report["steps"][0]["row_step_max"]
    assert_all_class_floors(small, labeled, tight)


def test_labels_never_enter_online_admission():
    model, modules, labeled, _ = perfect_appnp_case()
    with pytest.raises(ValueError, match="unlabeled"):
        ReferenceTailEngine(model, modules, settings()).serve(labeled, {0: 0.0})


def test_nonzero_molecular_admission_handles_128_masked_binary_tasks():
    torch.manual_seed(83)
    model = build_model("contractive_diffusion", 9, 128, {"hidden": 4, "T": 2}).double().freeze()
    x = torch.zeros(4, 9, dtype=torch.long)
    context = {"batch": torch.tensor([0, 0, 1, 1]),
               "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
               "edge_attr": torch.zeros(4, 3, dtype=torch.long)}
    labels = torch.full((2, 128), float("nan"), dtype=torch.float64)
    labels[0, 0], labels[0, 7], labels[1, 127] = 1, 0, 1
    mask = torch.isfinite(labels)
    labeled = make_batch(x, context, labels=labels, mask=mask, multilabel=True)
    work = dict(context)
    with torch.no_grad():
        state = model.transition(model.encode(x, work), 0, work)
    gradient = true_tail_gradient(model, labeled, state, 1, work)
    assert gradient.norm() > 0
    first = -0.4 * gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    modules = FixtureModules(FrozenFixtureHead(first, gradient))
    config = settings()
    result = ReferenceTailEngine(model, modules, config).serve(labeled.unlabeled(), {0: 0.0})
    assert result.report["mode"] == "R1" and result.report["accepted_steps"] == 1
    assert result.logits.shape == (2, 128) and result.state.shape == (4, 4)
    before = supervised_loss(result.reference_logits, labels, True, mask)
    after = supervised_loss(result.logits, labels, True, mask)
    assert after < before and result.report["sum_tail_charges"] > 0
    losses = []
    for outcome in (0.0, 1.0):
        possible_labels = torch.full_like(labels, outcome)
        new = F.binary_cross_entropy_with_logits(result.logits, possible_labels, reduction="none")
        old = F.binary_cross_entropy_with_logits(result.reference_logits, possible_labels, reduction="none")
        losses.append(new - old)
    worst = torch.stack(losses).amax(0)
    assert float(worst[mask].mean()) <= config["H_plus"]
    assert bool((worst[mask] <= config["H_row"]).all())
    assert result.report["call_damage_bound"] == pytest.approx(float(worst[mask].mean()), abs=1e-12)


@pytest.mark.parametrize("radius", [-0.1, -float("inf"), float("nan")])
def test_negative_and_nan_radii_are_rejected(radius):
    model, modules, labeled, _ = perfect_appnp_case()
    with pytest.raises(ValueError, match="nonnegative"):
        ReferenceTailEngine(model, modules, settings()).serve(labeled.unlabeled(), {0: radius})


@pytest.mark.parametrize("budget", [-0.1, float("inf"), float("nan")])
def test_invalid_call_budgets_are_rejected(budget):
    model, modules, labeled, _ = perfect_appnp_case()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ReferenceTailEngine(model, modules, settings()).serve(labeled.unlabeled(), {0: 0.0}, budget=budget)


@pytest.mark.parametrize("key", ["H_plus", "H_row", "covariance_floor", "coefficient_cap", "row_trim", "omega", "kappa_train", "tent_lr"])
def test_nan_adaptation_configuration_is_rejected(key):
    config = {"training": copy.deepcopy(DEFAULT_TRAINING), "protocol": copy.deepcopy(DEFAULT_PROTOCOL),
              "adaptation": settings(), "model": {"name": "appnp"}}
    config["adaptation"][key] = float("nan")
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("violation", ["call", "row", "nonfinite", "robust_predicate"])
def test_independent_checker_rejects_tampered_step_with_dishonest_local_ledger(monkeypatch, violation):
    x = torch.zeros(4, 3, dtype=torch.float64)
    batch = make_batch(x)
    model = AnalyticReference()
    modules = FixtureModules(FrozenFixtureHead(torch.zeros_like(x), torch.zeros_like(x)))
    config = settings(H_plus=0.01 if violation == "call" else 1.0,
                      H_row=0.1 if violation == "row" else 2.0)
    engine = ReferenceTailEngine(model, modules, config)

    def tampered_step(batch, state, depth, context, *unused):
        proposed = state.clone()
        if violation == "call":
            proposed[:, 0] -= 0.4
        elif violation == "row":
            proposed[0, 0] -= 0.4
        elif violation == "nonfinite":
            proposed[0, 0] = float("nan")
        return proposed, torch.zeros_like(batch.mask, dtype=torch.float64), {
            "depth": depth, "status": "ADMITTED", "charge": 0.0,
            "robust_model": 1.0 if violation == "robust_predicate" else -1.0,
        }

    monkeypatch.setattr(engine, "_step", tampered_step)
    result = engine.serve(batch, {0: 0.0})
    assert result.report["mode"] == "F"
    assert not result.report["checker_passed_before_fallback"]
    assert result.report["attempted_accepted_steps"] == 1
    assert result.report["accepted_steps"] == 0
    assert result.report["call_damage_bound"] == result.report["row_damage_max"] == 0
    torch.testing.assert_close(result.logits, x, rtol=0, atol=0)
    torch.testing.assert_close(result.logits, result.reference_logits, rtol=0, atol=0)
    torch.testing.assert_close(result.state, result.reference_state, rtol=0, atol=0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_128_independent_binary_tasks_use_observed_units_and_both_outcomes(monkeypatch, sign):
    x = torch.zeros(2, 128, dtype=torch.float64)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0, 0] = mask[1, 127] = True
    batch = make_batch(x, mask=mask, multilabel=True)
    model = AnalyticReference()
    modules = FixtureModules(FrozenFixtureHead(torch.zeros_like(x), torch.zeros_like(x)))
    config = settings(H_plus=0.02, H_row=0.03)
    engine = ReferenceTailEngine(model, modules, config)

    def tampered_step(batch, state, depth, context, *unused):
        proposed = state.clone()
        proposed[0, 0] = 0.05 * sign
        # A large change to an unobserved task must not act as a softmax class.
        proposed[:, 1] = 100.0
        return proposed, torch.zeros_like(batch.mask, dtype=torch.float64), {
            "depth": depth, "status": "ADMITTED", "charge": 0.0, "robust_model": -1.0,
        }

    monkeypatch.setattr(engine, "_step", tampered_step)
    result = engine.serve(batch, {0: 0.0})
    assert result.report["mode"] == "R1"
    assert result.logits.shape == (2, 128)
    # For z=+/-0.05 versus z=0, the worse binary outcome has this exact damage.
    expected_row = math.log1p(math.exp(0.05)) - math.log(2)
    assert result.report["row_damage_max"] == pytest.approx(expected_row, abs=1e-12)
    assert result.report["call_damage_bound"] == pytest.approx(expected_row / 2, abs=1e-12)
    possible_losses = []
    for outcome in (0.0, 1.0):
        labels = torch.full_like(x, outcome)
        changed_loss = F.binary_cross_entropy_with_logits(result.logits, labels, reduction="none")
        source_loss = F.binary_cross_entropy_with_logits(x, labels, reduction="none")
        possible_losses.append(changed_loss - source_loss)
    worst = torch.stack(possible_losses).amax(0)
    assert bool((worst[mask] <= config["H_row"]).all())
    assert float(worst[mask].mean()) <= config["H_plus"]
    assert float(worst[~mask].max()) > 90  # verifies the mask is material to this example


def test_crossed_relu_kink_has_nonzero_finite_remainder_and_is_not_linearized_away(monkeypatch):
    state = torch.tensor([[-0.1]], dtype=torch.float64)
    direction = torch.tensor([[0.3]], dtype=torch.float64)
    # This deliberately optimistic field would admit the step if the crossed
    # ReLU were treated as having zero remainder because its local JVP is zero.
    head = FrozenFixtureHead(direction, torch.tensor([[-0.01]], dtype=torch.float64))
    model, batch = AnalyticReference(relu_tail=True), make_batch(state)
    config = settings(roles=1, H_plus=2.0, H_row=2.0, line_search_steps=1, rho_x=0.0)
    engine = ReferenceTailEngine(model, FixtureModules(head), config)
    original_jvp, observed = torch.autograd.functional.jvp, []

    def capture_jvp(function, inputs, vector, **kwargs):
        output, tangent = original_jvp(function, inputs, vector, **kwargs)
        observed.append((inputs.detach().clone(), vector.detach().clone(), output.detach(), tangent.detach()))
        return output, tangent

    monkeypatch.setattr(torch.autograd.functional, "jvp", capture_jvp)
    result = engine.serve(batch, {0: 0.0})
    assert observed and result.report["mode"] == "NO_STEP"
    assert result.report["steps"][0]["status"] == "NO_FEASIBLE_DESCENT"
    initial, displacement, before, tangent = observed[0]
    assert float(initial[0, 0]) < 0 < float((initial + displacement)[0, 0])
    torch.testing.assert_close(tangent, torch.zeros_like(tangent), rtol=0, atol=0)
    after = model.tail(initial + displacement, 1, {}).log_softmax(-1)
    remainder = float((before - after + tangent).max())
    analytic = math.log1p(math.exp(float(initial + displacement))) - math.log(2)
    assert remainder == pytest.approx(analytic, abs=1e-12) and remainder > 0.09
    torch.testing.assert_close(result.logits, result.reference_logits, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["appnp", "grand", "residual_tanh_mlp", "gin_virtual", "resnet50"])
def test_infinite_radius_no_step_replays_actual_reference_for_each_task_level(family):
    torch.manual_seed(43)
    config = settings(solver_steps=2)
    if family == "resnet50":
        model = build_model(family, 3, 3, {"pretrained": False}).freeze()
        batch = make_batch(torch.randn(1, 3, 32, 32))
    elif family == "gin_virtual":
        model = build_model(family, 9, 128, {"hidden": 4, "T": 2}).freeze()
        x = torch.zeros(4, 9, dtype=torch.long)
        context = {"batch": torch.tensor([0, 0, 1, 1]),
                   "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
                   "edge_attr": torch.zeros(4, 3, dtype=torch.long)}
        batch = make_batch(x, context, mask=torch.ones(2, 128, dtype=torch.bool), multilabel=True)
    else:
        model = build_model(family, 3, 3, {"hidden": 4, "T": 2}).freeze()
        context = {} if family == "residual_tanh_mlp" else {"edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])}
        batch = make_batch(torch.randn(3, 3), context)
    modules = AdaptationModules(model, config)
    result = ReferenceTailEngine(model, modules, config).serve(batch, {})  # absent radii fail closed to infinity
    assert result.report["mode"] == "NO_STEP" and result.report["accepted_steps"] == 0
    assert result.report["checker_passed_before_fallback"]
    assert result.report["endpoint_checked"] and not result.report["labels_supplied"]
    assert all(step["status"] == "zero_or_infinite_radius" for step in result.report["steps"])
    torch.testing.assert_close(result.logits, result.reference_logits, rtol=0, atol=0)
    torch.testing.assert_close(result.state, result.reference_state, rtol=0, atol=0)
    assert result.report["call_damage_bound"] == result.report["row_damage_max"] == 0
    if family == "gin_virtual":
        assert len(result.state) == 6 and result.logits.shape == (2, 128)
    if family == "resnet50":
        assert result.state.ndim == 4 and result.state.shape[1] == 2048


def test_nonfinite_reference_cannot_be_hidden_by_a_fallback():
    x = torch.tensor([[float("nan"), 0.0]], dtype=torch.float64)
    model = AnalyticReference()
    modules = FixtureModules(FrozenFixtureHead(torch.zeros_like(x), torch.zeros_like(x)))
    with pytest.raises(FloatingPointError, match="incumbent.*nonfinite"):
        ReferenceTailEngine(model, modules, settings()).serve(make_batch(x), {}, budget=0)
