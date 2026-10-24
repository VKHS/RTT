"""Serving-path tests: nonzero admission, prefix reuse and whole-call fallback.

All inputs and networks are deterministic and tiny. These checks require no OGB
download, training run, cluster account or deployment labels.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def rtt():
    path = Path(__file__).resolve().parents[1] / "run_rtt_iclr2027_arrhenius.py"
    spec = importlib.util.spec_from_file_location("rtt_deployment_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    return module


@pytest.fixture
def setup(rtt):
    torch.manual_seed(213)
    cfg = rtt.RTTConfig(T=5, open_first=1, b_window=2, hidden=2, enc_hidden=4,
                        att_dim=2, head_hidden=8, bank_size=4, spectral_rank=1,
                        spectral_rank_tight=2, probes_per_sign=2, nq=4,
                        enc_dropout=0.0, head_dropout=0.0, solver_max_newton=60)
    graph = rtt.synthetic_graph(n=40, num_classes=2, num_features=3, seed=13, avg_degree=3)
    roles = rtt.DataRoles(pools={"fit": np.arange(graph.n)}, notes=[])
    env = rtt.BaseEnv(graph, roles, cfg, torch.device("cpu"), suite_seed=8)
    inc = rtt.TanhDiffusionIncumbent(graph.num_features, graph.num_classes, cfg, seed=91)
    with torch.no_grad():
        inc.enc1.weight.zero_()
        inc.enc1.weight[0, 0] = 0.1
        inc.enc1.bias.fill_(1.0)
        inc.enc2.weight.zero_()
        inc.enc2.weight[0, 0] = -0.2
        inc.enc2.bias.copy_(torch.tensor([-0.2, 0.15]))
        inc.head.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        inc.head.bias.zero_()
        inc.W_frozen.copy_(torch.diag(torch.tensor([1.6, 0.8])))
        inc.temperature.fill_(1.0)
    inc.frozen = True
    inc.eval()
    proposal = rtt.NativeProposal(cfg.hidden, cfg, seed=92)
    with torch.no_grad():
        proposal.bank.copy_(torch.tensor([[0.6, 0.15]]).expand(cfg.bank_size, -1))
        proposal.v_att.weight.zero_()
    proposal.eval()
    heads = rtt.RTTHeads(rtt.feature_dim(2, len(cfg.open_depths()), cfg.probes_per_sign),
                         2, len(cfg.open_depths()), cfg)
    with torch.no_grad():
        # A fixed label-free posterior shift creates useful, nonzero local fields.
        heads.post.lins[-1].bias.copy_(torch.tensor([0.8, -0.8]))
    heads.eval()
    caps = np.zeros((cfg.T, cfg.K_roles))
    caps[cfg.open_first:] = 0.25
    registry = {"caps": caps, "open_depths": cfg.open_depths(),
                "gamma": inc.gamma_delta(), "growth": np.ones(cfg.T + 1)}
    system = rtt.RTTSystem(inc, proposal, heads, registry, cfg, torch.device("cpu"))
    env.attach_incumbent(inc)
    checker = rtt.Checker(system)
    return SimpleNamespace(cfg=cfg, graph=graph, env=env, inc=inc, system=system, checker=checker)


def _context(s, component="clean", need_labels=False, index=0):
    spec = s.env.generator.make("serving-test", "fit", component, index, nq=4)
    return s.env.build(spec, need_labels=need_labels)


def _policy(rtt, s, *, nonzero=True, opened=None):
    return rtt.PolicySpec(name="deterministic-serving", kind="rtt",
                          H_plus=0.05, H_row=1.0,
                          open_depths=[s.cfg.T - 1] if opened is None else opened,
                          radii={l: 0.0 if nonzero else float("inf") for l in s.cfg.open_depths()})


def _reference(s, ctx):
    # Independently compute the complete same-call float32 reference. This is
    # deliberately invoked only AFTER the serving assertion in accepted cases.
    with torch.no_grad():
        return s.inc.deployed_logits(ctx.x, ctx.op, ctx=ctx)[ctx.q_idx]


def _trace_transitions(monkeypatch, inc):
    traced = []
    original = inc.transition

    def call(H, H0, op, W, ctx, j, *args, **kwargs):
        traced.append((int(j), H.detach().clone(), H.dtype))
        return original(H, H0, op, W, ctx, j, *args, **kwargs)

    monkeypatch.setattr(inc, "transition", call)
    return traced


def test_nonzero_release_needs_only_prefix_and_no_reference_endpoint(rtt, setup, monkeypatch):
    s = setup
    ctx = _context(s, need_labels=False)
    traced = _trace_transitions(monkeypatch, s.inc)
    with torch.no_grad():
        served, res, cert, timing = rtt.serve(s.system, s.checker, ctx, _policy(rtt, s))
    assert cert.mode in ("R1", "R2"), (cert.mode, cert.first_fail, cert.descent_max)
    assert res.l0 == s.cfg.T - 1 and res.steps
    assert float(res.steps[s.cfg.T - 1].abs().max()) > 1e-5
    assert ctx.y_q is None
    assert ctx.inc_logits is None
    assert len(ctx.inc_states) == s.cfg.T  # H0 through H_{T-1}, no baseline H_T
    # At the final intervention there is no nonempty window reference tail.
    assert [j for j, _, dtype in traced if dtype == torch.float32] == list(range(s.cfg.T))
    reference = _reference(s, ctx)
    assert not torch.equal(served, reference)
    assert float(torch.log_softmax(served.double(), 1)[:, 0].mean()) > float(
        torch.log_softmax(reference.double(), 1)[:, 0].mean())
    price = (torch.log_softmax(reference.double(), 1)
             - torch.log_softmax(served.double(), 1)).max(1).values
    assert float(price.mean()) <= 0.05 and float(price.max()) <= 1.0
    assert timing["fallback_s"] == 0.0
    pieces = (timing["prefix_s"] + timing["exec_s"] + timing["prepare_features_s"]
              + timing["proposal_prepare_s"] + timing["certify_s"] + timing["fallback_s"])
    assert timing["total_s"] == pytest.approx(pieces)
    assert timing["total_with_build_s"] == pytest.approx(pieces + timing["build_s"])


def test_explicit_late_prefix_is_incremental_and_does_not_build_endpoint(rtt, setup):
    s = setup
    ctx = _context(s)
    s.system.prepare_call(ctx, prefix_only=True, prefix_depth=2)
    original_prefix = [x.clone() for x in ctx.inc_states]
    assert len(ctx.inc_states) == 3 and ctx.inc_logits is None
    s.system.prepare_call(ctx, prefix_only=True, prefix_depth=4)
    assert len(ctx.inc_states) == 5 and ctx.inc_logits is None
    for saved, reused in zip(original_prefix, ctx.inc_states):
        assert torch.equal(saved, reused)
    with torch.no_grad():
        _, res, cert, _ = rtt.serve(s.system, s.checker, ctx, _policy(rtt, s, opened=[4]))
    assert res.start_depth == 4 and cert.mode in ("R1", "R2")
    assert ctx.inc_logits is None and len(ctx.inc_states) == 5


def test_absent_or_poisoned_labels_do_not_change_nonzero_serving(rtt, setup):
    s = setup
    outputs = []
    for labels in ("absent", "real", "invalid"):
        ctx = _context(s, need_labels=labels != "absent")
        if labels == "invalid":
            ctx.y_q = torch.full_like(ctx.y_q, 10_000)
        with torch.no_grad():
            served, res, cert, _ = rtt.serve(s.system, s.checker, ctx, _policy(rtt, s))
        assert cert.mode in ("R1", "R2") and res.steps
        assert ctx.inc_logits is None
        outputs.append(served)
    assert all(torch.equal(outputs[0], other) for other in outputs[1:])


@pytest.mark.parametrize("component", ["clean", "noise", "cache", "stress"])
def test_fallback_recomputes_same_call_from_first_intervention(rtt, setup, monkeypatch, component):
    s = setup
    ctx = _context(s, component=component)
    # Make the rejection deterministic after an otherwise valid nonzero path.
    certify = s.checker.certify

    def reject(ctx, res, spec):
        result = certify(ctx, res, spec)
        assert res.l0 == s.cfg.T - 1
        result.mode = "F"
        result.passed = False
        result.first_fail = "test_forced_rejection"
        return result

    monkeypatch.setattr(s.checker, "certify", reject)
    traced = _trace_transitions(monkeypatch, s.inc)
    with torch.no_grad():
        served, res, cert, timing = rtt.serve(s.system, s.checker, ctx, _policy(rtt, s))
    assert cert.mode == "F"
    assert timing["fallback_s"] > 0
    assert ctx.inc_logits is None
    calls = [j for j, _, dtype in traced if dtype == torch.float32]
    assert calls == list(range(s.cfg.T)) + [s.cfg.T - 1]
    fallback_input = [state for j, state, dtype in traced if dtype == torch.float32][-1]
    assert torch.equal(fallback_input, res.states[res.l0])
    assert torch.equal(served, _reference(s, ctx))


@pytest.mark.parametrize("opened", [[], [4], [1, 2, 3, 4]])
def test_no_step_output_equals_same_call_incumbent(rtt, setup, opened):
    s = setup
    ctx = _context(s, component="noise")
    with torch.no_grad():
        served, res, cert, _ = rtt.serve(s.system, s.checker, ctx,
                                       _policy(rtt, s, nonzero=False, opened=opened))
    assert cert.mode == "NO_STEP"
    assert res.l0 is None and not res.steps
    assert ctx.inc_logits is None
    assert torch.equal(served, _reference(s, ctx))


def test_fallback_starts_at_first_actual_step_after_earlier_depths_are_closed(rtt, setup, monkeypatch):
    s = setup
    ctx = _context(s, component="stress")
    spec = _policy(rtt, s, nonzero=False, opened=[1, 2, 3, 4])
    spec.radii[4] = 0.0
    certify = s.checker.certify

    def reject(ctx, res, spec):
        result = certify(ctx, res, spec)
        assert res.start_depth == 1 and res.l0 == 4
        result.mode, result.passed = "F", False
        return result

    monkeypatch.setattr(s.checker, "certify", reject)
    traced = _trace_transitions(monkeypatch, s.inc)
    with torch.no_grad():
        served, res, cert, _ = rtt.serve(s.system, s.checker, ctx, spec)
    assert cert.mode == "F"
    assert len(ctx.inc_states) == 2, "Only the configured shared prefix should be cached."
    calls = [j for j, _, dtype in traced if dtype == torch.float32]
    assert calls == list(range(s.cfg.T)) + [4]
    assert torch.equal(traced[-1][1], res.states[4])
    assert torch.equal(served, _reference(s, ctx))


@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("corruption", ["nan_logit", "wrong_logit", "wrong_state"])
def test_invalid_execution_never_releases_corrupted_predictions(rtt, setup, monkeypatch, nonzero, corruption):
    s = setup
    ctx = _context(s, component="cache")
    execute = s.system.run

    def tamper(*args, **kwargs):
        result = execute(*args, **kwargs)
        if corruption == "wrong_state":
            # Keep logits consistent with a false final state: head-only integrity
            # checks cannot establish that a declared no-step path is unchanged.
            result.states[-1] = result.states[-1].clone() + 10.0
            result.logits_T = s.inc.logits(result.states[-1])
        else:
            result.logits_T = result.logits_T.clone()
            result.logits_T[ctx.q_idx[0], 0] = float("nan") if corruption == "nan_logit" else 100.0
        return result

    monkeypatch.setattr(s.system, "run", tamper)
    with torch.no_grad():
        served, _, cert, _ = rtt.serve(s.system, s.checker, ctx,
                                     _policy(rtt, s, nonzero=nonzero))
    assert cert.mode == "F" and not cert.passed
    assert bool(torch.isfinite(served).all())
    assert torch.equal(served, _reference(s, ctx))


def test_comparators_pay_for_candidate_features_and_cached_plugin(rtt, setup, monkeypatch):
    s = setup
    s.cfg.frontier_grid = 6
    corrector = rtt.OnePassCorrector(s.system.d, s.system.C, 2, True,
                                     2 * s.cfg.probes_per_sign, 8, "rows")
    with torch.no_grad():
        corrector.net.lins[-1].bias.copy_(torch.tensor([0.4, -0.4]))
    candidate_depths = []
    candidates = s.system.proposal.candidates

    def trace_candidates(Hbar, depth, ctx):
        candidate_depths.append(depth)
        return candidates(Hbar, depth, ctx)

    monkeypatch.setattr(s.system.proposal, "candidates", trace_candidates)
    models = {"correctors": {"corr_2hop_rows": corrector}}
    params = {"corrector": "corr_2hop_rows", "H_plus": 0.05, "H_row": 1.0}
    policies = [rtt.PolicyDef("corrector_first", "corrector", params=params),
                rtt.PolicyDef("corrector_second", "corrector", params=params),
                rtt.PolicyDef("mixing_first", "mixing", params={"allocation": "allocated"}),
                rtt.PolicyDef("mixing_second", "mixing", params={"allocation": "allocated"}),
                rtt.PolicyDef("projection", "projection", params={"allocation": "allocated"}),
                rtt.PolicyDef("raw", "raw")]
    evaluator = rtt.Evaluator(s.system, s.checker, s.env, s.cfg, models)
    call = s.env.generator.make("timing-test", "fit", "clean", 0, nq=4)
    with torch.no_grad():
        records = evaluator.evaluate_call(call, policies)
    assert all(record["status"] == "OK" for record in records)
    # Two correctors each compute three row-feature depths; the raw trajectory
    # has four open depths; one reusable plug-in calculation computes three more.
    # Without explicit cache accounting later correctors would use free features.
    assert len(candidate_depths) == 3 + 3 + 4 + 3
    by_name = {record["policy"]: record for record in records}
    plugin_times = [by_name[name]["plugin_s"] for name in ("mixing_first", "mixing_second", "projection")]
    assert plugin_times[0] > 0 and len(set(plugin_times)) == 1
    assert by_name["mixing_first"]["gain"] == by_name["mixing_second"]["gain"]
    assert all(by_name[name]["output_rule_s"] > 0 for name in ("mixing_first", "projection"))
    assert all(record["timing_convention"] == "fresh_call_component_sum_cached_work_recharged"
               for record in records)
    assert all(record["latency_with_build_s"] >= record["latency_s"] > 0 for record in records)


def test_default_corrector_capacity_matches_actual_rtt_head_parameters(rtt):
    cfg = rtt.RTTConfig()
    classes = 40
    heads = rtt.RTTHeads(rtt.feature_dim(classes, len(cfg.open_depths()), cfg.probes_per_sign),
                         classes, len(cfg.open_depths()), cfg)
    actual_head_parameters = sum(parameter.numel() for parameter in heads.parameters())
    for name, hops, rows in rtt.CORRECTOR_SPECS:
        width, target = rtt.matched_corrector_width(cfg.hidden, classes, hops, rows, cfg)
        corrector = rtt.OnePassCorrector(cfg.hidden, classes, hops, rows,
                                         2 * cfg.probes_per_sign, width, name)
        count = sum(parameter.numel() for parameter in corrector.parameters())
        assert target == actual_head_parameters
        assert abs(count - target) / target < 0.005


def test_corrector_fit_uses_the_same_phase_two_episode_draws(rtt, setup, monkeypatch):
    s = setup
    captured = []

    def record_calls(system, env, model, specs, steps, lr, wd, **kwargs):
        captured.extend(specs)
        return {"steps": steps}

    monkeypatch.setattr(rtt, "train_classifier_on_calls", record_calls)
    _, info = rtt.train_correctors(s.system, s.env, s.cfg, steps=3, seed=44,
                                   which=["corr_2hop_rows"])
    expected = [s.env.generator.make("phase2_heads", "fit", "mixture", index) for index in range(3)]
    assert len(captured) == len(expected)
    for actual, reference in zip(captured, expected):
        assert actual.call_id == reference.call_id
        assert actual.component == reference.component
        assert actual.noise_seed == reference.noise_seed
        np.testing.assert_array_equal(actual.query, reference.query)
        np.testing.assert_array_equal(actual.edge_keep, reference.edge_keep)
        np.testing.assert_array_equal(actual.cache_rows, reference.cache_rows)
    assert info["corr_2hop_rows"]["training_call_generator"] == "phase2_heads"
    assert info["corr_2hop_rows"]["steps"] == 3
