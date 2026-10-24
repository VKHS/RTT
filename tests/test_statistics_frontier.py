"""Tests for budget allocation and audit validity, independent of benchmark results."""
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
    spec = importlib.util.spec_from_file_location("rtt_statistics_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_water_fill_partial_segment_spends_budget(rtt):
    h, gain = rtt.water_fill(torch.tensor([[0.0, 1.0]], dtype=torch.float64),
                             [0.0, 1.0], torch.tensor([1.0], dtype=torch.float64), 0.5)
    assert float(h[0]) == pytest.approx(0.5)
    assert gain == pytest.approx(0.5)


def test_water_fill_handles_ties_zero_weights_and_row_caps(rtt):
    G = torch.tensor([[0.0, 0.5, 1.0]] * 3, dtype=torch.float64)
    w = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    cap = torch.tensor([0.3, 0.8, 1.0], dtype=torch.float64)
    h, gain = rtt.water_fill(G, [0.0, 0.5, 1.0], w, 0.35, hmax=cap)
    assert float(w @ h) == pytest.approx(0.35)
    assert bool((h <= cap + 1e-14).all())
    assert float(h[2]) == 0.0
    assert gain == pytest.approx(0.35)


def test_water_fill_matches_small_linear_program(rtt):
    from scipy.optimize import linprog

    grid = [0.0, 0.2, 0.6, 1.0]
    slopes = np.asarray([[4.0, 2.0, 0.4], [3.0, 1.0, -0.1], [2.5, 1.5, 0.2]])
    widths = np.diff(grid)
    G = np.column_stack([np.zeros(3), np.cumsum(slopes * widths, axis=1)])
    w = np.asarray([0.2, 0.3, 0.5])
    budget = 0.42
    # Concave slopes make the segment relaxation exact: cheap later segments
    # cannot displace an earlier, higher-value segment of the same row.
    c = -(w[:, None] * slopes).reshape(-1)
    costs = np.broadcast_to(w[:, None], slopes.shape).reshape(1, -1)
    bounds = [(0.0, width) for _ in range(3) for width in widths]
    sol = linprog(c, A_ub=costs, b_ub=[budget], bounds=bounds, method="highs")
    assert sol.success
    h, gain = rtt.water_fill(torch.from_numpy(G), grid, torch.from_numpy(w), budget)
    assert float(torch.from_numpy(w) @ h) <= budget + 1e-14
    assert gain == pytest.approx(-sol.fun, abs=2e-12)


def test_horizon_free_betting_prefix_does_not_read_future_outcomes(rtt):
    first = np.asarray([-0.012, -0.006, 0.002, -0.008, -0.011, -0.003, -0.007, -0.01])
    _, short_path = rtt.betting_ucb(first, 0.05, 0.05, mode="horizon_free", return_path=True)
    _, long_path = rtt.betting_ucb(np.concatenate([first, [-2.0, -0.8, 0.04, 0.01]]),
                                   0.05, 0.05, mode="horizon_free", return_path=True)
    assert dict(short_path)[8] == pytest.approx(dict(long_path)[8], abs=1e-14)


def test_kl_lambda_grid_is_independent_of_evaluated_data(rtt):
    a = rtt.kl_robust_curve(np.asarray([-0.001, -0.002, 0.001]), 0.05, 0.05, [0.0])
    b = rtt.kl_robust_curve(np.asarray([-3.0, 0.04, -0.4]), 0.05, 0.05, [0.0])
    assert a[0]["lambda_grid"] == b[0]["lambda_grid"]
    with pytest.raises(rtt.RTTFailure):
        rtt.kl_robust_curve([0.06], 0.05, 0.05, [0.0])


def test_component_identity_does_not_pool_mixture_draws(rtt):
    mixture = {"call_id": "audit:final19:mixture:23", "component": "noise"}
    independent = {"call_id": "audit:final19:noise:23", "component": "noise"}
    assert rtt.audit_generator_component(mixture) == "mixture"
    assert rtt.audit_generator_component(independent) == "noise"


def test_failed_call_prevents_statistical_claim_from_survivors(rtt):
    ok = {"status": "OK", "kind": "rtt", "gain": 0.01, "down": -0.01,
          "latency_ratio": 1.4, "acc_change_pp": 0.0, "acc_r": 0.7,
          "acc_s": 0.7, "neg_flip_frac": 0.0, "H_plus": 0.05,
          "violation_row": True}
    failed = {"status": "ERROR", "kind": "rtt", "error": "injected failure"}
    summary = rtt.summarize_policy([ok, failed], rtt.RTTConfig(), 0.05)
    assert summary["status"] == "INCOMPLETE"
    assert summary["n_attempted"] == 2 and summary["n_failed"] == 1
    assert summary["betting_lb"] is None and summary["u"] is None
    assert summary["ci_lo"] is None and summary["ci_hi"] is None
    assert summary["violations_row"] == 1
    assert summary["certificate_status"] == rtt.CERTIFICATE_STATUS


def test_reserved_node_audit_reads_only_the_inspection_sample(rtt, monkeypatch, tmp_path):
    reserved = np.arange(8, dtype=np.int64)
    accessed = []
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    logits = torch.nn.functional.one_hot(labels, 2).float() * 2.0

    class ObservedLabels:
        def __getitem__(self, index):
            accessed.extend(index.cpu().tolist())
            return labels[index]

    class Env:
        device = torch.device("cpu")
        y = ObservedLabels()

        @staticmethod
        def build(sp, need_labels=True):
            assert need_labels is False, "Whole-population prediction must be label-free"
            return SimpleNamespace(q_idx=torch.from_numpy(sp.query), inc_logits=logits, y_q=None)

        generator = SimpleNamespace(make=lambda *args: SimpleNamespace(query=None, nq=None))

    L = SimpleNamespace(env=Env(), roles=SimpleNamespace(pools={"reserved19": reserved}),
                        system=SimpleNamespace(prepare_call=lambda ctx: None), checker=None,
                        models={"correctors": {"corr_2hop_rows": object()}})
    cfg = rtt.RTTConfig(nq=4, node_inspect=4)
    policy = rtt.PolicySpec(name="test", H_plus=0.05, H_row=1.0)
    monkeypatch.setattr(rtt, "load_system", lambda *args, **kwargs: L)
    monkeypatch.setattr(rtt, "registered_policy", lambda *args: (policy, "SELECTED"))
    monkeypatch.setattr(rtt, "serve", lambda *args, **kwargs: (logits[args[2].q_idx], None, None, {}))
    monkeypatch.setattr(rtt, "corrector_serve", lambda *args, **kwargs:
                        (torch.softmax(logits[args[2].q_idx].double(), 1), None, None))
    out = rtt.task_node_transfer(None, None, cfg, torch.device("cpu"), "per_node", tmp_path)
    assert len(accessed) == 4 and set(accessed) == set(out["inspected_node_ids"])
    assert out["uninspected_labels_accessed"] is False
    assert "diagnostic_full_population_flip_rate" not in out["rtt_registered"]
    assert (tmp_path / "reserved_predictions.npz").is_file()
    assert out["rtt_registered"]["inspected_flips"] == 0
