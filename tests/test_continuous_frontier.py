"""Independent optimization checks for the analytic continuous floor frontier."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def frontier():
    path = Path(__file__).resolve().parents[1] / "rtt_frontier.py"
    spec = importlib.util.spec_from_file_location("rtt_continuous_frontier_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_binary_row_has_analytic_solution(frontier):
    q = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    pr = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
    h, p, gain, meta = frontier.continuous_floor_frontier(q, pr, [1.0], 0.1, 1.0)
    expected = torch.tensor([[1.0 - 0.4 * math.exp(-0.1), 0.4 * math.exp(-0.1)]], dtype=torch.float64)
    torch.testing.assert_close(h, torch.tensor([0.1], dtype=torch.float64), atol=1e-10, rtol=0)
    torch.testing.assert_close(p, expected, atol=1e-10, rtol=0)
    assert gain == pytest.approx(float((q * torch.log(expected / pr)).sum()), abs=1e-10)
    assert meta["converged"]


def test_identical_rows_receive_equal_marginal_allocations(frontier):
    q = torch.tensor([[0.8, 0.2]] * 3, dtype=torch.float64)
    pr = torch.tensor([[0.5, 0.5]] * 3, dtype=torch.float64)
    h, p, _, meta = frontier.continuous_floor_frontier(q, pr, [0.2, 0.3, 0.5], 0.2, 1.0)
    torch.testing.assert_close(h, torch.full((3,), 0.2, dtype=torch.float64), atol=1e-10, rtol=0)
    assert meta["duality_gap_estimate"] < 1e-10


def test_zero_posterior_components_and_zero_weight_rows(frontier):
    q = [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]]
    pr = [[0.5, 0.5]] * 3
    h, p, gain, meta = frontier.continuous_floor_frontier(q, pr, [0.0, 0.5, 0.5], 0.7, float("inf"))
    torch.testing.assert_close(h, torch.tensor([0.0, 1.4, 0.0], dtype=torch.float64), atol=1e-9, rtol=0)
    assert float(p[1, 1]) == pytest.approx(0.5 * math.exp(-1.4), abs=1e-10)
    assert gain == pytest.approx(0.5 * math.log((1.0 - 0.5 * math.exp(-1.4)) / 0.5), abs=1e-10)
    assert meta["converged"]


def test_saturation_spends_only_the_needed_budget(frontier):
    q = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    pr = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
    h, p, gain, meta = frontier.continuous_floor_frontier(q, pr, [1.0], 5.0, 10.0)
    assert float(h[0]) == pytest.approx(math.log(2.0))
    torch.testing.assert_close(p, q, atol=2e-15, rtol=0)
    assert gain == pytest.approx(float((q * torch.log(q / pr)).sum()))
    assert meta["multiplier"] == 0.0 and meta["duality_gap_estimate"] == 0.0
    assert meta["row_caps_or_posterior_saturation_reached"]


def test_zero_budget_and_row_caps(frontier):
    q = [[0.95, 0.05], [0.7, 0.3]]
    pr = [[0.4, 0.6], [0.4, 0.6]]
    h0, p0, g0, m0 = frontier.continuous_floor_frontier(q, pr, [0.5, 0.5], 0.0, [0.1, 0.2])
    assert bool((h0 == 0).all()) and g0 == 0.0
    torch.testing.assert_close(p0, torch.tensor(pr, dtype=torch.float64))
    h, p, gain, meta = frontier.continuous_floor_frontier(q, pr, [0.5, 0.5], 2.0, [0.1, 0.2])
    torch.testing.assert_close(h, torch.tensor([0.1, 0.2], dtype=torch.float64))
    assert meta["allocated_spend"] == pytest.approx(0.15)
    assert meta["budget_violation"] == 0.0 and meta["converged"]


def _direct_slsqp(q, pr, w, budget, caps):
    """Optimize p and h directly, without row active sets or dual inversion."""
    from scipy.optimize import minimize

    n, c = q.shape
    npv = n * c

    def fun(x):
        return -float((w[:, None] * q * np.log(x[:npv].reshape(n, c))).sum())

    def jac(x):
        return np.r_[-(w[:, None] * q / x[:npv].reshape(n, c)).ravel(), np.zeros(n)]

    def floor_constraint(x):
        return (np.log(x[:npv].reshape(n, c)) + x[npv:, None] - np.log(pr)).ravel()

    def floor_jac(x):
        mat = np.zeros((npv, npv + n))
        mat[np.arange(npv), np.arange(npv)] = 1.0 / x[:npv]
        mat[np.arange(npv), npv + np.repeat(np.arange(n), c)] = 1.0
        return mat

    equality_jac = np.zeros((n, npv + n))
    for i in range(n):
        equality_jac[i, i * c:(i + 1) * c] = 1.0
    start_h = np.minimum(caps, budget * 0.5)
    sol = minimize(fun, np.r_[pr.ravel(), start_h], jac=jac, method="SLSQP",
                   bounds=[(1e-12, 1.0)] * npv + [(0.0, float(cap)) for cap in caps],
                   constraints=[
                       {"type": "eq", "fun": lambda x: x[:npv].reshape(n, c).sum(1) - 1.0,
                        "jac": lambda x: equality_jac},
                       {"type": "ineq", "fun": floor_constraint, "jac": floor_jac},
                       {"type": "ineq", "fun": lambda x: budget - w @ x[npv:],
                        "jac": lambda x: np.r_[np.zeros(npv), -w]},
                   ], options={"ftol": 1e-12, "maxiter": 2000})
    assert sol.success, sol.message
    p = sol.x[:npv].reshape(n, c)
    return float((w[:, None] * q * np.log(p / pr)).sum()), p


@pytest.mark.parametrize("seed", [3, 17, 121])
def test_frontier_matches_direct_convex_program(seed, frontier):
    rng = np.random.default_rng(seed)
    q = rng.dirichlet(np.ones(4) * 1.2, size=3)
    pr = rng.dirichlet(np.ones(4) * 1.5, size=3)
    w = np.asarray([0.2, 0.3, 0.5])
    caps = np.asarray([0.15, 0.8, 0.45])
    budget = 0.18
    h, p, gain, meta = frontier.continuous_floor_frontier(q, pr, w, budget, caps)
    independent_gain, independent_p = _direct_slsqp(q, pr, w, budget, caps)
    assert gain == pytest.approx(independent_gain, abs=3e-8)
    np.testing.assert_allclose(p.numpy(), independent_p, atol=2e-5, rtol=0)
    assert float(torch.from_numpy(w) @ h) <= budget + 1e-13
    assert meta["max_probability_floor_violation"] <= 1e-15
    assert meta["max_probability_normalization_residual"] <= 1e-14
    assert meta["converged"]


def test_projection_handles_zero_and_infinite_budgets(frontier):
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.7, 0.3]], dtype=torch.float64)
    pr = torch.tensor([[0.5, 0.5]] * 3, dtype=torch.float64)
    p = frontier.project_probability_floor(q, pr, [0.0, 0.3, float("inf")])
    torch.testing.assert_close(p[0], pr[0])
    assert float(p[1, 1]) == pytest.approx(0.5 * math.exp(-0.3))
    torch.testing.assert_close(p[2], q[2])


def test_invalid_probabilities_are_rejected(frontier):
    with pytest.raises(ValueError, match="strictly positive"):
        frontier.continuous_floor_frontier([[0.8, 0.2]], [[1.0, 0.0]], [1.0], 0.1)
    with pytest.raises(ValueError, match="normalized"):
        frontier.continuous_floor_frontier([[0.8, 0.3]], [[0.5, 0.5]], [1.0], 0.1)
