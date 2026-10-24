"""Regressions for redundant physical spans and honest optimizer termination."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def rtt():
    path = Path(__file__).resolve().parents[1] / "run_rtt_iclr2027_arrhenius.py"
    spec = importlib.util.spec_from_file_location("rtt_solver_status_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    return module


def test_zero_slope_redundant_rank_two_gauge_bounds_actual_displacements(rtt):
    d0 = torch.tensor([[1.0, 0.0], [0.2, 0.4]], dtype=torch.float64)
    d1 = torch.tensor([[0.0, 1.0], [0.8, 0.3]], dtype=torch.float64)
    D = torch.stack([d0, d1, 0.3 * d0 + 0.7 * d1])  # M=3, physical rank=2
    G_rows = torch.einsum("mnd,pnd->nmp", D, D)
    weights = torch.tensor([0.7, 1.2], dtype=torch.float64)
    gauge = rtt.build_block_gauge(G_rows, weights, np.zeros(3), 64)
    assert gauge.shape == (64, 3) and np.isfinite(gauge).all()
    rng = np.random.default_rng(23)
    for coefficients in rng.standard_normal((40, 3)):
        displacement = torch.einsum("m,mnd->nd", torch.from_numpy(coefficients), D)
        exact_charge = float(weights @ torch.linalg.vector_norm(displacement, dim=1))
        gauge_charge = float((gauge @ coefficients).max())
        assert gauge_charge + 1e-11 >= exact_charge


def _bounds():
    return np.asarray([[-1.0], [1.0]]), np.asarray([1.0, 1.0])


def test_nonfinite_hessian_is_not_optimal_and_keeps_feasible_iterate(rtt):
    A, b = _bounds()

    def bad_hessian(x, order):
        return -float(x[0]), np.asarray([-1.0]), np.asarray([[np.nan]])

    x, info = rtt.barrier_solve(1, bad_hessian, A, b, [], np.asarray([0.0]), 1e-11)
    assert info["status"] == "nonfinite_hessian"
    assert info["feasible"] and np.isfinite(x).all()
    assert info["gap"] is None


def test_failed_line_search_is_reported(rtt):
    A, b = _bounds()

    def inconsistent_oracle(x, order):
        # Deliberately inconsistent values/derivatives: no proposed nonzero step
        # can satisfy Armijo, however small. Solver must report the failure.
        value = 1.0 if x[0] == 0.0 else 2.0
        return value, np.asarray([-1.0]), np.zeros((1, 1))

    x, info = rtt.barrier_solve(1, inconsistent_oracle, A, b, [], np.asarray([0.0]), 1e-11)
    assert info["status"] == "line_search_failed"
    assert info["feasible"] and float(x[0]) == 0.0


def test_newton_iteration_limit_is_not_optimal(rtt):
    A, b = _bounds()

    def quadratic(x, order):
        return float((x[0] - 0.8) ** 2), np.asarray([2 * (x[0] - 0.8)]), np.asarray([[2.0]])

    x, info = rtt.barrier_solve(1, quadratic, A, b, [], np.asarray([0.0]), 1e-20, max_newton=1)
    assert info["status"] == "max_newton_iterations"
    assert info["feasible"] and bool((A @ x < b).all())


def test_outer_iteration_limit_and_nonfinite_start_are_not_optimal(rtt):
    A, b = _bounds()
    flat = lambda x, order: (0.0, np.zeros(1), np.zeros((1, 1)))
    _, info = rtt.barrier_solve(1, flat, A, b, [], np.asarray([0.0]), 1e-20, max_outer=1)
    assert info["status"] == "max_outer_iterations"
    assert info["centered"] and info["feasible"]
    x, info = rtt.barrier_solve(1, flat, A, b, [], np.asarray([np.nan]), 1e-11)
    assert info["status"] == "nonfinite_or_invalid_start"
    assert np.isfinite(x).all() and not info["feasible"]


def _program():
    return {"nv": 1, "blocks": [(0, 1)], "caps": [1.0], "a_t": np.asarray([-1.0]),
            "S_t": np.ones((1, 1)), "Q_t": np.ones((1, 1)), "kappa": 0.0,
            "branch": "gauge", "B": 0.5, "gauge": [np.ones((1, 1))], "omega": 0.9}


def test_nonfinite_solver_coefficients_are_rejected(rtt, monkeypatch):
    monkeypatch.setattr(rtt, "barrier_solve", lambda *args, **kwargs:
                        (np.asarray([np.nan, np.nan]), {"status": "nonfinite_newton_step"}))
    result = rtt.solve_block_program(_program(), rtt.RTTConfig())
    assert bool((result.t == 0).all()) and np.isfinite(result.f) and result.charge == 0.0
    assert result.status.startswith("zero_invalid_solver_iterate")


def test_valid_feasible_approximate_output_is_preserved_with_its_status(rtt, monkeypatch):
    monkeypatch.setattr(rtt, "barrier_solve", lambda *args, **kwargs:
                        (np.asarray([0.1, 0.2]), {"status": "max_newton_iterations", "newton": 1}))
    result = rtt.solve_block_program(_program(), rtt.RTTConfig())
    assert float(result.t[0]) == pytest.approx(0.1)
    assert result.status == "max_newton_iterations" and result.f < 0.0
    assert result.charge == pytest.approx(0.1)


def test_infeasible_approximate_output_is_rejected(rtt, monkeypatch):
    monkeypatch.setattr(rtt, "barrier_solve", lambda *args, **kwargs:
                        (np.asarray([1.2, 1.3]), {"status": "line_search_failed"}))
    result = rtt.solve_block_program(_program(), rtt.RTTConfig())
    assert bool((result.t == 0).all()) and result.charge == 0.0
    assert result.status.startswith("zero_invalid_solver_result")
