"""Mathematical regression tests for RTT's reconstructed arxiv implementation.

These small, independent numerical checks exercise actual continuation maps.
They are not evidence that paper-scale metrics have been reproduced or that
every floating-point kernel satisfies a universal interval-arithmetic proof.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def rtt():
    path = Path(__file__).resolve().parents[1] / "run_rtt_iclr2027_arrhenius.py"
    spec = importlib.util.spec_from_file_location("rtt_math_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    return module


def _op(rtt):
    rows = torch.tensor([0, 0, 1, 2, 2, 2, 3])
    cols = torch.tensor([1, 2, 2, 0, 1, 3, 1])
    return rtt.GraphOp(4, rows, cols, torch.device("cpu"))


def test_transport_normalizes_rows_and_uses_actual_transpose(rtt):
    op = _op(rtt)
    expected = torch.tensor(
        [[1 / 3, 1 / 3, 1 / 3, 0], [0, 1 / 2, 1 / 2, 0],
         [1 / 4, 1 / 4, 1 / 4, 1 / 4], [0, 1 / 2, 0, 1 / 2]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(op.mm(torch.eye(4, dtype=torch.float64)), expected)
    torch.testing.assert_close(op.mm(torch.ones(4, dtype=torch.float64)), torch.ones(4, dtype=torch.float64))
    x = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 7
    y = torch.flip(x, [0]) - 0.2
    torch.testing.assert_close((op.mm(x) * y).sum(), (x * op.tmm(y)).sum())
    assert not torch.allclose(expected, expected.T), "The test must distinguish P from P.T."


def _tiny_trajectory(rtt):
    generator = torch.Generator().manual_seed(183)
    op = _op(rtt)
    P = op.mm(torch.eye(4, dtype=torch.float64))
    W = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    W = W * (1.6 / torch.linalg.matrix_norm(W, 2))
    H0 = torch.randn(4, 3, generator=generator, dtype=torch.float64) * 1.1
    A = torch.randn(3, 3, generator=generator, dtype=torch.float64) * 0.7
    a, tau, T = 0.1, 0.9, 4

    def transition(H):
        return a * H0 + (1 - a) * ((1 - tau) * H + tau * torch.tanh(P @ H @ W))

    v = [0.025 * torch.randn(4, 3, generator=generator, dtype=torch.float64) for _ in range(T)]
    H, bars = [H0], []
    for step in v:
        bars.append(transition(H[-1]))
        H.append(bars[-1] + step)
    split = rtt.SpectralSplit(W, k=2)
    radii = [torch.zeros(4, dtype=torch.float64)]
    L, mu = [], []
    for j in range(T):
        lj, mj, _ = rtt.interval_factors(P @ H[j] @ W, P @ radii[-1], split,
                                        "tanh", torch.float64)
        L.append(lj)
        mu.append(mj)
        radii.append(rtt.tube_step(op, radii[-1], lj, None,
                                  torch.linalg.vector_norm(v[j], dim=1), a, tau))
    return SimpleNamespace(P=P, op=op, W=W, H0=H0, A=A, alpha=a, tau=tau, T=T,
                           H=H, bars=bars, v=v, radii=radii, L=L, mu=mu,
                           transition=transition, generator=generator)


def test_tube_encloses_incumbent_and_every_sampled_segment_continuation(rtt):
    ex = _tiny_trajectory(rtt)
    incumbent = ex.H0
    for j in range(ex.T):
        incumbent = ex.transition(incumbent)
        error = torch.linalg.vector_norm(incumbent - ex.H[j + 1], dim=1)
        assert bool((error <= ex.radii[j + 1] + 1e-12).all())
    # Independent dense tails have no later intervention. Include both endpoints
    # and interior segment points; checking only the adapted path would miss this.
    for l in range(ex.T):
        for fraction in (0.0, 0.13, 0.5, 0.87, 1.0):
            counterfactual = ex.bars[l] + fraction * ex.v[l]
            for j in range(l + 1, ex.T + 1):
                distance = torch.linalg.vector_norm(counterfactual - ex.H[j], dim=1)
                assert bool((distance <= ex.radii[j] + 1e-12).all()), (l, fraction, j)
                if j < ex.T:
                    counterfactual = ex.transition(counterfactual)


def test_interval_jacobian_dominates_points_inside_tube(rtt):
    ex = _tiny_trajectory(rtt)
    for j in range(ex.T):
        for _ in range(6):
            delta = torch.randn(4, 3, generator=ex.generator, dtype=torch.float64)
            delta = delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True)
            delta = 0.9 * ex.radii[j].unsqueeze(1) * delta
            z = ex.P @ (ex.H[j] + delta) @ ex.W
            slopes = 1 - torch.tanh(z) ** 2
            for i in range(4):
                jacobian = ex.W * slopes[i].unsqueeze(0)
                exact = torch.linalg.matrix_norm(jacobian, 2)
                assert float(exact) <= float(ex.L[j][i]) + 1e-11


def test_first_and_second_order_contracts_against_tail_derivatives(rtt):
    ex = _tiny_trajectory(rtt)
    weights = torch.tensor([0.65, 0.0, 0.35, 0.0], dtype=torch.float64)
    labels = torch.tensor([0, 2, 1, 0])
    gamma = float(torch.cdist(ex.A.T, ex.A.T).max())
    lambdas, xis = rtt.contracts_backward(
        ex.op, ex.T, 0, lambda j: ex.L[j], lambda j: ex.mu[j], lambda j: None,
        weights, gamma, ex.alpha, ex.tau,
    )

    def loss_at_depth(H, depth):
        for _ in range(depth, ex.T):
            H = ex.transition(H)
        logp = torch.log_softmax(H @ ex.A, dim=1)
        return -(weights * logp[torch.arange(4), labels]).sum()

    for depth in range(ex.T + 1):
        # A noncentral point tests tube factors, not only derivatives at its center.
        shift = torch.randn(4, 3, generator=ex.generator, dtype=torch.float64)
        shift /= torch.linalg.vector_norm(shift, dim=1, keepdim=True)
        H = (ex.H[depth] + 0.3 * ex.radii[depth].unsqueeze(1) * shift).requires_grad_(True)
        direction = torch.randn(4, 3, generator=ex.generator, dtype=torch.float64)
        direction /= torch.linalg.vector_norm(direction)
        value = loss_at_depth(H, depth)
        gradient = torch.autograd.grad(value, H, create_graph=True)[0]
        first = (gradient * direction).sum()
        hessian_direction = torch.autograd.grad(first, H)[0]
        second = (hessian_direction * direction).sum()
        first_bound = (lambdas[depth] * torch.linalg.vector_norm(direction, dim=1)).sum()
        second_bound = (xis[depth] * (direction ** 2).sum(1)).sum()
        assert abs(float(first.detach())) <= float(first_bound) + 1e-11
        assert abs(float(second.detach())) <= float(second_bound) + 1e-10
        h = 2e-4
        numeric_second = (loss_at_depth(H.detach() + h * direction, depth)
                          - 2 * value.detach()
                          + loss_at_depth(H.detach() - h * direction, depth)) / h ** 2
        torch.testing.assert_close(numeric_second, second.detach(), atol=8e-8, rtol=2e-4)


def test_contract_charges_dominate_endpoint_worst_label_damage(rtt):
    ex = _tiny_trajectory(rtt)
    weights = torch.tensor([0.65, 0.0, 0.35, 0.0], dtype=torch.float64)
    gamma = float(torch.cdist(ex.A.T, ex.A.T).max())
    lambdas, _ = rtt.contracts_backward(
        ex.op, ex.T, 0, lambda j: ex.L[j], lambda j: ex.mu[j], lambda j: None,
        weights, gamma, ex.alpha, ex.tau,
    )
    ref = ex.H0
    for _ in range(ex.T):
        ref = ex.transition(ref)
    difference = torch.log_softmax(ref @ ex.A, 1) - torch.log_softmax(ex.H[-1] @ ex.A, 1)
    endpoint = (weights * difference.max(1).values).sum()
    charged = sum((lambdas[l + 1] * torch.linalg.vector_norm(ex.v[l], dim=1)).sum()
                  for l in range(ex.T))
    assert float(endpoint) <= float(charged) + 1e-11
    row_bound = gamma * ex.radii[-1]
    assert bool((difference.max(1).values <= row_bound + 1e-11).all())


def test_loss_telescopes_through_reference_tails(rtt):
    ex = _tiny_trajectory(rtt)
    labels = torch.tensor([0, 1, 2, 1])
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

    def V(H, start):
        for _ in range(start, ex.T):
            H = ex.transition(H)
        return -(weights * torch.log_softmax(H @ ex.A, 1)[torch.arange(4), labels]).sum()

    pieces = sum(V(ex.H[l + 1], l + 1) - V(ex.bars[l], l + 1) for l in range(ex.T))
    endpoint = V(ex.H[-1], ex.T) - V(ex.H0, 0)
    torch.testing.assert_close(pieces, endpoint, rtol=1e-10, atol=1e-12)


def test_support_features_are_invariant_to_posttrim_hull_operations(rtt):
    rng = torch.Generator().manual_seed(77)
    D = torch.randn(2, 5, 3, generator=rng, dtype=torch.float64)
    probes = torch.randn(3, 11, generator=rng, dtype=torch.float64)
    base = rtt.support_features(D, probes)
    variants = [D.flip(0), torch.cat([D, D[:1]], 0),
                torch.cat([D, (0.2 * D[0] + 0.5 * D[1]).unsqueeze(0)], 0)]
    for candidate_hull in variants:
        torch.testing.assert_close(rtt.support_features(candidate_hull, probes), base)


def _gauge_problem(rtt, D):
    # One physical row, one role. Additional columns may be redundant vertices.
    D = np.asarray(D, dtype=np.float64)
    G = D.T @ D
    slope = D.T @ np.array([-1.1, -0.6])
    gauges = rtt.build_block_gauge(torch.from_numpy(G[None]),
                                   torch.ones(1, dtype=torch.float64), slope, 64)
    return dict(nv=D.shape[1], B=0.27, branch="gauge", blocks=[(0, D.shape[1])],
                caps=[1.0], a_t=slope, S_t=0.02 * G, Q_t=0.6 * G,
                kappa=0.2, gauge=[gauges])


def test_optimizer_is_stable_under_redundant_candidate_parameterization(rtt):
    cfg = rtt.RTTConfig(solver_max_newton=80)
    original = np.eye(2)
    variants = [original, original[:, ::-1],
                np.column_stack([original, original[:, 0]]),
                np.column_stack([original, np.array([0.2, 0.5])])]
    physical_steps = []
    for D in variants:
        solved = rtt.solve_block_program(_gauge_problem(rtt, D), cfg)
        assert solved.charge <= 0.27 + 1e-9
        assert solved.f < -0.1, "Invariance must exercise a nonzero admitted solution."
        physical_steps.append(D @ solved.t)
    for step in physical_steps[1:]:
        np.testing.assert_allclose(step, physical_steps[0], atol=3e-5, rtol=3e-5)


def test_rank_zero_and_infinite_calibration_radius_return_zero(rtt):
    zero_gram = torch.zeros(2, 2, dtype=torch.float64)
    assert rtt._block_x_coords(zero_gram, [torch.zeros(2)], [zero_gram]) is None
    assert rtt.build_block_gauge(zero_gram[None], torch.ones(1), np.zeros(2), 16) is None
    for nv, budget, kappa in ((0, 0.1, 0.0), (2, 0.0, 0.0), (2, 0.1, math.inf)):
        result = rtt.solve_block_program(dict(nv=nv, B=budget, kappa=kappa, branch="gauge"), rtt.RTTConfig())
        np.testing.assert_array_equal(result.t, np.zeros(nv))
        assert result.charge == 0 and result.f == 0


def test_terminal_charge_equals_worst_all_label_loss_and_remainder_is_kl(rtt):
    z = torch.tensor([[2.0, -1.0, 0.2], [-0.8, 1.1, 0.4]], dtype=torch.float64)
    U = torch.tensor([[[0.3, 0.2], [-0.1, 0.5], [0.4, -0.3]],
                      [[0.2, -0.1], [-0.5, 0.2], [0.1, 0.4]]], dtype=torch.float64)
    w = torch.tensor([0.3, 0.7], dtype=torch.float64)
    p = torch.softmax(z, 1)
    t = np.array([0.7, 0.2])
    heavy = rtt.HeavyTerms(U, w, p=p, z=z, beta=2000)
    dz = torch.einsum("icv,v->ic", U, torch.from_numpy(t))
    log_ref = torch.log_softmax(z, 1)
    log_new = torch.log_softmax(z + dz, 1)
    # Exhaust all 3^2 label vectors independently of the charge's min-logit formula.
    enumerated = [float(w[0] * (log_ref[0, a] - log_new[0, a])
                        + w[1] * (log_ref[1, b] - log_new[1, b]))
                  for a in range(3) for b in range(3)]
    assert heavy.exact_cplus(t) == pytest.approx(max(enumerated), abs=1e-12)
    expected_kl = float((w * (p * (log_ref - log_new)).sum(1)).sum())
    assert heavy.remainder(t, 0)[0] == pytest.approx(expected_kl, abs=1e-12)
    for surrogate, exact in ((heavy.cplus(t, 0)[0], heavy.exact_cplus(t)),
                             (heavy.env(t, 0)[0], heavy.exact_env_linear(t))):
        assert exact - 1e-12 <= surrogate <= exact + math.log(3) / 2000 + 1e-12


@pytest.mark.parametrize("branch", ["gauge", "env", "cplus"])
def test_each_convex_charge_branch_returns_feasible_descent(rtt, branch):
    z = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    p = torch.softmax(z, 1)
    heavy = rtt.HeavyTerms(torch.tensor([[[0.8], [-0.8]]], dtype=torch.float64),
                          torch.ones(1, dtype=torch.float64), p=p, z=z, beta=10000)
    prob = dict(nv=1, B=0.2, branch=branch, blocks=[(0, 1)], caps=[0.8],
                a_t=np.array([-0.7]), S_t=np.array([[0.01]]), Q_t=np.array([[0.5]]),
                kappa=0.1, gauge=[np.array([[1.6], [-1.6]])], heavy=heavy,
                Qe=np.array([[0.15]]), last=(branch == "cplus"), Qr=np.array([[0.0012]]), omega=0.9)
    solved = rtt.solve_block_program(prob, rtt.RTTConfig())
    assert 0 < solved.t[0] <= 0.8 + 1e-10
    assert solved.f < 0
    assert 0 <= solved.charge <= 0.2 + 1e-10
    if branch == "cplus":
        assert solved.charge == pytest.approx(heavy.exact_cplus(solved.t), abs=1e-12)
    elif branch == "env":
        assert solved.charge == pytest.approx(heavy.exact_env_linear(solved.t)
                                              + 0.5 * float(solved.t @ prob["Qe"] @ solved.t), abs=1e-12)


def test_finite_sample_conformal_quantile_has_infinite_small_sample_case(rtt):
    scores = [4.0, 1.0, 8.0, 2.0]
    assert rtt.conformal_quantile(scores, 0.4) == 4.0  # ceil(.6*5)=3, no interpolation
    assert rtt.conformal_quantile(scores, 0.2) == 8.0
    assert math.isinf(rtt.conformal_quantile(scores, 0.1))
    assert math.isinf(rtt.conformal_quantile([], 0.2))
    assert math.isinf(rtt.conformal_quantile(scores, 0.0))


def test_sequential_calibration_freezes_upstream_radii_and_new_depths_get_fresh_calls(rtt, monkeypatch):
    cfg = rtt.RTTConfig(T=2, open_first=0, K_roles=1)
    calls = []
    observed_policies = []

    class Generator:
        def make(self, stage, pool, component, index):
            calls.append((stage, index))
            return SimpleNamespace(stage=stage, index=index)

    class System:
        def caps_for(self, policy):
            return np.ones((2, 1))

        def prepare_call(self, ctx, time_prefix=False, **kwargs):
            return None

        def run(self, ctx, policy, hook, stop_after=None, keep_records=False):
            observed_policies.append((stop_after, dict(policy.radii)))
            assert math.isinf(policy.radii[stop_after])
            hook({"l": stop_after, "index": ctx.index, "prior": policy.radii.get(0, 0.0)})

    monkeypatch.setattr(rtt, "calibration_score", lambda system, ctx, info:
                        float(info["index"]) + (info["prior"] if info["l"] == 1 else 0.0))
    member = SimpleNamespace(name="fixed", radii_mode="trajectory", delta_cov=0.5,
                             H_plus=lambda config: config.H_plus)
    env = SimpleNamespace(generator=Generator(), build=lambda spec: spec)
    result = rtt.calibrate_member(System(), env, cfg, member, "caps", {0: 0.5, 1: 0.5},
                                  n_r=12, stage_tag="independent")
    assert result["radii"]["0"] == 9.0
    assert result["radii"]["1"] == 18.0
    assert all(radii[0] == 9.0 for depth, radii in observed_policies if depth == 1)
    assert len(calls) == len(set(calls)) == 24
    assert {stage for stage, _ in calls} == {"independent:d0", "independent:d1"}
