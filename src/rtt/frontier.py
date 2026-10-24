"""Continuous probability-floor frontier from RTT, Theorem 2(i).

This module implements the analytic active sets and continuous budget multiplier,
without a budget grid. Inputs ``q`` are a supplied conditional class distribution
or a plug-in estimate. Optimality for that supplied distribution does not imply
that an estimated posterior is the true Bayes law or that its gain is an empirical
performance guarantee. Computations use float64 and report numerical residuals;
they do not constitute formally verified interval arithmetic.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch


def _probabilities(q, p_ref):
    device = q.device if isinstance(q, torch.Tensor) else (
        p_ref.device if isinstance(p_ref, torch.Tensor) else torch.device("cpu"))
    q = torch.as_tensor(q, dtype=torch.float64, device=device)
    p_ref = torch.as_tensor(p_ref, dtype=torch.float64, device=device)
    if q.ndim != 2 or q.shape != p_ref.shape or min(q.shape) < 1:
        raise ValueError("q and p_ref must have the same nonempty [rows, classes] shape")
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(p_ref).all()):
        raise ValueError("Probabilities must be finite")
    if bool((q < 0).any()) or bool((p_ref <= 0).any()):
        raise ValueError("q must be nonnegative and p_ref strictly positive")
    sq, sr = q.sum(1), p_ref.sum(1)
    error = max(float((sq - 1).abs().max()), float((sr - 1).abs().max()))
    if error > 1e-5:
        raise ValueError("Rows must be normalized probabilities (tolerance 1e-5)")
    # Accept float32 softmax inputs while doing the optimization in float64.
    q = q / sq[:, None]
    p_ref = p_ref / sr[:, None]
    return q, p_ref, error


def _row_caps(H_row, n: int, device):
    caps = torch.as_tensor(H_row, dtype=torch.float64, device=device)
    if caps.ndim == 0:
        caps = caps.expand(n)
    if caps.shape != (n,) or bool(torch.isnan(caps).any()) or bool((caps < 0).any()):
        raise ValueError("H_row must be a nonnegative scalar or one cap per row")
    return caps


def _active_sets(q, p_ref):
    # Logs avoid overflow in q/pr; zero posterior classes have ratio -infinity.
    log_ratio = torch.log(q) - torch.log(p_ref)
    order = torch.argsort(log_ratio, dim=1, stable=True)
    ratios = log_ratio.gather(1, order)
    qs = q.gather(1, order)
    rs = p_ref.gather(1, order)
    A, B = rs.cumsum(1), qs.cumsum(1)
    # These terminal masses are exactly one in the mathematical simplex.
    A[:, -1] = 1.0
    B[:, -1] = 1.0
    return ratios, qs, rs, A, B


def _budgets_at_log_multiplier(log1p_lambda: float, ratios, A, B, caps, positive):
    """Argmax_h [G_i(h) - lambda*h] for each row, with its cap.

    With floor-active prefix masses A=sum(pr), B=sum(q), the active threshold is
    theta=(lambda+B)/A and exp(-h)=(lambda+B)/((lambda+1)*A).
    Working in log(1+lambda) avoids overflowing lambda or q/pr.
    """
    t = float(log1p_lambda)
    inv = math.exp(-t)
    scaled_numerator = -math.expm1(-t) + inv * B
    log_numerator = torch.log(scaled_numerator)
    log_A = torch.log(A)
    log_theta = log_numerator - log_A + t
    next_ratio = torch.cat([ratios[:, 1:], torch.full_like(ratios[:, :1], float("inf"))], 1)
    # The first prefix satisfying the upper threshold automatically satisfies
    # its lower threshold: adding a class occurs only after the previous prefix
    # threshold exceeds that class's ratio. This also handles tied ratios.
    active = (log_theta <= next_ratio).to(torch.int64).argmax(1)
    h = (log_A - log_numerator).gather(1, active[:, None]).squeeze(1)
    h = torch.minimum(torch.clamp(h, min=0.0), caps)
    return torch.where(positive, h, torch.zeros_like(h))


def _project_prepared(q, p_ref, h, active_sets=None):
    """Forward-KL floor projection, including q=0 and h=0/infinity."""
    ratios, qs, rs, A, B = active_sets or _active_sets(q, p_ref)
    n, c = q.shape
    out = p_ref.clone()
    infinite = torch.isinf(h)
    out[infinite] = q[infinite]
    use = (h > 0) & ~infinite
    if not bool(use.any()):
        return out
    hs = h[use]
    # Candidate prefixes have 0..C-1 active classes. Reverse sums avoid losing
    # a very small free-class mass by evaluating it as 1 - cumulative_mass.
    Ap = torch.cat([torch.zeros_like(A[:, :1]), A[:, :-1]], 1)[use]
    q_tail = torch.flip(torch.flip(qs, [1]).cumsum(1), [1])[use]
    r_tail = torch.flip(torch.flip(rs, [1]).cumsum(1), [1])[use]
    free_mass = r_tail + Ap * (-torch.expm1(-hs))[:, None]
    log_theta = -hs[:, None] + torch.log(q_tail) - torch.log(free_mass)
    k = (log_theta <= ratios[use]).to(torch.int64).argmax(1)
    scale = (free_mass / q_tail).gather(1, k[:, None])
    floors = torch.exp(-hs)[:, None] * p_ref[use]
    extra = torch.clamp(scale * q[use] - floors, min=0.0)
    extra_mass = extra.sum(1)
    residual = -torch.expm1(-hs)
    # Reconstruct from floors plus nonnegative residual to preserve the floors
    # despite normalization roundoff. If sub-ulp h resolves no extra mass, the
    # incumbent is the numerically representable feasible answer.
    regular = extra_mass > 0
    projected = p_ref[use].clone()
    projected[regular] = (floors[regular] + residual[regular, None]
                           * extra[regular] / extra_mass[regular, None])
    out[use] = projected
    return out


@torch.no_grad()
def project_probability_floor(q, p_ref, h):
    """Return argmin KL(q||p) subject to p >= exp(-h)*p_ref, row by row."""
    q, p_ref, _ = _probabilities(q, p_ref)
    budgets = _row_caps(h, q.shape[0], q.device)
    return _project_prepared(q, p_ref, budgets)


def _gap(log1p_lambda: float, slack: float) -> float:
    if log1p_lambda <= 0 or slack <= 0:
        return 0.0
    log_lambda = log1p_lambda + math.log(-math.expm1(-log1p_lambda))
    log_gap = log_lambda + math.log(slack)
    return math.exp(log_gap) if log_gap < 709.0 else float("inf")


@torch.no_grad()
def continuous_floor_frontier(
        q, p_ref, weights, H_plus: float, H_row=1.0, *,
        spend_atol: float = 1e-11, gain_atol: float = 1e-10,
        max_iterations: int = 160,
) -> Tuple[torch.Tensor, torch.Tensor, float, Dict[str, Any]]:
    """Solve the continuous row/call probability-floor frontier.

    Returns ``(h, p, gain, metadata)`` with float64 tensors on the input device.
    The constraints are ``0 <= h_i <= H_row_i`` and ``sum(w_i*h_i) <= H_plus``.
    ``gain = sum(w_i * sum_c q_ic * log(p_ic/p_ref_ic))``. Weights must sum to one;
    zero-weight rows retain the incumbent and spend zero budget.

    Row active sets are analytic. A scalar bisection chooses the common marginal
    gain, retaining the feasible side of the budget throughout. The reported
    duality gap is a floating-point Lagrangian estimate, not an interval proof.
    ``converged`` reports both requested spend and objective tolerances, except
    that a saturated/capped optimum may legitimately leave budget unused.
    """
    q, p_ref, normalization_error = _probabilities(q, p_ref)
    n, _ = q.shape
    w = torch.as_tensor(weights, dtype=torch.float64, device=q.device)
    if w.shape != (n,) or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("weights must be finite, nonnegative and have one entry per row")
    weight_mass = float(w.sum())
    if abs(weight_mass - 1.0) > 1e-5:
        raise ValueError("weights must sum to one (tolerance 1e-5)")
    w = w / weight_mass
    if not math.isfinite(H_plus) or H_plus < 0:
        raise ValueError("H_plus must be finite and nonnegative")
    if spend_atol <= 0 or gain_atol <= 0 or max_iterations < 1:
        raise ValueError("Positive tolerances and max_iterations >= 1 are required")
    caps = _row_caps(H_row, n, q.device)
    positive = w > 0
    prepared = _active_sets(q, p_ref)
    ratios, _, _, A, B = prepared
    saturation = (torch.log(p_ref) - torch.log(q)).max(1).values.clamp_min(0)
    h_saturated = torch.where(positive, torch.minimum(saturation, caps), torch.zeros_like(caps))
    saturated_spend = float(w @ h_saturated)
    upper = max(0.0, float(ratios[positive, -1].max()))
    saturated = saturated_spend <= H_plus
    iterations = 0
    if H_plus == 0.0:
        h = torch.zeros_like(caps)
        t = upper
    elif saturated:
        h = h_saturated
        t = 0.0
    else:
        lower, t = 0.0, upper
        h = torch.zeros_like(caps)  # feasible multiplier upper endpoint
        for iterations in range(1, max_iterations + 1):
            middle = 0.5 * (lower + t)
            proposal = _budgets_at_log_multiplier(middle, ratios, A, B, caps, positive)
            spend = float(w @ proposal)
            if spend > H_plus:
                lower = middle
            else:
                t, h = middle, proposal
            slack = max(0.0, H_plus - float(w @ h))
            if slack <= spend_atol and _gap(t, slack) <= gain_atol:
                break
    spend = float(w @ h)
    p = _project_prepared(q, p_ref, h, prepared)
    log_change = torch.log(p) - torch.log(p_ref)
    row_gain = torch.where(q > 0, q * log_change, torch.zeros_like(q)).sum(1)
    gain = float(w @ row_gain)
    floors = torch.exp(-h)[:, None] * p_ref
    floor_violation = float(torch.clamp(floors - p, min=0).max())
    normalization_residual = float((p.sum(1) - 1).abs().max())
    dinf = (torch.log(p_ref) - torch.log(p)).max(1).values
    finite_h = torch.isfinite(h)
    row_violation = float(torch.clamp(dinf[finite_h] - h[finite_h], min=0).max()) if bool(finite_h.any()) else 0.0
    gap = _gap(t, max(0.0, H_plus - spend))
    multiplier: Optional[float] = math.expm1(t) if t < 709.0 else None
    metadata = {
        "algorithm": "continuous_active_sets_and_dual_bisection",
        "posterior_interpretation": "optimal_for_supplied_q_only",
        "numerics": "float64_with_reported_residuals_not_formal_interval_arithmetic",
        "iterations": iterations,
        "converged": bool((saturated or H_plus - spend <= spend_atol) and gap <= gain_atol
                          and row_violation <= 1e-9 and normalization_residual <= 1e-12),
        "row_caps_or_posterior_saturation_reached": saturated,
        "allocated_spend": spend,
        "budget_slack": max(0.0, H_plus - spend),
        "budget_violation": max(0.0, spend - H_plus),
        "multiplier": multiplier,
        "log1p_multiplier": t,
        "duality_gap_estimate": gap,
        "max_probability_floor_violation": floor_violation,
        "max_row_divergence_violation": row_violation,
        "max_probability_normalization_residual": normalization_residual,
        "max_input_probability_normalization_error": normalization_error,
        "input_weight_normalization_error": abs(weight_mass - 1.0),
    }
    if not math.isfinite(gain) or not bool(torch.isfinite(p).all()):
        raise FloatingPointError("Frontier projection lost finite precision; use a log-probability formulation")
    return h, p, gain, metadata


__all__ = ["continuous_floor_frontier", "project_probability_floor"]
