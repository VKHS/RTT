"""Probability-floor geometry and comparator budget allocation."""
from __future__ import annotations

from ._imports import (
    Any,
    List,
    Sequence,
    Tuple,
    math,
    np,
    torch,
)
from . import api as _api

# ============================================================================
# Section 10: output-space geometry: floor projection (Prop. 1(iv)), the row
#             frontier and water-filling (Thm. 2(i)), mixing rules
# ============================================================================

def d_inf(p, q):
    """D_inf(p || q) per row = max_c log(p_c / q_c) (float64)."""
    return (torch.log(torch.clamp(p.to(torch.float64), min=1e-300)) -
            torch.log(torch.clamp(q.to(torch.float64), min=1e-300))).max(1).values


def floor_project(q, pr, h, shrink: float = 1e-9):
    """argmin_{p in F_h(pr)} KL(q || p): p_c = max(e^{-h} pr_c, q_c / lambda) with the
    normaliser found by sorting probability-to-floor ratios (active-set solution).
    h may be 0 (returns pr) or +inf (returns q).  A tiny shrink of h keeps the
    floor valid after the final renormalisation."""
    q = q.to(torch.float64)
    pr = pr.to(torch.float64)
    h = h.to(torch.float64)
    n, C = q.shape
    out = torch.empty_like(q)
    zero = h <= 1e-15
    inf = ~torch.isfinite(h)
    mid = ~(zero | inf)
    out[zero] = pr[zero]
    out[inf] = q[inf]
    if bool(mid.any().item()):
        qm = q[mid]
        pm = pr[mid]
        hm = torch.clamp(h[mid] * (1.0 - shrink) - 1e-12, min=0.0)
        a = torch.exp(-hm).unsqueeze(1) * pm
        ratio = qm / torch.clamp(a, min=1e-300)
        order = torch.argsort(ratio, dim=1, descending=True)
        rs = torch.gather(ratio, 1, order)
        qs = torch.gather(qm, 1, order)
        as_ = torch.gather(a, 1, order)
        Qk = torch.cumsum(qs, 1)                                   # top-k q
        A_rest = torch.exp(-hm).unsqueeze(1) - torch.cumsum(as_, 1)
        denom = torch.clamp(1.0 - A_rest, min=1e-300)
        lam = Qk / denom                                          # lambda_k, k=1..C
        nxt = torch.cat([rs[:, 1:], torch.zeros_like(rs[:, :1])], 1)
        ok = (rs >= lam - 1e-15 * lam.abs()) & (lam >= nxt - 1e-15 * lam.abs())
        kidx = torch.argmax(ok.to(torch.int64), dim=1)
        lam_star = lam.gather(1, kidx.view(-1, 1))
        p = torch.maximum(a, qm / lam_star)
        p = p / p.sum(1, keepdim=True)
        out[mid] = p
    return out


def frontier_curves(pB, pr, hgrid):
    """G_i(h_g) = sum_c pB_c log(p(h_g)_c / pr_c), p(h) the forward floor projection."""
    n = pB.shape[0]
    G = torch.zeros(n, len(hgrid), dtype=torch.float64, device=pB.device)
    lpr = torch.log(torch.clamp(pr.to(torch.float64), min=1e-300))
    for gi, hv in enumerate(hgrid):
        if hv <= 0:
            continue
        p = floor_project(pB, pr, torch.full((n,), float(hv), dtype=torch.float64, device=pB.device))
        G[:, gi] = (pB.to(torch.float64) * (torch.log(torch.clamp(p, min=1e-300)) - lpr)).sum(1)
    return G


def make_hgrid(H_row: float, H_plus: float, n: int = 48) -> List[float]:
    top = H_row if math.isfinite(H_row) else max(8.0, 40.0 * H_plus)
    g = [0.0] + list(np.geomspace(min(1e-5, top / 1e4), top, n - 1))
    return [float(x) for x in g]


def water_fill(G, hgrid: Sequence[float], w, H_plus: float, hmax=None) -> Tuple[Any, float]:
    """Allocate a continuous budget on a piecewise-linear concave surrogate.

    For concave input curves this solves their piecewise-linear interpolation,
    including fractional and tied marginal segments. For nonconcave comparator
    curves, cumulative-minimum slopes define a conservative concave surrogate;
    this is explicitly a grid approximation, not the paper's continuous oracle.
    The returned gain is evaluated on the original interpolated curves.
    """
    dev = G.device
    hg = torch.tensor(hgrid, dtype=torch.float64, device=dev)
    n, ng = G.shape
    w = w.to(torch.float64)
    if ng < 2 or hg.numel() != ng or not bool(torch.isfinite(G).all().item()):
        raise ValueError("water_fill requires finite curves and at least two grid points")
    if not bool(torch.isfinite(hg).all().item()) or float(hg[0]) != 0.0 or not bool((hg[1:] > hg[:-1]).all().item()):
        raise ValueError("water_fill grid must start at zero and increase strictly")
    if w.shape != (n,) or not bool(torch.isfinite(w).all().item()) or bool((w < 0).any().item()):
        raise ValueError("water_fill weights must be finite and nonnegative")
    if not math.isfinite(H_plus) or H_plus < 0:
        raise ValueError("water_fill budget must be finite and nonnegative")
    seg = hg[1:] - hg[:-1]                                         # [ng-1]
    slopes = (G[:, 1:] - G[:, :-1]) / seg.unsqueeze(0)
    # Concave lower surrogate; exact for already-concave piecewise-linear curves.
    s = slopes.clone()
    for gidx in range(1, s.shape[1]):
        s[:, gidx] = torch.minimum(s[:, gidx], s[:, gidx - 1])
    if hmax is not None:
        hm = hmax.to(torch.float64)
        if hm.shape != (n,) or bool(torch.isnan(hm).any().item()) or bool((hm < 0).any().item()):
            raise ValueError("water_fill row caps must be nonnegative and not NaN")
        seg_hi = hg[1:].unsqueeze(0)
        avail = torch.clamp(torch.minimum(seg_hi, hm.unsqueeze(1)) - hg[:-1].unsqueeze(0), min=0.0)
    else:
        avail = seg.unsqueeze(0).expand(n, -1)

    capacity = (w.unsqueeze(1) * avail).reshape(-1)
    flat_slopes = s.reshape(-1)
    eligible = (flat_slopes > 0.0) & (capacity > 0.0)
    ids = torch.nonzero(eligible, as_tuple=False).squeeze(1)
    take_flat = torch.zeros_like(capacity)
    if ids.numel() and H_plus > 0.0:
        # Stable ties preserve each row's segment order, so every allocation is
        # a contiguous prefix of that row's curve, including equal-slope ties.
        order = torch.argsort(flat_slopes[ids], descending=True, stable=True)
        ids = ids[order]
        costs = capacity[ids]
        before = torch.cumsum(costs, dim=0) - costs
        spent = torch.minimum(torch.clamp(H_plus - before, min=0.0), costs)
        take_flat[ids] = spent / costs
    h = (take_flat.reshape(n, ng - 1) * avail).sum(1)
    actual_spend = float(torch.dot(w, h).item())
    if actual_spend > H_plus and actual_spend > 0.0:
        h = h * (H_plus / actual_spend) * (1.0 - 8.0 * _api.EPS64)
    # gain by linear interpolation of G at h
    idx = torch.clamp(torch.searchsorted(hg.contiguous(), h.contiguous(), right=True) - 1, 0, ng - 2)
    h_lo = hg[idx]
    frac = torch.clamp((h - h_lo) / seg[idx], 0.0, 1.0)
    Gl = G.gather(1, idx.view(-1, 1)).squeeze(1)
    Gh = G.gather(1, (idx + 1).view(-1, 1)).squeeze(1)
    gain_rows = Gl + frac * (Gh - Gl)
    return h, float((w * gain_rows).sum().item())


def mixing_eps_max(pr, pu, h):
    """Largest eps with D_inf(pr || (1-eps) pr + eps pu) <= h."""
    pr = pr.to(torch.float64)
    pu = pu.to(torch.float64)
    one_m = (1.0 - torch.exp(-h.to(torch.float64))).unsqueeze(1)
    dec = pr - pu
    bound = torch.where(dec > 0, one_m * pr / torch.clamp(dec, min=1e-300), torch.full_like(pr, float("inf")))
    e = torch.clamp(bound.min(1).values, max=1.0)
    return torch.clamp(e * (1.0 - 1e-9), min=0.0)


def mixing_curves(q, pr, pu, hgrid):
    n = q.shape[0]
    G = torch.zeros(n, len(hgrid), dtype=torch.float64, device=q.device)
    lpr = torch.log(torch.clamp(pr.to(torch.float64), min=1e-300))
    for gi, hv in enumerate(hgrid):
        if hv <= 0:
            continue
        e = mixing_eps_max(pr, pu, torch.full((n,), float(hv), dtype=torch.float64, device=q.device))
        mix = (1.0 - e).unsqueeze(1) * pr.to(torch.float64) + e.unsqueeze(1) * pu.to(torch.float64)
        G[:, gi] = (q.to(torch.float64) * (torch.log(torch.clamp(mix, min=1e-300)) - lpr)).sum(1)
    return G


def projection_curves(q, pr, pu, hgrid):
    """Gain (under plug-in q) of the floor projection of the raw update at budget h."""
    n = q.shape[0]
    G = torch.zeros(n, len(hgrid), dtype=torch.float64, device=q.device)
    lpr = torch.log(torch.clamp(pr.to(torch.float64), min=1e-300))
    for gi, hv in enumerate(hgrid):
        if hv <= 0:
            continue
        p = floor_project(pu, pr, torch.full((n,), float(hv), dtype=torch.float64, device=q.device))
        G[:, gi] = (q.to(torch.float64) * (torch.log(torch.clamp(p, min=1e-300)) - lpr)).sum(1)
    return G


def row_frontier_derivative_at_zero(pB, pr):
    """G_i'(0+) = R_i - 1 with R_i = max_c pB_c / pr_c (Thm. 2(i))."""
    return (pB.to(torch.float64) / torch.clamp(pr.to(torch.float64), min=1e-300)).max(1).values - 1.0

