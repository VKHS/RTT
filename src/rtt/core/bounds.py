"""Interval factors, trajectory tubes, and backward derivative contracts."""
from __future__ import annotations

from ._imports import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    math,
    np,
    torch,
)
from . import api as _api

# ============================================================================
# Section 5: interval factors (Lemma 1), curvature factors, contracts (Prop. 3)
#            and tubes (Lemma 7, eq. 9)
# ============================================================================

EPS64 = float(np.finfo(np.float64).eps) / 2.0     # unit roundoff, float64
EPS32 = float(np.finfo(np.float32).eps) / 2.0     # unit roundoff, float32
TANH2_PEAK_Y = math.atanh(1.0 / math.sqrt(3.0))  # argmax |tanh''|
TANH2_PEAK = 4.0 / (3.0 * math.sqrt(3.0))


def gamma_m(m: float, u: float) -> float:
    """Higham's accumulation factor gamma_m = m u / (1 - m u)."""
    mu = float(m) * float(u)
    if mu >= 1.0:
        return float("inf")
    return mu / (1.0 - mu)


class Rounding:
    """Outward rounding by a-priori inflation.  'up' maps a nonnegative float64
    upper bound to a slightly larger float64 (relative + absolute margin, then one
    ulp towards +inf).  Disabled when exact=False (provisional / test mode)."""

    def __init__(self, rel: float = 1e-9, abs_: float = 1e-15, enabled: bool = True):
        self.rel = float(rel)
        self.abs = float(abs_)
        self.enabled = bool(enabled)

    def up(self, x):
        if not self.enabled:
            return x
        if isinstance(x, float):
            y = x + abs(x) * self.rel + self.abs
            return float(np.nextafter(y, np.inf))
        return x + x.abs() * self.rel + self.abs

    def down(self, x):
        if not self.enabled:
            return x
        if isinstance(x, float):
            y = x - abs(x) * self.rel - self.abs
            return float(np.nextafter(y, -np.inf))
        return x - x.abs() * self.rel - self.abs


NO_ROUNDING = Rounding(enabled=False)


def sum_upper(x, rnd: Rounding) -> float:
    """Upper bound on the exact sum of the float64 entries of x: recursive or tree
    summation errs by at most gamma_{m-1} * sum|x_i| (Higham, Thm. 4.4)."""
    s = float(x.sum().item())
    if not rnd.enabled:
        return s
    m = int(x.numel())
    a = float(x.abs().sum().item())
    return rnd.up(s + gamma_m(max(m, 1), EPS64) * a)


class SpectralSplit:
    """Rank-k split of G = W^T W (float64) for Lemma 1, with a rigorous bound on
    the residual ||G - B B^T||_2 that absorbs eigendecomposition error."""

    def __init__(self, W, k: int, device=None):
        W64 = W.detach().to(torch.float64)
        d = W64.shape[0]
        self.d = d
        self.k = int(min(max(1, k), d))
        G = W64.t() @ W64
        evals, evecs = torch.linalg.eigh(G)
        evals = torch.flip(evals, dims=[0])
        evecs = torch.flip(evecs, dims=[1])
        Sk = torch.clamp(evals[: self.k], min=0.0)
        B = evecs[:, : self.k] * torch.sqrt(Sk).unsqueeze(0)       # d x k
        self.B = B
        self.BB = (B.unsqueeze(2) * B.unsqueeze(1)).reshape(d, self.k * self.k)
        R = G - B @ B.t()
        R = 0.5 * (R + R.t())
        rev = torch.linalg.eigvalsh(R)
        rfro = float(torch.linalg.norm(R).item())
        wfro2 = float((W64 * W64).sum().item())
        # |fl(G) - G| <= gamma_d |W|^T|W| entrywise -> spectral <= gamma_d ||W||_F^2
        g_err = gamma_m(d + 2, EPS64) * wfro2
        self.r_norm = float(rev.abs().max().item()) + 10.0 * d * EPS64 * rfro + 2.0 * g_err
        lam_max = float(evals[0].item())
        self.w_norm = math.sqrt(max(lam_max, 0.0) + 10.0 * d * EPS64 * max(lam_max, 0.0) + g_err)
        self.w_norm = float(np.nextafter(self.w_norm * (1.0 + 1e-12), np.inf))
        self.wcol = torch.sqrt((W64 * W64).sum(0))                 # ||W_{:,u}||_2
        self.W64 = W64
        self.absW64 = W64.abs()
        if device is not None:
            self.to(device)

    def to(self, device):
        for name in ("B", "BB", "wcol", "W64", "absW64"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def as_dtype(self, dtype):
        return {"BB": self.BB.to(dtype), "wcol": self.wcol.to(dtype)}


def _act_deriv_interval(activation: str, absZ, shift):
    """sup over the interval of the activation derivative = act'(max(0,|z|-shift))."""
    if activation == "tanh":
        y = torch.clamp(absZ - shift, min=0.0)
        t = torch.tanh(y)
        return 1.0 - t * t
    return torch.ones_like(absZ)


def _tanh2_sup(lo, hi):
    """sup |tanh''(y)| for y in [lo, hi] with 0 <= lo <= hi."""
    def f(y):
        t = torch.tanh(y)
        return 2.0 * t * (1.0 - t * t)
    peak = torch.full_like(lo, TANH2_PEAK)
    inside = (lo <= TANH2_PEAK_Y) & (hi >= TANH2_PEAK_Y)
    left = hi < TANH2_PEAK_Y
    return torch.where(inside, peak, torch.where(left, f(hi), f(lo)))


def interval_factors(Z, rho_row, split: SpectralSplit, activation: str, dtype,
                     margin_Z=None, need_curvature: bool = True, chunk: int = 32768,
                     rounding: Rounding = NO_ROUNDING, global_only: bool = False):
    """Per-row factors on a tube.  Z [n,d] pre-activations at the tube centre,
    rho_row [n] = (P r)_i (pre-activation input radius).  Returns (L [n], mu [n],
    stats).  L_i >= ||W diag(sbar_i)||_2 >= every Jacobian norm on the tube
    (Lemma 1); mu_i bounds the curvature of the row map on the tube."""
    n, d = Z.shape
    wn = split.w_norm
    if activation != "tanh" or global_only:
        L = torch.full((n,), wn, dtype=dtype, device=Z.device)
        if activation != "tanh":
            mu = torch.zeros(n, dtype=dtype, device=Z.device)
        else:
            mu = torch.full((n,), wn * wn * TANH2_PEAK, dtype=dtype, device=Z.device)
        return rounding.up(L), rounding.up(mu), {"mode": "global"}
    wcol = split.wcol.to(dtype)
    BB = split.BB.to(dtype)
    k = split.k
    Ls = []
    mus = []
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        absZ = Z[s0:s1].to(dtype).abs()
        shift = rho_row[s0:s1].to(dtype).unsqueeze(1) * wcol.unsqueeze(0)
        if margin_Z is not None:
            shift = shift + margin_Z[s0:s1].to(dtype)
        sbar = _act_deriv_interval(activation, absZ, shift)
        s2 = sbar * sbar
        M = (s2 @ BB).reshape(s1 - s0, k, k)
        M = 0.5 * (M + M.transpose(1, 2))
        lam = torch.linalg.eigvalsh(M)[:, -1]
        tr = torch.diagonal(M, dim1=1, dim2=2).sum(1)
        if dtype == torch.float64:
            lam = lam + (10.0 * k + 2.0 * d * k) * EPS64 * tr.abs()
        bound2 = torch.clamp(lam, min=0.0) + split.r_norm * s2.max(dim=1).values
        Lr = torch.sqrt(torch.clamp(bound2, min=0.0))
        Lr = torch.clamp(Lr, max=wn)
        Ls.append(Lr)
        if need_curvature:
            lo = torch.clamp(absZ - shift, min=0.0)
            hi = absZ + shift
            sup2 = _tanh2_sup(lo, hi).max(dim=1).values
            mus.append((wn * wn) * sup2)
    L = torch.cat(Ls)
    mu = torch.cat(mus) if need_curvature else None
    L = rounding.up(L)
    if mu is not None:
        mu = rounding.up(mu)
    return L, mu, {"mode": f"rank{k}"}


def contracts_backward(op: _api.GraphOp, T: int, j_stop: int, L_of: Callable[[int], Any], mu_of: Callable[[int], Any],
                       keep_of: Callable[[int], Any], w, Gamma: float, alpha: float, tau: float,
                       rounding: Rounding = NO_ROUNDING, need_xi: bool = True,
                       store: Optional[Sequence[int]] = None):
    """Lambda_j, Xi_j for j = T .. j_stop (eq. 9 and Prop. 3).  Returns dicts."""
    dt = w.dtype
    Lam = w * Gamma
    Xi = w * (Gamma * Gamma / 4.0) if need_xi else None
    Lam = rounding.up(Lam)
    if need_xi:
        Xi = rounding.up(Xi)
    lam_store = {T: Lam}
    xi_store = {T: Xi} if need_xi else {}
    want = set(store) if store is not None else None
    for j in range(T - 1, j_stop - 1, -1):
        keep = keep_of(j + 1)
        Lp = Lam if keep is None else Lam * keep.to(dt)
        Lj = L_of(j).to(dt)
        Lam = (1.0 - alpha) * ((1.0 - tau) * Lp + tau * op.tmm(Lj * Lp))
        Lam = rounding.up(Lam)
        if need_xi:
            Xp = Xi if keep is None else Xi * keep.to(dt)
            mj = mu_of(j).to(dt)
            Xi = (1.0 - alpha) ** 2 * ((1.0 - tau) * Xp + tau * op.tmm(Lj * Lj * Xp)) \
                + (1.0 - alpha) * tau * op.tmm(mj * Lp)
            Xi = rounding.up(Xi)
        if want is None or j in want:
            lam_store[j] = Lam
            if need_xi:
                xi_store[j] = Xi
    return lam_store, xi_store


def tube_step(op: _api.GraphOp, r, L, keep, step_norm, alpha: float, tau: float, extra=None,
              rounding: Rounding = NO_ROUNDING):
    """r_{j+1} = keep * (1-a)[(1-t) r_j + t L_j (P r_j)] + ||v_j|| + extra (eq. 9)."""
    Pr = op.mm(r)
    rn = (1.0 - alpha) * ((1.0 - tau) * r + tau * L * Pr)
    if keep is not None:
        rn = rn * keep.to(rn.dtype)
    if step_norm is not None:
        rn = rn + step_norm.to(rn.dtype)
    if extra is not None:
        rn = rn + extra.to(rn.dtype)
    return rounding.up(rn)


def scalar_growth(L_mean_by_depth: Dict[int, float], alpha: float, tau: float, T: int) -> Dict[int, float]:
    """Pi_l = prod_{j>l} (1-a)(1-t+t*Lbar_j): scalar contract growth from depth l+1 to T."""
    out = {T - 1: 1.0}
    g = 1.0
    for l in range(T - 2, -1, -1):
        j = l + 1
        g *= (1.0 - alpha) * (1.0 - tau + tau * float(L_mean_by_depth.get(j, 1.0)))
        out[l] = g
    return out

