"""Incumbent reference-tail forward, Jacobian, and adjoint calculus."""
from __future__ import annotations

from ._imports import (
    Any,
    List,
    Optional,
    torch,
)
from . import api as _api

def incumbent_rollout(inc: _api.TanhDiffusionIncumbent, H0, op: _api.GraphOp, masks: Optional[_api.CallContext] = None,
                      H_start=None, j_start: int = 0, keep_states: bool = True):
    """float32 incumbent continuation from (H_start, j_start); returns the list of
    states [H_{j_start}, ..., H_T] (or only the terminal state)."""
    W = inc.weight()
    H = H0 if H_start is None else H_start
    states = [H] if keep_states else None
    for j in range(j_start, inc.T):
        H = inc.transition(H, H0, op, W, masks, j)
        if keep_states:
            states.append(H)
    return states if keep_states else [H]


def tail_forward(inc: _api.TanhDiffusionIncumbent, ctx: _api.CallContext, H_start, j_start: int, dtype=None):
    """Reference tail from H_start at depth j_start.  Returns (states, derivs) with
    derivs[j - j_start] = act'(Z_j) for the transition j -> j+1."""
    W = inc.weight()
    H0 = ctx.H0
    if dtype is not None:
        W = W.to(dtype)
        H0 = H0.to(dtype)
        H_start = H_start.to(dtype)
    H = H_start
    states = [H]
    derivs = []
    for j in range(j_start, inc.T):
        Hn, Z = inc.transition(H, H0, ctx.op, W, ctx, j, return_pre=True)
        derivs.append(inc.act_deriv(Z))
        states.append(Hn)
        H = Hn
    return states, derivs


def tail_vjp(inc: _api.TanhDiffusionIncumbent, ctx: _api.CallContext, derivs: List[Any], j_start: int, G_T):
    """Backpropagate a terminal state gradient G_T through the tail to depth j_start."""
    W = inc.weight().to(G_T.dtype)
    a, t = inc.alpha, inc.tau
    G = G_T
    for jj in range(len(derivs) - 1, -1, -1):
        j = j_start + jj
        keep = ctx.keep(j + 1, G.dtype)
        Gm = G if keep is None else G * keep.unsqueeze(1)
        inner = (Gm * derivs[jj].to(G.dtype)) @ W.t()
        G = (1.0 - a) * ((1.0 - t) * Gm + t * ctx.op.tmm(inner))
    return G


def tail_jvp(inc: _api.TanhDiffusionIncumbent, ctx: _api.CallContext, derivs: List[Any], j_start: int, dH):
    """Push tangents dH ([n,d] or [k,n,d]) from depth j_start to the terminal depth."""
    W = inc.weight().to(dH.dtype)
    a, t = inc.alpha, inc.tau
    batched = dH.dim() == 3
    X = dH
    for jj in range(len(derivs)):
        j = j_start + jj
        if batched:
            k, n, d = X.shape
            PX = ctx.op.mm(X.permute(1, 0, 2).reshape(n, k * d)).reshape(n, k, d).permute(1, 0, 2)
            Z = (PX @ W) * derivs[jj].to(X.dtype).unsqueeze(0)
        else:
            Z = (ctx.op.mm(X) @ W) * derivs[jj].to(X.dtype)
        Xn = (1.0 - a) * ((1.0 - t) * X + t * Z)
        keep = ctx.keep(j + 1, X.dtype)
        if keep is not None:
            Xn = Xn * (keep.unsqueeze(1) if not batched else keep.view(1, -1, 1))
        X = Xn
    return X


def terminal_ce_grad(inc: _api.TanhDiffusionIncumbent, ctx: _api.CallContext, H_T, soft_targets=None):
    """Gradient of V_T(H) = sum_i w_i CE(z_i, y_i) with respect to H_T (scored rows);
    soft_targets [nq, C] replaces the one-hot labels (posterior slopes)."""
    dt = H_T.dtype
    A = inc.A_eff(dt)
    z = H_T[ctx.q_idx] @ A + inc.b_eff(dt)
    p = torch.softmax(z, dim=1)
    if soft_targets is None:
        tgt = torch.zeros_like(p)
        tgt[torch.arange(p.shape[0], device=p.device), ctx.y_q] = 1.0
    else:
        tgt = soft_targets.to(dt)
    wq = ctx.w.to(dt)[ctx.q_idx].unsqueeze(1)
    G = torch.zeros_like(H_T)
    G[ctx.q_idx] = (wq * (p - tgt)) @ A.t()
    return G


def ce_rows(logits, y):
    """Per-row cross-entropy in float64."""
    z = logits.to(torch.float64)
    return torch.logsumexp(z, dim=1) - z.gather(1, y.view(-1, 1)).squeeze(1)


