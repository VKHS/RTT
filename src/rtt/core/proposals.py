"""Candidate proposals, support features, role assignment, and learned heads."""
from __future__ import annotations

from ._imports import (
    Any,
    F,
    List,
    Sequence,
    math,
    nn,
    np,
    torch,
)
from . import api as _api

# ============================================================================
# Section 6: proposals, support-function features, roles and heads (App. A)
# ============================================================================

class NativeProposal(nn.Module):
    """Two candidates per row at every open depth: (1) a state-conditioned convex
    combination of bank vectors, (2) a signed one-hop attention update."""
    M = 2
    name = "native"

    def __init__(self, d: int, cfg: _api.RTTConfig, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(_api.stable_seed("proposal", seed))
        self.d = d
        self.att_dim = int(cfg.att_dim)
        self.bank = nn.Parameter(torch.from_numpy((rng.standard_normal((cfg.bank_size, d)) * 0.3).astype(np.float32)))
        self.depth_emb = nn.Parameter(torch.zeros(cfg.T, 16))
        self.q_bank = nn.Linear(d + 16, self.att_dim)
        self.k_bank = nn.Parameter(torch.from_numpy((rng.standard_normal((cfg.bank_size, self.att_dim)) * 0.1).astype(np.float32)))
        self.q_att = nn.Linear(d, self.att_dim)
        self.k_att = nn.Linear(d, self.att_dim)
        self.v_att = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.v_att.weight.copy_(torch.eye(d) * 0.5)
        # learnable per-depth, per-candidate log-scale: the learned update starts close
        # to the incumbent and grows only where training finds it useful
        self.log_scale = nn.Parameter(torch.full((cfg.T, 2), math.log(0.05)))

    def prepare_call(self, ctx: _api.CallContext) -> None:
        return None

    def candidates(self, Hbar, depth: int, ctx: _api.CallContext):
        n, d = Hbar.shape
        emb = self.depth_emb[depth].unsqueeze(0).expand(n, -1)
        logits = self.q_bank(torch.cat([Hbar, emb], 1)) @ self.k_bank.t() / math.sqrt(self.att_dim)
        a = torch.softmax(logits, dim=1)
        c1 = a @ self.bank - Hbar
        op = ctx.op
        q = self.q_att(Hbar)
        k = self.k_att(Hbar)
        e = torch.tanh((q[op.rows] * k[op.cols]).sum(1) / math.sqrt(self.att_dim))
        e = e * (op.rows != op.cols).to(e.dtype)
        vals = op.vals64.to(e.dtype) * e
        msg = op.edge_values_mm(vals, Hbar)
        rs = op.edge_values_mm(vals, torch.ones(n, 1, dtype=Hbar.dtype, device=Hbar.device))
        c2 = self.v_att(msg - rs * Hbar)
        s = torch.exp(self.log_scale[depth])
        return torch.stack([c1 * s[0], c2 * s[1]], 0)


class StateProposal(nn.Module):
    """External proposal exposing row-aligned states at every depth (retrained
    incumbent, LoRA delta, GPR-GNN, exported states), mapped into the
    incumbent's state space by frozen linear adapters Omega_l.  Candidate 1 is
    the mapped state, candidate 2 its one-hop message."""
    M = 2

    def __init__(self, name: str, d_src: int, d: int, T: int):
        super().__init__()
        self.name = name
        self.d_src = d_src
        self.d = d
        self.T = T
        self.register_buffer("Omega", torch.eye(d_src, d).unsqueeze(0).repeat(T + 1, 1, 1))
        self.identity_adapter = (d_src == d)
        self.source = None      # callable(ctx) -> list of source states indexed by depth

    def prepare_call(self, ctx: _api.CallContext) -> None:
        key = f"proposal_states:{self.name}"
        if key not in ctx.cache:
            with torch.no_grad():
                ctx.cache[key] = self.source(ctx)

    def mapped_state(self, depth_next: int, ctx: _api.CallContext):
        states = ctx.cache[f"proposal_states:{self.name}"]
        S = states[min(depth_next, len(states) - 1)]
        if self.identity_adapter:
            return S
        return S @ self.Omega[depth_next]

    def candidates(self, Hbar, depth: int, ctx: _api.CallContext):
        Hc = self.mapped_state(depth + 1, ctx)
        c1 = Hc - Hbar
        c2 = ctx.op.mm(Hc) - Hbar
        return torch.stack([c1, c2], 0)


def fit_adapters_least_squares(src_states: List[List[Any]], tgt_states: List[List[Any]], rows: Any,
                               depths: Sequence[int], ridge: float = 1e-3):
    """Omega_l = argmin ||S_l Omega - H_l||^2 + ridge ||Omega||^2 over fitting rows of
    several calls (depth l indexes the incumbent state H_l)."""
    out = {}
    for l in depths:
        A = None
        Bm = None
        for S, H in zip(src_states, tgt_states):
            Sl = S[min(l, len(S) - 1)][rows].to(torch.float64)
            Hl = H[l][rows].to(torch.float64)
            A = Sl.t() @ Sl if A is None else A + Sl.t() @ Sl
            Bm = Sl.t() @ Hl if Bm is None else Bm + Sl.t() @ Hl
        lam = ridge * float(torch.diagonal(A).mean().item() + 1e-12)
        Om = torch.linalg.solve(A + lam * torch.eye(A.shape[0], dtype=A.dtype, device=A.device), Bm)
        out[l] = Om.to(torch.float32)
    return out


# ----------------------------------------------------------------------------
# candidate trims, support-function features, roles
# ----------------------------------------------------------------------------

def norm_trim(cand, trim: float):
    nrm = torch.linalg.norm(cand, dim=2, keepdim=True)
    scale = torch.clamp(trim / torch.clamp(nrm, min=1e-30), max=1.0)
    return cand * scale


def row_budget_trim(cand, rho_rows, cap_rows):
    """Scale candidates so that cap_k * ||d_i^{(k,m)}|| <= rho_i (eq. 13).  A zero cap
    or zero direction yields zero displacement without division."""
    nrm = torch.linalg.norm(cand, dim=2)                          # [M, n]
    ok = (cap_rows > 0).unsqueeze(0) & (nrm > 0)
    allowed = rho_rows.unsqueeze(0) / torch.clamp(cap_rows.unsqueeze(0), min=1e-30)
    scale = torch.where(ok, torch.clamp(allowed / torch.clamp(nrm, min=1e-30), max=1.0), torch.zeros_like(nrm))
    return cand * scale.unsqueeze(2)


def make_probes(d: int, per_sign: int, seed: int):
    rng = np.random.default_rng(_api.stable_seed("probes", seed, d, per_sign))
    U = rng.standard_normal((d, per_sign))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    return torch.from_numpy(U.astype(np.float32))


def support_features(cand, probes, mode: str = "support", scale: float = 1.0):
    """s_ij = max{0, max_m <u_j, d_i^(m)>} over +-probes (hull support function).
    mode='mean' is the non-invariant mean-pooling ablation."""
    proj = cand @ probes.to(cand.dtype)                           # [M, n, P]
    if mode == "mean":
        pos = proj.mean(0)
        neg = (-proj).mean(0)
    else:
        pos = torch.clamp(proj.max(0).values, min=0.0)
        neg = torch.clamp((-proj).max(0).values, min=0.0)
    return torch.cat([pos, neg], 1) / scale


def assign_roles(scored_rows, conf_rows, thresholds: Sequence[float]):
    q = torch.zeros_like(conf_rows, dtype=torch.long)
    for t in thresholds:
        q = q + (conf_rows > float(t)).to(torch.long)
    return q + 4 * (~scored_rows).to(torch.long)


def lens_stats(p):
    ent = -(p * torch.log(torch.clamp(p, min=1e-12))).sum(1, keepdim=True)
    top2 = torch.topk(p, k=min(2, p.shape[1]), dim=1).values
    maxp = top2[:, :1]
    margin = (top2[:, :1] - top2[:, 1:2]) if p.shape[1] > 1 else top2[:, :1]
    return torch.cat([ent, maxp, margin], 1)


def feature_dim(C: int, n_open: int, P: int) -> int:
    return 3 * C + 3 + 2 + 5 + 3 + n_open + 2 + 2 * P


def build_row_features(rows, ctx: _api.CallContext, depth_index: int, n_open: int, lens_p, nbr_lens, h0_lens,
                       lam_prov, r_prov, rho, s_feat, gamma: float, H_row: float, x_norm, h0_norm):
    nq = float(ctx.spec.nq)
    n_s = rows.shape[0]
    dev = lens_p.device
    parts = [lens_p[rows], lens_stats(lens_p[rows]), nbr_lens[rows], h0_lens[rows],
             ctx.log_deg_call[rows].unsqueeze(1), ctx.env_log_deg0[rows].unsqueeze(1),
             ctx.scored[rows].to(torch.float32).unsqueeze(1),
             F.one_hot(ctx.hop[rows], 4).to(torch.float32),
             torch.log(lam_prov[rows] * nq / max(gamma, 1e-12) + 1e-12).unsqueeze(1),
             (r_prov[rows] * gamma / max(H_row, 1e-12)).unsqueeze(1),
             torch.clamp(rho[rows] * gamma / max(H_row, 1e-12), max=10.0).unsqueeze(1),
             F.one_hot(torch.full((n_s,), depth_index, dtype=torch.long, device=dev), n_open).to(torch.float32),
             x_norm[rows].unsqueeze(1), h0_norm[rows].unsqueeze(1), s_feat]
    out = torch.cat([p.to(torch.float32) for p in parts], 1)
    return torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-30.0)


def call_feature_dim(C: int) -> int:
    return 4 * C + 3 + 2 + 4 + 2


def build_call_features(ctx: _api.CallContext, pT_full, rows):
    """Label-free, candidate-free call-level features of the incumbent's own
    terminal prediction (used by the call-level posterior head)."""
    parts = [pT_full[rows], lens_stats(pT_full[rows]), ctx.cache["pT_nbr1"][rows], ctx.cache["pT_nbr2"][rows],
             ctx.cache["h0_lens"][rows], ctx.log_deg_call[rows].unsqueeze(1), ctx.env_log_deg0[rows].unsqueeze(1),
             F.one_hot(ctx.hop[rows], 4).to(torch.float32), ctx.cache["x_norm"][rows].unsqueeze(1),
             ctx.cache["h0_norm"][rows].unsqueeze(1)]
    out = torch.cat([q.to(torch.float32) for q in parts], 1)
    return torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-30.0)


N_FIELD_COEFFS = 4      # three local residual fields and a support-row residual


class RTTHeads(nn.Module):
    """Node-weight, field, covariance and posterior heads (App. A).

    Inference reads only the current executed state and its label-free features.
    In particular, the field never needs the incumbent's terminal prediction.
    """

    def __init__(self, F_in: int, C: int, n_open: int, cfg: _api.RTTConfig):
        super().__init__()
        h = int(cfg.head_hidden)
        self.post = _api.MLP(F_in, h, C, zero_last=True)
        self.field = _api.MLP(F_in, h, N_FIELD_COEFFS, zero_last=True)
        init = torch.full((n_open, N_FIELD_COEFFS), -3.0)
        init[:, 0] = math.log(math.e - 1.0)
        self.field_scale = nn.Parameter(init)
        self.cov = _api.MLP(F_in, h, 1)
        self.cov_scale = nn.Parameter(torch.zeros(n_open))
        self.nodew = _api.MLP(F_in, h, 1)
        with torch.no_grad():
            self.nodew.lins[-1].bias.fill_(2.0)
        self.F_in = F_in
        self.C = C
        self.n_open = n_open

    def posterior(self, feat, lens_rows):
        return torch.softmax(torch.log(torch.clamp(lens_rows, min=1e-8)) + self.post(feat), dim=1)

    def field_coeffs(self, feat, depth_index: int):
        return F.softplus(self.field(feat) + self.field_scale[depth_index].unsqueeze(0))

    def variance(self, feat, depth_index: int, unit2: float, floor_rel: float):
        v = F.softplus(self.cov(feat).squeeze(1) + self.cov_scale[depth_index])
        return (v + floor_rel) * unit2

    def node_weight(self, feat_pre):
        return torch.sigmoid(self.nodew(feat_pre).squeeze(1))


class TrustRuleHead(nn.Module):
    """Matched trust rule (TR): a learned trust scale in [0,1] per role block,
    computed from the same features, scaling the proposal's step inside the same
    hull, budgets and certificate."""

    def __init__(self, F_in: int, K: int, n_open: int, cfg: _api.RTTConfig):
        super().__init__()
        self.K = K
        self.net = _api.MLP(F_in + K + 1, int(cfg.head_hidden), 1)
        with torch.no_grad():
            self.net.lins[-1].bias.fill_(-1.0)

    def forward(self, block_feat, block_ids, block_sizes):
        oh = F.one_hot(block_ids, self.K).to(torch.float32)
        z = torch.cat([block_feat, oh, torch.log1p(block_sizes.to(torch.float32)).unsqueeze(1)], 1)
        return torch.sigmoid(self.net(z).squeeze(1))


def field_estimate(heads: RTTHeads, feat, depth_index: int, rows, E0_hops: List[Any], E_self_rows, lam_rows,
                   gamma: float, adj_rows=None):
    """A learned, state-local field; no incumbent endpoint and no labels.

    Targets are the full reference-tail loss slopes, evaluated offline.
    """
    beta = heads.field_coeffs(feat, depth_index)                   # [n_s, 4]
    g = beta[:, 0:1] * E0_hops[0][rows] + beta[:, 1:2] * E0_hops[1][rows] + beta[:, 2:3] * E0_hops[2][rows]
    g = g + beta[:, 3:4] * (lam_rows / max(gamma, 1e-12)).unsqueeze(1) * E_self_rows
    if adj_rows is not None:
        raise ValueError("An incumbent-endpoint adjoint is not a deployment field input")
    return g

