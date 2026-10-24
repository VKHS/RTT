"""Correctors, cross-fitted posterior, and optional comparator networks."""
from __future__ import annotations

from ._imports import (
    Any,
    F,
    List,
    Optional,
    Sequence,
    Tuple,
    math,
    nn,
    np,
    time,
    torch,
)
from . import api as _api

# ============================================================================
# Section 11: baselines -- one-pass correctors, cross-fitted posterior,
#             fine-tuned incumbents, learn-then-test switch (Sec. 4.1, Table 12)
# ============================================================================

def proposal_rows_features(system: _api.RTTSystem, ctx: _api.CallContext, depths: Optional[Sequence[int]] = None):
    """Support-function features of the proposal's candidates computed along the
    incumbent trajectory ('proposal candidate rows' given to the correctors)."""
    key = "proposal_rows_feat"
    if key in ctx.cache:
        return ctx.cache[key]
    cfg = system.cfg
    if depths is None:
        depths = [cfg.open_first, (cfg.open_first + cfg.T - 1) // 2, cfg.T - 1]
    feats = []
    with torch.no_grad():
        for l in depths:
            Hbar = ctx.inc_states[l + 1]
            cand = _api.norm_trim(system.proposal.candidates(Hbar, l, ctx), cfg.row_trim)
            feats.append(_api.support_features(cand, system.probes, "support", scale=cfg.row_trim))
    out = torch.cat(feats, 1)
    ctx.cache[key] = out
    return out


class OnePassCorrector(nn.Module):
    """Runs after one incumbent pass; reads terminal states, predictions, k-hop fields
    and (optionally) the proposal's candidate rows; predicts a distribution that is
    then floor-projected with water-filled row budgets (shares G1-G3 with RTT)."""

    def __init__(self, d: int, C: int, hops: int, use_rows: bool, P2: int, hidden: int, name: str):
        super().__init__()
        self.hops = int(hops)
        self.use_rows = bool(use_rows)
        self.name = name
        in_dim = d + C + ((self.hops * C + d) if self.hops > 0 else 0) + (3 * P2 if use_rows else 0)
        self.in_dim = in_dim
        self.net = _api.MLP(in_dim, hidden, C, zero_last=True)

    def features(self, system: _api.RTTSystem, ctx: _api.CallContext, H_T, p_full, rows_feat=None):
        q = ctx.q_idx
        parts = [H_T[q], p_full[q]]
        if self.hops > 0:
            x = p_full
            for _ in range(self.hops):
                x = ctx.op.mm(x)
                parts.append(x[q])
            parts.append(ctx.op.mm(H_T)[q])
        if self.use_rows:
            rf = rows_feat if rows_feat is not None else proposal_rows_features(system, ctx)
            parts.append(rf[q])
        return torch.cat([p.to(torch.float32) for p in parts], 1)

    def forward(self, feat, z_q):
        return torch.softmax(z_q.to(torch.float32) + self.net(feat), dim=1)


def matched_corrector_width(d: int, C: int, hops: int, use_rows: bool,
                            cfg: _api.RTTConfig) -> Tuple[int, int]:
    """Nearest integer width to the RTT heads' total trainable parameter count.

    Matching a hidden width alone does not match model capacity. The residual
    parameter-count mismatch is reported; there are no unused padding weights.
    """
    fin = _api.feature_dim(C, len(cfg.open_depths()), cfg.probes_per_sign)
    h = cfg.head_hidden
    def mlp_params(inputs, outputs, width):
        return inputs * width + width + width * width + width + width * outputs + outputs
    target = sum(mlp_params(fin, out, h) for out in (C, _api.N_FIELD_COEFFS, 1, 1))
    target += len(cfg.open_depths()) * (_api.N_FIELD_COEFFS + 1)
    inputs = d + C + ((hops * C + d) if hops > 0 else 0) + (6 * cfg.probes_per_sign if use_rows else 0)
    width_real = (-(inputs + C + 2) + math.sqrt((inputs + C + 2) ** 2 + 4 * (target - C))) / 2
    candidates = {max(1, int(math.floor(width_real))), max(1, int(math.ceil(width_real)))}
    width = min(candidates, key=lambda w: (abs(mlp_params(inputs, C, w) - target), w))
    return width, target


def corrector_serve(system: _api.RTTSystem, corr: OnePassCorrector, ctx: _api.CallContext, H_plus: float, H_row: float,
                    base_logits_q=None, base_H_T=None, row_allow=None, call_allow=None, cfg: Optional[_api.RTTConfig] = None):
    """Serve a corrector: q from the head, row budgets by water-filling on the plug-in
    frontier of q, floor projection relative to the base prediction."""
    cfg = cfg or system.cfg
    q = ctx.q_idx
    with torch.no_grad():
        H_T = base_H_T if base_H_T is not None else ctx.inc_states[-1]
        full_logits = system.inc.logits(H_T)
        p_full = torch.softmax(full_logits, 1)
        feat = corr.features(system, ctx, H_T, p_full)
        zq = full_logits[q] if base_logits_q is None else base_logits_q
        qd = corr(feat, zq).to(torch.float64)
        pr = torch.softmax(zq.to(torch.float64), 1)
        budget = H_plus if call_allow is None else call_allow
        wq = ctx.w64[q]
        hmax = row_allow if row_allow is not None else torch.full_like(wq, H_row)
        # The aggregate budget itself caps any positive-weight row's spend.
        finite_call_cap = torch.where(wq > 0, max(budget, 0.0) / torch.clamp(wq, min=1e-300), 0.0)
        hmax = torch.minimum(hmax.to(torch.float64), finite_call_cap)
        from ..frontier import continuous_floor_frontier
        h, ps, _, frontier_meta = continuous_floor_frontier(qd, pr, wq, max(budget, 0.0), hmax)
        if not frontier_meta["converged"]:
            raise _api.RTTFailure("SCIENTIFIC", "Corrector probability-floor allocation did not meet numerical tolerances")
        ctx.cache.setdefault("corrector_frontier_metadata", {})[corr.name] = frontier_meta
    return ps, h, qd


class CrossFitPosterior:
    """Cross-fitted calibrated classifier on 2018 calibration labels, used only for
    the posterior-based frontier estimate and the posterior-slope diagnostic."""

    def __init__(self, models: List[OnePassCorrector], temperature: float = 1.0):
        self.models = models
        self.temperature = float(temperature)

    def predict(self, system: _api.RTTSystem, ctx: _api.CallContext, exclude_fold: Optional[int] = None):
        with torch.no_grad():
            H_T = ctx.inc_states[-1]
            zl = system.inc.logits(H_T)
            p_full = torch.softmax(zl, 1)
            zq = zl[ctx.q_idx]
            acc = None
            cnt = 0
            for f, m in enumerate(self.models):
                if exclude_fold is not None and f == exclude_fold:
                    continue
                feat = m.features(system, ctx, H_T, p_full)
                lg = torch.log(torch.clamp(m(feat, zq), min=1e-12))
                acc = lg if acc is None else acc + lg
                cnt += 1
            lg = acc / max(cnt, 1)
            return torch.softmax(lg.to(torch.float64) / self.temperature, 1)


def train_classifier_on_calls(system: _api.RTTSystem, env: _api.BaseEnv, model: OnePassCorrector, specs: List[_api.CallSpec],
                              steps: int, lr: float, wd: float, node_filter=None, log_tag: str = "") -> dict:
    """CE training of a one-pass head on a stream of calls (labels on scored rows)."""
    opt = _api.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    losses = []
    t0 = time.time()
    for s in range(steps):
        spec = specs[s % len(specs)]
        ctx = env.build(spec)
        system.prepare_call(ctx, time_prefix=False)
        with torch.no_grad():
            H_T = ctx.inc_states[-1]
            zl = system.inc.logits(H_T)
            p_full = torch.softmax(zl, 1)
            feat = model.features(system, ctx, H_T, p_full)
            zq = zl[ctx.q_idx]
            y = ctx.y_q
            keep = None
            if node_filter is not None:
                keep = node_filter(ctx)
        model.train()
        qd = model(feat, zq)
        lp = torch.log(torch.clamp(qd, min=1e-12))
        nll = -lp.gather(1, y.view(-1, 1)).squeeze(1)
        if keep is not None:
            if not bool(keep.any().item()):
                continue
            nll = nll[keep]
        loss = nll.mean()
        opt.zero_grad()
        loss.backward()
        _api.clip_grad_norm(list(model.parameters()), 5.0)
        opt.step()
        losses.append(float(loss.item()))
        if (s + 1) % max(1, steps // 10) == 0:
            _api.log(f"[TRAIN {log_tag}] step {s+1}/{steps} loss(last100)={np.mean(losses[-100:]):.4f} ({time.time()-t0:.0f}s)")
    model.eval()
    return {"steps": steps, "final_loss": float(np.mean(losses[-100:])) if losses else None}


def fit_scalar_temperature(logps: List[Any], ys: List[Any], lo: float = 0.2, hi: float = 5.0) -> float:
    z = torch.cat(logps, 0).to(torch.float64)
    y = torch.cat(ys, 0)

    def nll(beta):
        zz = z * beta
        return float((torch.logsumexp(zz, 1) - zz.gather(1, y.view(-1, 1)).squeeze(1)).mean().item())
    beta = _api.golden_section_min(nll, 1.0 / hi, 1.0 / lo)
    return 1.0 / beta


# ----------------------------------------------------------------------------
# fine-tuned incumbents (training-time anti-regression baselines, Table 12)
# ----------------------------------------------------------------------------

FT_VARIANTS = {
    "ft_same_labels": "Incumbent fine-tuned on the same labels",
    "ft_kl": "KL-to-incumbent fine-tuning",
    "ft_dinf": "D_inf-regularized fine-tuning (floor trained, not enforced)",
    "ft_pct": "Positive-congruent training",
    "ft_regaware": "Regression-aware fine-tuning",
    "ft_corruptions": "Incumbent trained on the registered corruptions",
}


def clone_for_finetune(inc: _api.TanhDiffusionIncumbent, cfg: _api.RTTConfig, device) -> _api.TanhDiffusionIncumbent:
    obj = inc.export_state()
    c = _api.build_incumbent_from_state(obj, cfg, device)
    c.frozen = False
    with torch.no_grad():
        c.V.copy_(c.W_frozen)
        W64 = c.W_frozen.to(torch.float64)
        U, S, Vh = torch.linalg.svd(W64)
        c.u_pi.copy_(U[:, 0].to(torch.float32))
    for p in c.parameters():
        p.requires_grad_(True)
    c.temperature.requires_grad_(False)
    return c


def finetune_incumbent(base: _api.TanhDiffusionIncumbent, env: _api.BaseEnv, roles: _api.DataRoles, cfg: _api.RTTConfig, variant: str,
                       steps: int, seed: int, device) -> Tuple[_api.TanhDiffusionIncumbent, dict]:
    _api.set_global_seeds(seed)
    ft = clone_for_finetune(base, cfg, device)
    params = [p for n_, p in ft.named_parameters() if p.requires_grad]
    opt = _api.AdamW(params, lr=cfg.ft_lr, weight_decay=cfg.inc_wd)
    gen = env.generator
    comp = "clean" if variant == "ft_same_labels" else "mixture"
    losses = []
    for s in range(steps):
        spec = gen.make(f"ft_{variant}", "fit", comp, s)
        ctx = env.build(spec)
        with torch.no_grad():
            zr = base.deployed_logits(ctx.x, ctx.op, ctx=ctx)[ctx.q_idx]
            pr = torch.softmax(zr, 1)
        ft.train()
        H0 = ft.encode(ctx.x, train=True)
        W = ft.weight()
        H = H0
        for j in range(ft.T):
            H = ft.transition(H, H0, ctx.op, W, ctx, j, differentiable=True)
        z = ft.logits(H[ctx.q_idx])
        y = ctx.y_q
        ce = F.cross_entropy(z, y, reduction="none")
        logp = torch.log_softmax(z, 1)
        if variant == "ft_kl":
            reg = (pr * (torch.log(torch.clamp(pr, min=1e-12)) - logp)).sum(1)
            loss = ce.mean() + 1.0 * reg.mean()
        elif variant == "ft_dinf":
            dinf = (torch.log(torch.clamp(pr, min=1e-12)) - logp).max(1).values
            loss = ce.mean() + 1.0 * (F.relu(dinf - cfg.H_row) ** 2).mean() + 0.1 * dinf.mean()
        elif variant == "ft_pct":
            correct = (pr.argmax(1) == y).to(torch.float32)
            kd = (pr * (torch.log(torch.clamp(pr, min=1e-12)) - logp)).sum(1)
            loss = ce.mean() + 1.0 * (correct * kd).sum() / torch.clamp(correct.sum(), min=1.0)
        elif variant == "ft_regaware":
            ce_r = F.cross_entropy(torch.log(torch.clamp(pr, min=1e-12)), y, reduction="none")
            loss = ce.mean() + 1.0 * F.relu(ce - ce_r).mean()
        else:
            loss = ce.mean()
        opt.zero_grad()
        loss.backward()
        _api.clip_grad_norm(params, 5.0)
        opt.step()
        losses.append(float(loss.item()))
        if (s + 1) % max(1, steps // 10) == 0:
            _api.log(f"[FT {variant}] step {s+1}/{steps} loss={np.mean(losses[-50:]):.4f}")
    ft.eval()
    info = ft.freeze()
    for p in ft.parameters():
        p.requires_grad_(False)
    return ft, {"variant": variant, "steps": steps, "final_loss": float(np.mean(losses[-50:])) if losses else None, **info}


class LoRAIncumbent(nn.Module):
    """LoRA-16 weight delta on the transport and the head of the incumbent (a
    proposal whose states map exactly into the incumbent's state space)."""

    def __init__(self, base: _api.TanhDiffusionIncumbent, rank: int = 16, seed: int = 0):
        super().__init__()
        self.base = base
        d = base.hidden
        C = base.num_classes
        rng = np.random.default_rng(_api.stable_seed("lora", seed))
        self.Aw = nn.Parameter(torch.from_numpy((rng.standard_normal((d, rank)) * 0.01).astype(np.float32)))
        self.Bw = nn.Parameter(torch.zeros(rank, d))
        self.Ae = nn.Parameter(torch.from_numpy((rng.standard_normal((base.enc2.weight.shape[1], rank)) * 0.01).astype(np.float32)))
        self.Be = nn.Parameter(torch.zeros(rank, d))
        self.rank = rank
        self.T = base.T

    def states(self, ctx: _api.CallContext, differentiable: bool = False):
        b = self.base
        h = F.relu(b.enc1(ctx.x))
        H0 = b.enc2(h) + (h @ self.Ae) @ self.Be
        W = b.weight() + self.Aw @ self.Bw
        H = H0
        out = [H]
        for j in range(b.T):
            Z = ctx.op.mm(H) @ W
            H = b.alpha * H0 + (1.0 - b.alpha) * ((1.0 - b.tau) * H + b.tau * b.act(Z))
            H = ctx.apply_mask(H, j + 1, differentiable=differentiable)
            out.append(H)
        return out


class GPRGNN(nn.Module):
    """GPR-GNN (Chien et al., 2021): Z0 = MLP(X); H_k = sum_{s<=k} gamma_s P^s Z0."""

    def __init__(self, in_dim: int, hidden: int, C: int, K: int = 10, alpha: float = 0.1):
        super().__init__()
        self.mlp1 = nn.Linear(in_dim, hidden)
        self.mlp2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, C)
        g = alpha * (1 - alpha) ** np.arange(K + 1)
        g[-1] = (1 - alpha) ** K
        self.gamma = nn.Parameter(torch.from_numpy(g.astype(np.float32)))
        self.K = K
        self.hidden = hidden

    def states(self, x, op, train: bool = False):
        h = F.relu(self.mlp1(x))
        if train:
            h = F.dropout(h, 0.2, True)
        Z = self.mlp2(h)
        acc = self.gamma[0] * Z
        out = [acc]
        P = Z
        for k in range(1, self.K + 1):
            P = op.mm(P)
            acc = acc + self.gamma[k] * P
            out.append(acc)
        return out

    def logits(self, x, op, train: bool = False):
        return self.head(self.states(x, op, train)[-1])

