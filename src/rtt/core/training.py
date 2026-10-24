"""Proposal, admission-head, and comparator fitting."""
from __future__ import annotations

from ._imports import (
    Any,
    F,
    List,
    Optional,
    TORCH_AVAILABLE,
    Tuple,
    np,
    time,
    torch,
)
from . import api as _api

# ============================================================================
# Section 12: phase-two training (App. B), external proposals and efficiency
#             placement of caps (App. E, 'Efficiency and placement')
# ============================================================================

def uniform_caps(cfg: _api.RTTConfig) -> np.ndarray:
    caps = np.zeros((cfg.T, cfg.K_roles), dtype=np.float64)
    caps[cfg.open_first:cfg.T] = cfg.cap_init
    return caps


def train_native_proposal(system: _api.RTTSystem, env: _api.BaseEnv, cfg: _api.RTTConfig, steps: int, seed: int,
                          pool: str = "fit") -> dict:
    """The learned update: minimise terminal CE of the raw update (t = 1 on both
    norm-trimmed candidates at every open depth) on <=2017 episodes."""
    _api.set_global_seeds(seed)
    prop = system.proposal
    inc = system.inc
    for p in prop.parameters():
        p.requires_grad_(True)
    opt = _api.AdamW(prop.parameters(), lr=cfg.lr_proposal, weight_decay=1e-5)
    gen = env.generator
    losses, gains = [], []
    t0 = time.time()
    for s in range(steps):
        ctx = env.build(gen.make("phase2_proposal", pool, "mixture", s))
        system.prepare_call(ctx, time_prefix=False)
        W = inc.weight()
        H = ctx.inc_states[cfg.open_first].detach()
        for l in range(cfg.open_first, cfg.T):
            Hbar = inc.transition(H, ctx.H0, ctx.op, W, ctx, l, differentiable=True)
            cand = _api.norm_trim(prop.candidates(Hbar, l, ctx), cfg.row_trim)
            H = Hbar + cand.sum(0)
        z = inc.logits(H[ctx.q_idx])
        loss = F.cross_entropy(z, ctx.y_q)
        with torch.no_grad():
            base = F.cross_entropy(ctx.inc_logits[ctx.q_idx], ctx.y_q)
        opt.zero_grad()
        loss.backward()
        _api.clip_grad_norm(list(prop.parameters()), 5.0)
        opt.step()
        losses.append(float(loss.item()))
        gains.append(float(base.item() - loss.item()))
        if (s + 1) % max(1, steps // 10) == 0:
            _api.log(f"[PROPOSAL] step {s+1}/{steps} CE={np.mean(losses[-100:]):.4f} raw gain vs incumbent={np.mean(gains[-100:]):+.5f} ({time.time()-t0:.0f}s)")
    for p in prop.parameters():
        p.requires_grad_(False)
    prop.eval()
    return {"steps": steps, "final_ce": float(np.mean(losses[-100:])), "final_raw_gain": float(np.mean(gains[-100:]))}


@torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
def compute_role_thresholds(system: _api.RTTSystem, env: _api.BaseEnv, cfg: _api.RTTConfig, n_calls: int = 32) -> List[float]:
    """Label-free quartiles of logit-lens confidence at open depths on fitting calls."""
    confs = []
    gen = env.generator
    for s in range(n_calls):
        ctx = env.build(gen.make("role_thresholds", "fit", "mixture", s), need_labels=False)
        system.prepare_call(ctx, time_prefix=False)
        for l in cfg.open_depths()[:: max(1, len(cfg.open_depths()) // 4)]:
            p = torch.softmax(system.inc.logits(ctx.inc_states[l + 1]), 1)
            c = p.max(1).values
            idx = np.random.default_rng(_api.stable_seed("rt", s, l)).choice(ctx.n, size=min(ctx.n, 20000), replace=False)
            confs.append(c[torch.from_numpy(idx).to(c.device)].cpu().numpy())
    allc = np.concatenate(confs)
    return [float(np.quantile(allc, q)) for q in (0.25, 0.5, 0.75)]


def _block_x_coords(Gt_k, vecs: List[Any], mats: List[Any]):
    """Physical coordinates of a block from its Gram: x = Lambda^{1/2} V^T t.  Returns
    transformed vectors/matrices (slope vectors via Lambda^{-1/2} V^T, covariances
    via Lambda^{-1/2} V^T S V Lambda^{-1/2})."""
    ev, V = torch.linalg.eigh(0.5 * (Gt_k + Gt_k.t()))
    if float(ev.max().item()) <= 0:
        return None
    keep = ev > 1e-12 * ev.max()
    Vr = V[:, keep]
    lr = ev[keep]
    Tm = Vr / torch.sqrt(lr).unsqueeze(0)                          # M x r
    return [Tm.t() @ v for v in vecs], [Tm.t() @ m @ Tm for m in mats]


def true_slope_field(system: _api.RTTSystem, ctx: _api.CallContext, l: int, Hbar, mode: str = "full"):
    """Gradient of the incumbent's continuation loss at H-bar_{l+1} (full-loss
    target) or of the logit-lens probe loss at the same state (probe ablation)."""
    inc = system.inc
    with torch.no_grad():
        if mode == "probe":
            return _api.terminal_ce_grad(inc, ctx, Hbar)
        states, derivs = _api.tail_forward(inc, ctx, Hbar, l + 1)
        G_T = _api.terminal_ce_grad(inc, ctx, states[-1])
        return _api.tail_vjp(inc, ctx, derivs, l + 1, G_T)


def train_heads(system: _api.RTTSystem, env: _api.BaseEnv, cfg: _api.RTTConfig, steps: int, seed: int,
                heads_key: str = "heads", feature_mode: str = "support", slope_mode: str = "full",
                pool: str = "fit") -> dict:
    """Phase two: heads trained on trajectories executed by the current policy
    (through the executed gate) with full-loss slope targets."""
    _api.set_global_seeds(seed)
    heads = system.heads if heads_key == "heads" else system.extra_heads[heads_key]
    for p in heads.parameters():
        p.requires_grad_(True)
    opt = _api.AdamW(heads.parameters(), lr=cfg.lr_heads, weight_decay=cfg.wd_heads)
    gen = env.generator
    K = cfg.K_roles
    spec = _api.PolicySpec(name="train", kind="rtt", H_plus=cfg.H_plus, H_row=cfg.H_row, radii=None,
                      caps_key="caps_train", feature_mode=feature_mode, heads_key=heads_key)
    stats = {"field": [], "post": [], "cov": [], "theta": [], "rel_err": []}
    t0 = time.time()
    A32 = system.A32
    for s in range(steps):
        ctx = env.build(gen.make("phase2_heads", pool, "mixture", s))
        system.prepare_call(ctx, time_prefix=False, prefix_only=True)

        def hook(info):
            l = info["l"]
            di = info["di"]
            rows = info["rows"]
            g_true_full = true_slope_field(system, ctx, l, info["Hbar"], slope_mode)
            g_true = g_true_full[rows]
            feat = info["feat"].detach()
            feat_pre = info["feat_pre"].detach()
            lens_rows = info["lens_rows"].detach()
            pB = heads.posterior(feat, lens_rows)
            resid = (lens_rows - pB) @ A32.t()
            sc = ctx.scored[rows]
            srow = rows[sc]
            E0 = torch.zeros(ctx.n, system.d, dtype=torch.float32, device=pB.device)
            E0 = E0.index_put((srow,), ctx.w[srow].unsqueeze(1) * resid[sc])
            E1 = ctx.op.tmm(E0)
            E2 = ctx.op.tmm(E1)
            adj = info.get("adj")
            ghat = _api.field_estimate(heads, feat, di, rows, [E0, E1, E2], resid, info["lam"][rows].detach(), system.gamma,
                                  adj_rows=None if adj is None else adj[rows])
            D = info["D"].detach()
            roles = info["roles"]
            M = D.shape[0]
            ah = torch.einsum("mnd,nd->nm", D, ghat)
            at = torch.einsum("mnd,nd->nm", D, g_true)
            a_hat = torch.zeros(K, M, device=D.device).index_add(0, roles, ah)
            a_true = torch.zeros(K, M, device=D.device).index_add(0, roles, at)
            denom = (a_true * a_true).sum() + 1e-30
            L_field = ((a_hat - a_true) ** 2).sum() / denom
            # posterior CE on scored rows
            y_rows = torch.full((rows.shape[0],), -1, dtype=torch.long, device=D.device)
            pos = torch.full((ctx.n,), -1, dtype=torch.long, device=D.device)
            pos[ctx.q_idx] = torch.arange(ctx.q_idx.shape[0], device=D.device)
            ridx = pos[rows]
            has = ridx >= 0
            L_post = torch.zeros((), device=D.device)
            if bool(has.any().item()):
                yy = ctx.y_q[ridx[has]]
                L_post = F.nll_loss(torch.log(torch.clamp(pB[has], min=1e-12)), yy)
            # covariance NLL in physical coordinates (block-diagonal)
            unit2 = (system.gamma / float(ctx.spec.nq)) ** 2
            sig2 = heads.variance(feat, di, unit2, cfg.cov_floor_rel)
            D64 = D.to(torch.float64)
            Gi = torch.einsum("mnd,pnd->nmp", D64, D64)
            Gt = torch.zeros(K, M, M, dtype=torch.float64, device=D.device).index_add(0, roles, Gi)
            St = torch.zeros(K, M, M, dtype=torch.float64, device=D.device).index_add(
                0, roles, sig2.to(torch.float64).view(-1, 1, 1) * Gi) + cfg.cov_reg_rel * unit2 * Gt
            L_cov = torch.zeros((), dtype=torch.float64, device=D.device)
            err = (a_true - a_hat).detach().to(torch.float64)
            nb = 0
            for k in range(K):
                if float(torch.trace(Gt[k]).item()) <= 0:
                    continue
                tr = _block_x_coords(Gt[k].detach(), [err[k]], [St[k]])
                if tr is None:
                    continue
                (ex,), (Sx,) = tr
                Sx = 0.5 * (Sx + Sx.t()) + 1e-30 * torch.eye(Sx.shape[0], dtype=Sx.dtype, device=Sx.device)
                sol = torch.linalg.solve(Sx, ex)
                L_cov = L_cov + 0.5 * (ex * sol).sum() + 0.5 * torch.logdet(Sx / unit2)
                nb += 1
            if nb:
                L_cov = L_cov / nb
            # node weight: importance-weighted BCE on the sign of the true descent
            theta = heads.node_weight(feat_pre)
            dsc = torch.einsum("mnd,nd->n", info["cand_nt"].detach(), g_true)
            tgt = (dsc < 0).to(torch.float32)
            wts = dsc.abs() / (dsc.abs().mean() + 1e-30)
            L_theta = (wts * F.binary_cross_entropy(torch.clamp(theta, 1e-6, 1 - 1e-6), tgt, reduction="none")).mean()
            loss = L_field + L_post + 0.1 * L_cov.to(torch.float32) + 0.5 * L_theta
            loss.backward()
            stats["field"].append(float(L_field.item()))
            stats["post"].append(float(L_post.item()))
            stats["cov"].append(float(L_cov.item()))
            stats["theta"].append(float(L_theta.item()))

        opt.zero_grad()
        system.run(ctx, spec, hook=hook, keep_records=False)
        _api.clip_grad_norm(list(heads.parameters()), 5.0)
        opt.step()
        if (s + 1) % max(1, steps // 10) == 0:
            _api.log(f"[HEADS {heads_key}] step {s+1}/{steps} "
                f"field={np.mean(stats['field'][-200:]):.4f} "
                f"post={np.mean(stats['post'][-200:]):.4f} cov={np.mean(stats['cov'][-200:]):.3f} "
                f"theta={np.mean(stats['theta'][-200:]):.4f} ({time.time()-t0:.0f}s)")
    for p in heads.parameters():
        p.requires_grad_(False)
    heads.eval()
    return {k: (float(np.mean(v[-200:])) if v else None) for k, v in stats.items() if k != "rel_err"}


def train_tr_head(system: _api.RTTSystem, env: _api.BaseEnv, cfg: _api.RTTConfig, steps: int, seed: int, mode: str) -> dict:
    """Matched TR (supervised oracle trust) or TR trained end to end under the budgets
    (first-order realised change through the executed trust step)."""
    _api.set_global_seeds(seed)
    key = "tr_matched" if mode == "matched" else "tr_e2e"
    head = system.tr_heads[key]
    for p in head.parameters():
        p.requires_grad_(True)
    opt = _api.AdamW(head.parameters(), lr=cfg.lr_heads, weight_decay=cfg.wd_heads)
    spec = _api.PolicySpec(name=f"train_{key}", kind="tr", H_plus=cfg.H_plus, H_row=cfg.H_row, caps_key="caps_train",
                      tr_key=key)
    gen = env.generator
    K = cfg.K_roles
    losses = []
    for s in range(steps):
        ctx = env.build(gen.make(f"phase2_{key}", "fit", "mixture", s))
        system.prepare_call(ctx, time_prefix=False)

        def hook(info):
            l = info["l"]
            rows = info["rows"]
            roles = info["roles"]
            D = info["D"].detach()
            M = D.shape[0]
            g_true = true_slope_field(system, ctx, l, info["Hbar"])[rows]
            feat = info["feat"].detach()
            bf = torch.zeros(K, feat.shape[1], device=feat.device).index_add(0, roles, feat)
            cnt = torch.bincount(roles, minlength=K).to(torch.float32)
            bf = bf / torch.clamp(cnt, min=1.0).unsqueeze(1)
            capr = info["cap_rows"]
            u_rows = D.sum(0) * (capr / M).unsqueeze(1)                     # full-trust step per row
            a_rows = (u_rows * g_true).sum(1)
            q_rows = info["xi"][rows] * (u_rows * u_rows).sum(1)
            a_k = torch.zeros(K, device=D.device).index_add(0, roles, a_rows)
            q_k = torch.zeros(K, device=D.device).index_add(0, roles, q_rows)
            ids = torch.nonzero(cnt > 0, as_tuple=False).squeeze(1)
            if ids.numel() == 0:
                return
            tau = head(bf[ids], ids, cnt[ids])
            if mode == "matched":
                tgt = torch.where(a_k[ids] < 0, torch.clamp(-a_k[ids] / torch.clamp(q_k[ids], min=1e-30), 0.0, 1.0),
                                  torch.zeros_like(a_k[ids]))
                loss = F.binary_cross_entropy(torch.clamp(tau, 1e-6, 1 - 1e-6), tgt.detach())
            else:
                scale = a_k[ids].abs().sum() + 1e-30
                loss = ((tau * a_k[ids] + 0.5 * tau * tau * q_k[ids]) / scale).sum()
            loss.backward()
            losses.append(float(loss.item()))

        opt.zero_grad()
        system.run(ctx, spec, hook=hook, keep_records=False)
        _api.clip_grad_norm(list(head.parameters()), 5.0)
        opt.step()
        if (s + 1) % max(1, steps // 10) == 0:
            _api.log(f"[TR {key}] step {s+1}/{steps} loss={np.mean(losses[-200:]) if losses else float('nan'):.5f}")
    for p in head.parameters():
        p.requires_grad_(False)
    head.eval()
    return {"mode": mode, "steps": steps, "final_loss": float(np.mean(losses[-200:])) if losses else None}


CORRECTOR_SPECS = [
    ("corr_2hop_rows", 2, True),
    ("corr_2hop", 2, False),
    ("corr_per_row", 0, False),
    ("corr_4hop_rows", 4, True),
    ("corr_8hop_rows", 8, True),
]


def train_correctors(system: _api.RTTSystem, env: _api.BaseEnv, cfg: _api.RTTConfig, steps: int, seed: int,
                     which: Optional[List[str]] = None) -> Tuple[dict, dict]:
    models, info = {}, {}
    P2 = 2 * cfg.probes_per_sign
    specs = [env.generator.make("phase2_heads", "fit", "mixture", s) for s in range(steps)]
    for name, hops, rows in CORRECTOR_SPECS:
        if which is not None and name not in which:
            continue
        _api.set_global_seeds(_api.stable_seed(seed, name) % (2 ** 31))
        width, target = _api.matched_corrector_width(system.d, system.C, hops, rows, cfg)
        m = _api.OnePassCorrector(system.d, system.C, hops, rows, P2, width, name).to(system.device)
        info[name] = _api.train_classifier_on_calls(system, env, m, specs, steps, cfg.lr_heads, cfg.wd_heads, log_tag=name)
        count = sum(p.numel() for p in m.parameters())
        info[name].update({"hidden_width": width, "parameters": count,
                           "rtt_head_parameters": target, "parameter_mismatch_fraction": abs(count - target) / target,
                           "training_call_generator": "phase2_heads"})
        for p in m.parameters():
            p.requires_grad_(False)
        models[name] = m
    return models, info


def train_crossfit_posterior(system: _api.RTTSystem, env: _api.BaseEnv, roles: _api.DataRoles, cfg: _api.RTTConfig, seed: int) -> Tuple[_api.CrossFitPosterior, dict]:
    nodes = roles.pools["calib18"]
    K = int(cfg.crossfit_folds)
    fold_of = np.full(env.g.n, -1, dtype=np.int64)
    perm = np.random.default_rng(_api.stable_seed("crossfit", seed)).permutation(nodes)
    for f in range(K):
        fold_of[perm[f::K]] = f
    fold_t = torch.from_numpy(fold_of).to(system.device)
    P2 = 2 * cfg.probes_per_sign
    models = []
    specs = [env.generator.make("crossfit", "calib18", "mixture", s) for s in range(cfg.crossfit_episodes)]
    for f in range(K):
        _api.set_global_seeds(_api.stable_seed(seed, "cf", f) % (2 ** 31))
        m = _api.OnePassCorrector(system.d, system.C, 4, True, P2, 2 * cfg.head_hidden, f"crossfit_{f}").to(system.device)
        _api.train_classifier_on_calls(system, env, m, specs, cfg.crossfit_episodes, cfg.lr_heads, cfg.wd_heads,
                                  node_filter=lambda ctx, f=f: fold_t[ctx.q_idx] != f, log_tag=f"crossfit{f}")
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    # temperature from out-of-fold predictions
    cf = _api.CrossFitPosterior(models, 1.0)
    lps, ys = [], []
    for s in range(min(64, len(specs))):
        ctx = env.build(env.generator.make("crossfit_temp", "calib18", "mixture", s))
        system.prepare_call(ctx, time_prefix=False)
        with torch.no_grad():
            H_T = ctx.inc_states[-1]
            zl = system.inc.logits(H_T)
            pf = torch.softmax(zl, 1)
            zq = zl[ctx.q_idx]
            fq = fold_t[ctx.q_idx]
            for f, m in enumerate(models):
                sel = fq == f
                if not bool(sel.any().item()):
                    continue
                lp = torch.log(torch.clamp(m(m.features(system, ctx, H_T, pf), zq), min=1e-12))
                lps.append(lp[sel])
                ys.append(ctx.y_q[sel])
    temp = _api.fit_scalar_temperature(lps, ys) if lps else 1.0
    cf.temperature = temp
    return cf, {"folds": K, "temperature": temp, "episodes_per_fold": cfg.crossfit_episodes}


def train_lora(base: _api.TanhDiffusionIncumbent, env: _api.BaseEnv, cfg: _api.RTTConfig, steps: int, seed: int) -> Tuple[_api.LoRAIncumbent, dict]:
    _api.set_global_seeds(seed)
    lora = _api.LoRAIncumbent(base, 16, seed).to(env.device)
    params = [lora.Aw, lora.Bw, lora.Ae, lora.Be]
    for p in params:
        p.requires_grad_(True)
    opt = _api.AdamW(params, lr=1e-3, weight_decay=0.0)
    losses = []
    for s in range(steps):
        ctx = env.build(env.generator.make("lora", "fit", "clean", s))
        st = lora.states(ctx, differentiable=True)
        z = base.logits(st[-1][ctx.q_idx])
        loss = F.cross_entropy(z, ctx.y_q)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    for p in params:
        p.requires_grad_(False)
    return lora, {"steps": steps, "final_loss": float(np.mean(losses[-50:])) if losses else None}


def train_gprgnn(env: _api.BaseEnv, roles: _api.DataRoles, cfg: _api.RTTConfig, epochs: int, seed: int) -> Tuple[_api.GPRGNN, dict]:
    _api.set_global_seeds(seed)
    m = _api.GPRGNN(env.g.num_features, cfg.hidden, env.g.num_classes).to(env.device)
    opt = _api.AdamW(m.parameters(), lr=5e-3, weight_decay=5e-4)
    tr = torch.from_numpy(roles.pools["inc_train"]).to(env.device)
    ho = torch.from_numpy(roles.pools["inc_holdout"]).to(env.device)
    best = (-1.0, None)
    for ep in range(epochs):
        m.train()
        lg = m.logits(env.x, env.clean_op, train=True)
        loss = F.cross_entropy(lg[tr], env.y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            m.eval()
            with torch.no_grad():
                acc = float((m.logits(env.x, env.clean_op).argmax(1)[ho] == env.y[ho]).float().mean().item())
            if acc > best[0]:
                best = (acc, {k: v.detach().clone() for k, v in m.state_dict().items()})
    if best[1] is not None:
        m.load_state_dict(best[1])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, {"epochs": epochs, "holdout_acc": best[0]}


def gpr_depth_map(l_next: int, T: int, K: int) -> int:
    return int(round(K * l_next / T))


def build_state_proposal(name: str, system_inc: _api.TanhDiffusionIncumbent, env: _api.BaseEnv, cfg: _api.RTTConfig, source_obj,
                         n_fit_calls: int = 8) -> _api.StateProposal:
    """Wrap an external model as a StateProposal with least-squares adapters fitted
    on <=2017 calls (identity adapters when coordinates match)."""
    d = system_inc.hidden
    T = cfg.T
    if name in ("v2", "lora16"):
        src_dim = d
    else:
        src_dim = source_obj.hidden
    prop = _api.StateProposal(name, src_dim, d, T).to(env.device)
    if name == "v2":
        def src(ctx, m=source_obj):
            H0 = m.encode(ctx.x)
            return _api.incumbent_rollout(m, H0, ctx.op, masks=ctx)
    elif name == "lora16":
        def src(ctx, m=source_obj):
            return m.states(ctx)
    elif name == "gprgnn":
        def src(ctx, m=source_obj):
            st = m.states(ctx.x, ctx.op)
            return [st[gpr_depth_map(j, T, m.K)] for j in range(T + 1)]
    else:
        raise ValueError(name)
    prop.source = src
    if src_dim != d:
        rows = torch.from_numpy(env.roles.pools["inc_train"][:50000]).to(env.device)
        srcs, tgts = [], []
        with torch.no_grad():
            for s in range(n_fit_calls):
                ctx = env.build(env.generator.make(f"adapter_{name}", "fit", "clean", s), need_labels=False)
                ctx.H0 = system_inc.encode(ctx.x)
                tg = _api.incumbent_rollout(system_inc, ctx.H0, ctx.op, masks=ctx)
                srcs.append(src(ctx))
                tgts.append(tg)
        Om = _api.fit_adapters_least_squares(srcs, tgts, rows, list(range(1, T + 1)))
        with torch.no_grad():
            for l, M_ in Om.items():
                prop.Omega[l].copy_(M_)
    prop.eval()
    return prop


