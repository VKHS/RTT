"""Fitting-only efficiency replay and budget placement."""
from __future__ import annotations

from ._imports import (
    Dict,
    List,
    math,
    np,
    time,
    torch,
)
from . import api as _api

# ----------------------------------------------------------------------------
# efficiency placement of caps, growth proxies, admissibility gate
# ----------------------------------------------------------------------------

def seeded_growth_ratios(system: _api.RTTSystem, ctx: _api.CallContext, H_row: float) -> Dict[int, np.ndarray]:
    """Growth-proxy data (App. 'Per-row budgets during execution'): the provisional tube
    with cap-seeded provisional factors along the incumbent trajectory, seeded with a
    unit radius at state depth j and propagated to T.  Returns {j: r_T / r_j per row}."""
    cfg = system.cfg
    inc = system.inc
    T = cfg.T
    seed_r = cfg.provisional_seed_frac * (H_row if math.isfinite(H_row) else 1.0) / system.gamma
    W = inc.weight()
    out: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        Ls = {}
        for j in range(cfg.open_first, T):
            _, Z = inc.transition(ctx.inc_states[j], ctx.H0, ctx.op, W, ctx, j, return_pre=True)
            rho = ctx.op.mm(torch.full((ctx.n,), seed_r, dtype=torch.float32, device=Z.device))
            L, _, _ = _api.interval_factors(Z, rho, system.split_prov, inc.activation, torch.float32, need_curvature=False)
            Ls[j] = L
        for j0 in range(cfg.open_first, T + 1):
            r = torch.ones(ctx.n, dtype=torch.float32, device=ctx.H0.device)
            for j in range(j0, T):
                r = _api.tube_step(ctx.op, r, Ls[j], ctx.keep(j + 1), None, inc.alpha, inc.tau)
            out[j0] = r.cpu().numpy()
    return out


def run_placement(system: _api.RTTSystem, env: _api.BaseEnv, cfg: _api.RTTConfig, episodes: int, seed: int) -> dict:
    """Efficiency placement.  On replayed <=2017 episodes (policy executed with uniform
    caps), every (depth, role) cell is scored by the best certified-quadratic net gain
    of a scaled step along one of its candidates, using full-loss slopes, per certified
    nat of contract charge.  Caps are then water-filled in order of efficiency until the
    expected per-call spend reaches expected_spend_frac * H+."""
    K = cfg.K_roles
    T = cfg.T
    gain = np.zeros((T, K))
    spend = np.zeros((T, K))
    cnt = np.zeros((T, K))
    real_gain = np.zeros((T, K))
    real_spend = np.zeros((T, K))
    ratios: Dict[int, List[np.ndarray]] = {l: [] for l in range(T + 1)}
    Lgm: Dict[int, List[float]] = {l: [] for l in cfg.open_depths()}
    spec = _api.PolicySpec(name="placement", kind="rtt", H_plus=cfg.H_plus, H_row=cfg.H_row, caps_key="caps_train")
    gen = env.generator
    t0 = time.time()
    for s in range(episodes):
        ctx = env.build(gen.make("placement", "fit", "mixture", s))
        system.prepare_call(ctx, time_prefix=False)

        def hook(info):
            l = info["l"]
            rows = info["rows"]
            g_true = _api.true_slope_field(system, ctx, l, info["Hbar"])[rows].to(torch.float64)
            D = info["D"].to(torch.float64)
            roles = info["roles"]
            lam_rows = info["lam"][rows].to(torch.float64)
            xi_rows = info["xi"][rows].to(torch.float64)
            M = D.shape[0]
            a_i = torch.einsum("mnd,nd->nm", D, g_true)                      # [n, M]
            nrm = torch.linalg.norm(D, dim=2).t()                              # [n, M]
            a_k = torch.zeros(K, M, dtype=torch.float64, device=D.device).index_add(0, roles, a_i)
            s_k = torch.zeros(K, M, dtype=torch.float64, device=D.device).index_add(0, roles, lam_rows.view(-1, 1) * nrm)
            q_k = torch.zeros(K, M, dtype=torch.float64, device=D.device).index_add(
                0, roles, xi_rows.view(-1, 1) * nrm * nrm) * (0.5 * (1.0 + cfg.rho_D) / cfg.omega)
            a_k, s_k, q_k = a_k.cpu().numpy(), s_k.cpu().numpy(), q_k.cpu().numpy()
            for k in range(K):
                best_g, best_s = 0.0, 0.0
                for m in range(M):
                    if s_k[k, m] <= 0 or a_k[k, m] >= 0:
                        continue
                    st = 1.0 if q_k[k, m] <= 0 else min(1.0, -a_k[k, m] / (2.0 * q_k[k, m]))
                    gnet = -a_k[k, m] * st - q_k[k, m] * st * st
                    if gnet > best_g:
                        best_g, best_s = gnet, s_k[k, m] * st
                if best_s > 0:
                    gain[l, k] += best_g
                    spend[l, k] += best_s
                cnt[l, k] += 1

        def post(info):
            l = info["l"]
            g_true = _api.true_slope_field(system, ctx, l, info["Hbar"])[info["rows"]]
            D = info["D"]
            roles = info["roles"]
            t = info["t"]
            lam_rows = info["lam"][info["rows"]]
            for k in info["active"]:
                sel = roles == k
                vk = torch.einsum("m,mnd->nd", torch.from_numpy(t[k].astype(np.float32)).to(D.device), D[:, sel])
                real_spend[l, k] += float((lam_rows[sel] * torch.linalg.norm(vk, dim=1)).sum().item())
                real_gain[l, k] += -float((vk * g_true[sel]).sum().item())

        res = system.run(ctx, spec, hook=hook, post_hook=post)
        if s < max(1, min(episodes, cfg.growth_calls)):
            for j, rr in seeded_growth_ratios(system, ctx, cfg.H_row).items():
                ratios.setdefault(j, []).append(rr)
        for l, rec in res.records.items():
            if rec.L_gm == rec.L_gm:
                Lgm.setdefault(l, []).append(rec.L_gm)
        if (s + 1) % max(1, episodes // 10) == 0:
            _api.log(f"[PLACEMENT] replay {s+1}/{episodes} ({time.time()-t0:.0f}s)")
    eff = np.where(spend > 0, gain / np.maximum(spend, 1e-300), -np.inf)
    growth = np.ones(T + 1)
    for l in range(T + 1):
        if ratios.get(l):
            growth[l] = float(np.quantile(np.concatenate(ratios[l]), cfg.growth_quantile))
    Lbar = {l: float(np.quantile(v, cfg.growth_quantile)) for l, v in Lgm.items() if v}
    Pi = _api.scalar_growth({l: Lbar.get(l, system.w_norm) for l in range(T)}, cfg.alpha, cfg.tau, T)
    admissible = {l: bool(Pi.get(l, 1.0) <= cfg.Pi_max) for l in cfg.open_depths()}
    budget = cfg.expected_spend_frac * cfg.H_plus
    per_call_spend = spend / max(1, episodes)
    cells = [(eff[l, k], l, k) for l in cfg.open_depths() for k in range(K)
             if admissible.get(l, False) and np.isfinite(eff[l, k]) and eff[l, k] > 0]
    cells.sort(reverse=True)
    caps = np.zeros((T, K))
    used = 0.0
    water = None
    for e, l, k in cells:
        s_ = per_call_spend[l, k]
        if used + s_ <= budget or used == 0.0:
            caps[l, k] = cfg.cap_max
            used += s_
        else:
            frac = max(0.0, (budget - used) / max(s_, 1e-300))
            if frac > 0:
                caps[l, k] = cfg.cap_max * min(1.0, frac)
                used += s_ * min(1.0, frac)
            water = e
            break
    caps_spend = np.zeros((T, K))
    smax = spend[cfg.open_first:].max() if spend[cfg.open_first:].size else 0.0
    if smax > 0:
        for l in cfg.open_depths():
            if admissible.get(l, False):
                caps_spend[l] = cfg.cap_max * spend[l] / smax
    tot = caps * per_call_spend
    ssum = float(tot.sum())
    share = {l: (float(tot[l].sum() / ssum) if ssum > 0 else 0.0) for l in cfg.open_depths()}
    tot_s = caps_spend * per_call_spend
    ssum_s = float(tot_s.sum())
    share_spend = {l: (float(tot_s[l].sum() / ssum_s) if ssum_s > 0 else 0.0) for l in cfg.open_depths()}
    return {"caps": caps, "caps_spend": caps_spend, "growth": growth, "trust_share": share,
            "trust_share_spend": share_spend, "efficiency": eff, "gain": gain, "spend": spend,
            "realized_gain": real_gain, "realized_spend": real_spend,
            "admissible": admissible, "Pi": Pi, "Lbar_q95": Lbar, "water_level": water,
            "expected_spend_used": used, "episodes": episodes}

