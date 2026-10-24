"""Sequential physical-span conformal calibration."""
from __future__ import annotations

from ._imports import (
    Dict,
    List,
    Optional,
    Sequence,
    math,
    np,
    time,
    torch,
)
from . import api as _api

# ============================================================================
# Section 13: sequential radius calibration (Sec. 3.5, Lemma 6) and the
#             evaluation engine (paired records for every policy)
# ============================================================================

def conformal_quantile(scores: Sequence[float], delta: float) -> float:
    """ceil((1-delta)(n+1))-th order statistic, +inf when the index exceeds n."""
    n = len(scores)
    if n == 0 or delta <= 0:
        return float("inf")
    k = int(math.ceil((1.0 - delta) * (n + 1)))
    if k > n:
        return float("inf")
    return float(sorted(scores)[k - 1])


def calibration_score(system: _api.RTTSystem, ctx: _api.CallContext, info: dict, caps_row_level=None) -> float:
    """Joint physical-coordinate Mahalanobis error of the slope at the executed state."""
    cfg = system.cfg
    K = cfg.K_roles
    l = info["l"]
    rows = info["rows"]
    D = info["D"].to(torch.float64)
    roles = info["roles"]
    g_true = _api.true_slope_field(system, ctx, l, info["Hbar"])[rows].to(torch.float64)
    g_hat = info["g"].to(torch.float64)
    sig2 = info["sig2"].to(torch.float64)
    capr = info["cap_rows"]
    M = D.shape[0]
    Gi = torch.einsum("mnd,pnd->nmp", D, D)
    at = torch.zeros(K, M, dtype=torch.float64, device=D.device).index_add(0, roles, torch.einsum("mnd,nd->nm", D, g_true))
    ah = torch.zeros(K, M, dtype=torch.float64, device=D.device).index_add(0, roles, torch.einsum("mnd,nd->nm", D, g_hat))
    Gt = torch.zeros(K, M, M, dtype=torch.float64, device=D.device).index_add(0, roles, Gi)
    unit2 = (system.gamma / float(ctx.spec.nq)) ** 2
    St = torch.zeros(K, M, M, dtype=torch.float64, device=D.device).index_add(0, roles, sig2.view(-1, 1, 1) * Gi) \
        + cfg.cov_reg_rel * unit2 * Gt
    cap_k = torch.zeros(K, dtype=torch.float64, device=D.device).index_add(0, roles, capr.to(torch.float64))
    s2 = 0.0
    for k in range(K):
        if float(cap_k[k].item()) <= 0 or float(torch.trace(Gt[k]).item()) <= 0:
            continue
        tr = _api._block_x_coords(Gt[k], [at[k] - ah[k]], [St[k]])
        if tr is None:
            continue
        (ex,), (Sx,) = tr
        Sx = 0.5 * (Sx + Sx.t())
        s2 += float((ex * torch.linalg.solve(Sx, ex)).sum().item())
    return math.sqrt(max(s2, 0.0))


def calibrate_member(system: _api.RTTSystem, env: _api.BaseEnv, cfg: _api.RTTConfig, member: _api.MemberSpec, caps_key: str,
                     trust_share: Dict[int, float], n_r: int, stage_tag: str, pool: str = "calib18",
                     heads_key: str = "heads", overrides: Optional[dict] = None) -> dict:
    caps = system.caps_for(_api.PolicySpec(name="c", caps_key=caps_key))
    base_open = list(overrides.get("open_depths")) if overrides and isinstance(overrides.get("open_depths"), list) \
        else cfg.open_depths()
    depths = [l for l in base_open if caps[l].sum() > 0]

    def make_policy(radii):
        pol = _api.PolicySpec(name="calib", kind="rtt", H_plus=H_plus, H_row=cfg.H_row, radii=radii,
                         caps_key=caps_key, heads_key=heads_key)
        for k, v in (overrides or {}).items():
            if k in ("H_plus",):
                continue
            setattr(pol, k, v)
        return pol
    gen = env.generator
    H_plus = float((overrides or {}).get("H_plus", member.H_plus(cfg)))
    t0 = time.time()
    all_scores: Dict[int, List[float]] = {}
    radii: Dict[int, float] = {}
    if member.radii_mode == "per_depth":
        for l in depths:
            all_scores[l] = []
        for e in range(n_r):
            ctx = env.build(gen.make(f"{stage_tag}:perdepth", pool, "mixture", e))
            system.prepare_call(ctx, time_prefix=False, prefix_only=True)

            def hook(info):
                all_scores[info["l"]].append(calibration_score(system, ctx, info))
            pol = make_policy({l: float("inf") for l in depths})
            system.run(ctx, pol, hook=hook, keep_records=False)
        for l in depths:
            radii[l] = conformal_quantile(all_scores[l], member.per_depth_q)
    else:
        for l in depths:
            pi_l = float(trust_share.get(l, trust_share.get(str(l), 0.0)))
            dl = member.delta_cov * pi_l
            scores: List[float] = []
            if dl <= 0:
                radii[l] = float("inf")
                all_scores[l] = scores
                continue
            for e in range(n_r):
                ctx = env.build(gen.make(f"{stage_tag}:d{l}", pool, "mixture", e))
                system.prepare_call(ctx, time_prefix=False, prefix_only=True)
                box = {}

                def hook(info, l=l, box=box, ctx=ctx):
                    if info["l"] == l:
                        box["s"] = calibration_score(system, ctx, info)
                pol = make_policy({**radii, l: float("inf")})
                system.run(ctx, pol, hook=hook, stop_after=l, keep_records=False)
                scores.append(float(box.get("s", 0.0)))
            radii[l] = conformal_quantile(scores, dl)
            all_scores[l] = scores
            _api.log(f"[CALIB {member.name}] depth {l}: delta_l={dl:.4g} n={len(scores)} kappa={radii[l]:.4g} "
                f"median score={np.median(scores) if scores else float('nan'):.4g} ({time.time()-t0:.0f}s)")
    return {"member": member.name, "radii": {str(k): v for k, v in radii.items()},
            "scores": {str(k): v for k, v in all_scores.items()}, "n_r": n_r,
            "delta_cov": member.delta_cov, "mode": member.radii_mode}


