"""Served-policy evaluation, charged timing, and episode records."""
from __future__ import annotations

from ._imports import (
    Any,
    Dict,
    List,
    Optional,
    Path,
    Sequence,
    dataclass,
    field,
    gzip,
    json,
    math,
    np,
    os,
    time,
    torch,
    traceback,
)
from . import api as _api

# ----------------------------------------------------------------------------
# evaluation engine
# ----------------------------------------------------------------------------

@dataclass
class PolicyDef:
    name: str
    kind: str
    spec: Optional[_api.PolicySpec] = None
    params: Dict[str, Any] = field(default_factory=dict)


def _served_metrics(ctx: _api.CallContext, logp_r, logp_s, y, zeta: float) -> dict:
    """Paired call-level outcomes from float64 log-probabilities on scored rows."""
    ce_r = -logp_r.gather(1, y.view(-1, 1)).squeeze(1)
    ce_s = -logp_s.gather(1, y.view(-1, 1)).squeeze(1)
    down = float((ce_s - ce_r).mean().item())
    pr_pred = logp_r.argmax(1)
    ps_pred = logp_s.argmax(1)
    cor_r = pr_pred == y
    cor_s = ps_pred == y
    nq = y.shape[0]
    neg = float((cor_r & ~cor_s).sum().item()) / nq
    pos = float((~cor_r & cor_s).sum().item()) / nq
    dis = float((pr_pred != ps_pred).sum().item()) / nq
    acc_r = float(cor_r.float().mean().item())
    acc_s = float(cor_s.float().mean().item())
    dinf_rows = (logp_r - logp_s).max(1).values
    wq = 1.0 / nq
    return {"down": down, "gain": -down, "regress": bool(down > zeta), "acc_r": acc_r, "acc_s": acc_s,
            "acc_change_pp": 100.0 * (acc_s - acc_r), "neg_flip_frac": neg, "pos_flip_frac": pos,
            "disagree_frac": dis, "exact_S_plus": float(dinf_rows.sum().item()) * wq,
            "max_row_dinf": float(dinf_rows.max().item()), "_ce_r": ce_r, "_ce_s": ce_s, "_dinf_rows": dinf_rows,
            "_cor_r": cor_r, "_cor_s": cor_s}


def logp_from_logits(z):
    return torch.log_softmax(z.to(torch.float64), dim=1)


def logp_from_probs(p):
    return torch.log(torch.clamp(p.to(torch.float64), min=1e-300))


class Evaluator:
    """Serves paired policies and records a fresh-call component timing sum.

    The audit may reuse predictions, but every method is charged the measured
    cost of its required cached computation. Graph construction is reported
    separately. These component sums are not independent cold-start benchmarks.
    """

    def __init__(self, system: _api.RTTSystem, checker: _api.Checker, env: _api.BaseEnv, cfg: _api.RTTConfig, models: dict,
                 flops: Optional[dict] = None):
        self.sys = system
        self.checker = checker
        self.env = env
        self.cfg = cfg
        self.models = models
        self.flops = flops or {}

    def evaluate_call(self, spec: _api.CallSpec, policies: List[PolicyDef], extras: Sequence[str] = ()) -> List[dict]:
        cfg = self.cfg
        sys_ = self.sys
        ctx = self.env.build(spec)
        sys_.prepare_call(ctx)
        y = ctx.y_q
        q = ctx.q_idx
        t_inc = ctx.timing["incumbent_pass_s"]
        t_features = float(ctx.timing.get("prepare_features_s", 0.0))
        t_proposal = float(ctx.timing.get("proposal_prepare_s", 0.0))
        logp_r = logp_from_logits(ctx.inc_logits[q])
        cf = self.models.get("crossfit")
        t_crossfit = 0.0
        if cf is not None:
            _api.cuda_sync()
            t_cf_start = time.perf_counter()
            ctx.cache["crossfit_posterior"] = cf.predict(sys_, ctx)
            _api.cuda_sync()
            t_crossfit = time.perf_counter() - t_cf_start
        out: List[dict] = []
        shared: Dict[str, Any] = {}

        def base_rec(pd: PolicyDef) -> dict:
            return {"call_id": spec.call_id, "stage": spec.stage, "pool": spec.pool, "component": spec.component,
                    "generator_component": spec.call_id.rsplit(":", 2)[-2],
                    "index": spec.index, "nq": spec.nq, "policy": pd.name, "kind": pd.kind,
                    "t_incumbent_s": t_inc, "certificate_status": _api.CERTIFICATE_STATUS,
                    "timing_convention": "fresh_call_component_sum_cached_work_recharged",
                    "graph_build_s": float(ctx.timing.get("build_s", 0.0))}

        def raw_update():
            if "raw" not in shared:
                pol = _api.PolicySpec(name="raw", kind="raw", certify=False)
                _, res, _, timing = _api.serve(sys_, None, ctx, pol)
                shared["raw"] = (res, timing["total_s"])
            return shared["raw"]

        def plugin_q():
            if "plugin_q" not in shared:
                corr = self.models["correctors"]["corr_2hop_rows"]
                # Measure the complete proposal-row feature dependency once,
                # even if an earlier comparator populated the shared cache.
                ctx.cache.pop("proposal_rows_feat", None)
                _api.cuda_sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    H_T = ctx.inc_states[-1]
                    zl = sys_.inc.logits(H_T)
                    pf = torch.softmax(zl, 1)
                    qd = corr(corr.features(sys_, ctx, H_T, pf), zl[q]).to(torch.float64)
                _api.cuda_sync()
                shared["plugin_q"] = (qd, time.perf_counter() - t0)
            return shared["plugin_q"]

        for pd in policies:
            rec = base_rec(pd)
            try:
                if pd.kind in ("rtt", "tr", "posterior_gate"):
                    served, res, cert, timing = _api.serve(sys_, self.checker, ctx, pd.spec)
                    lp = logp_from_logits(served)
                    m = _served_metrics(ctx, logp_r, lp, y, cfg.zeta)
                    rec.update({k: v for k, v in m.items() if not k.startswith("_")})
                    rec.update({"mode": cert.mode, "first_fail": cert.first_fail, "first_fail_r2": cert.first_fail_r2,
                                "cert_damage": cert.damage_bound, "cert_charge_total": cert.charge_total,
                                "cert_terminal_allow": cert.terminal_allowance, "cert_row_max": cert.row_bound_max,
                                "cert_descent_max": cert.descent_max, "l0": res.l0,
                                "n_steps": len([l for l, r in res.records.items() if r.t is not None and np.any(np.asarray(r.t) > 0)]),
                                "B_left": res.B_left, "latency_s": timing["total_s"],
                                "exec_s": timing["exec_s"], "certify_s": timing.get("certify_s", 0.0),
                                "fallback_s": timing.get("fallback_s", 0.0),
                                "H_plus": pd.spec.H_plus, "H_row": pd.spec.H_row})
                    if pd.kind == "posterior_gate":
                        # This offline diagnostic needs the full reference for
                        # its cross-fitted posterior; base RTT does not.
                        extra_reference = max(0.0, t_inc - timing["prefix_s"])
                        rec["posterior_dependency_s"] = t_crossfit + extra_reference
                        rec["latency_s"] += rec["posterior_dependency_s"]
                    released = cert.mode in ("R1", "R2", "NO_STEP")
                    if released and cert.mode != "NO_STEP":
                        rec["violation_damage"] = bool(m["exact_S_plus"] > cert.damage_bound * (1 + 1e-12) + 1e-15)
                        rb = cert.row_bounds
                        rec["violation_row"] = bool((m["_dinf_rows"] > rb * (1 + 1e-12) + 1e-15).any().item()) if rb is not None else False
                        rec["exact_to_cert_ratio"] = (cert.damage_bound / m["exact_S_plus"]) if m["exact_S_plus"] > 0 else None
                    rec["depth_charges"] = {str(k): v for k, v in cert.charges.items() if cert.charge_kind.get(k) != "residual"}
                    rec["depth_kinds"] = {str(k): cert.charge_kind.get(k) for k in rec["depth_charges"]}
                    rec["depth_prov_charges"] = {str(l): r.charge_prov for l, r in res.records.items() if r.charge_prov > 0}
                    rec["factor_mean"] = {str(k): v for k, v in cert.factor_mean.items()} if cert.factor_mean else {}
                    if pd.params.get("keep_for_groups"):
                        shared[f"rtt:{pd.name}"] = (res, cert, lp, m)
                elif pd.kind == "rtt_plus_corrector":
                    base_name = pd.params["base"]
                    if f"rtt:{base_name}" not in shared:
                        raise _api.RTTFailure("ORCHESTRATION", f"{pd.name} requires {base_name} evaluated first with keep_for_groups")
                    res, cert, lp_base, m0 = shared[f"rtt:{base_name}"]
                    corr = self.models["correctors"][pd.params.get("corrector", "corr_2hop_rows")]
                    if corr.use_rows:
                        ctx.cache.pop("proposal_rows_feat", None)
                    _api.cuda_sync()
                    t0 = time.perf_counter()
                    if cert.mode in ("R1", "R2"):
                        H_T = res.states[-1]
                        z_base = res.logits_T[q]
                        row_allow = torch.clamp(pd.spec.H_row - cert.row_bounds, min=0.0) if math.isfinite(pd.spec.H_row) \
                            else torch.full((q.shape[0],), float("inf"), dtype=torch.float64, device=H_T.device)
                        call_allow = max(pd.spec.H_plus - cert.damage_bound, 0.0)
                    else:
                        H_T = ctx.inc_states[-1]
                        z_base = ctx.inc_logits[q]
                        row_allow = torch.full((q.shape[0],), pd.spec.H_row, dtype=torch.float64, device=H_T.device)
                        call_allow = pd.spec.H_plus
                    ps, h, _ = _api.corrector_serve(sys_, corr, ctx, call_allow, pd.spec.H_row, base_logits_q=z_base,
                                               base_H_T=H_T, row_allow=row_allow, call_allow=call_allow)
                    _api.cuda_sync()
                    t_c = time.perf_counter() - t0
                    lp = logp_from_probs(ps)
                    m = _served_metrics(ctx, logp_r, lp, y, cfg.zeta)
                    rec.update({k: v for k, v in m.items() if not k.startswith("_")})
                    # This comparator's row features are defined on the
                    # original reference trajectory. If RTT was accepted, its
                    # otherwise unnecessary reference tail must be charged.
                    # On fallback that tail was already run by the selector.
                    t_reference_extra = 0.0
                    if corr.use_rows and cert.mode in ("R1", "R2"):
                        t_reference_extra = max(0.0, t_inc - ctx.cache["prefix_time"][res.start_depth])
                    rec["corrector_s"] = t_c
                    rec["composed_reference_dependency_s"] = t_reference_extra
                    rec["latency_s"] = shared.get(f"lat:{base_name}", 0.0) + t_c + t_reference_extra
                    rec["mode"] = cert.mode
                elif pd.kind == "corrector":
                    corr = self.models["correctors"][pd.params["corrector"]]
                    if corr.use_rows:
                        # Each comparator pays for fresh proposal-row features;
                        # its position in the evaluation loop cannot make them free.
                        ctx.cache.pop("proposal_rows_feat", None)
                    _api.cuda_sync()
                    t0 = time.perf_counter()
                    if corr.use_rows:
                        _api.proposal_rows_features(sys_, ctx)
                    ps, h, _ = _api.corrector_serve(sys_, corr, ctx, pd.params["H_plus"], pd.params["H_row"])
                    _api.cuda_sync()
                    t_c = time.perf_counter() - t0
                    lp = logp_from_probs(ps)
                    m = _served_metrics(ctx, logp_r, lp, y, cfg.zeta)
                    rec.update({k: v for k, v in m.items() if not k.startswith("_")})
                    t_aux = t_features + (t_proposal if corr.use_rows else 0.0)
                    rec.update({"latency_s": t_inc + t_aux + t_c,
                                "corrector_s": t_c, "prepare_dependency_s": t_aux,
                                "H_plus": pd.params["H_plus"], "H_row": pd.params["H_row"],
                                "mode": "PROJECTED"})
                    rec["violation_damage"] = bool(m["exact_S_plus"] > pd.params["H_plus"] * (1 + 1e-9) + 1e-12)
                    if pd.params.get("keep_for_groups"):
                        shared[f"corr:{pd.name}"] = (lp, m)
                elif pd.kind in ("mixing", "projection", "switch", "raw"):
                    res_u, t_raw = raw_update()
                    requires_plugin = pd.kind == "switch" or (
                        pd.kind in ("mixing", "projection") and pd.params.get("allocation", "allocated") != "uniform")
                    qd, t_plugin = plugin_q() if requires_plugin else (None, 0.0)
                    # Exclude the possibly cached plug-in call from this timer;
                    # charge its recorded complete cost explicitly every time.
                    _api.cuda_sync()
                    t_post_start = time.perf_counter()
                    lpu = logp_from_logits(res_u.logits_T[q])
                    if pd.kind == "raw":
                        lp = lpu
                    else:
                        pr = torch.exp(logp_r)
                        pu = torch.exp(lpu)
                        H_plus = pd.params.get("H_plus", cfg.H_plus)
                        H_row = pd.params.get("H_row", cfg.H_row)
                        if pd.kind == "switch":
                            s_i = (qd * (lpu - logp_r)).sum(1)
                            thr = float(pd.params.get("threshold", float("inf")))
                            use = s_i > thr
                            lp = torch.where(use.unsqueeze(1), lpu, logp_r)
                        else:
                            n_q = q.shape[0]
                            if pd.params.get("allocation", "allocated") == "uniform":
                                h = torch.full((n_q,), min(H_plus, H_row), dtype=torch.float64, device=pr.device)
                            else:
                                hgrid = _api.make_hgrid(H_row, H_plus, cfg.frontier_grid)
                                G = _api.mixing_curves(qd, pr, pu, hgrid) if pd.kind == "mixing" else _api.projection_curves(qd, pr, pu, hgrid)
                                h, _ = _api.water_fill(G, hgrid, ctx.w64[q], H_plus,
                                                  hmax=torch.full((n_q,), H_row, dtype=torch.float64, device=pr.device))
                            if pd.kind == "mixing":
                                e = _api.mixing_eps_max(pr, pu, h)
                                ps = (1.0 - e).unsqueeze(1) * pr + e.unsqueeze(1) * pu
                            else:
                                ps = _api.floor_project(pu, pr, h)
                            lp = logp_from_probs(ps)
                    _api.cuda_sync()
                    t_post = time.perf_counter() - t_post_start
                    m = _served_metrics(ctx, logp_r, lp, y, cfg.zeta)
                    rec.update({k: v for k, v in m.items() if not k.startswith("_")})
                    # The unprotected raw update does not need an independent
                    # reference rollout to serve its own endpoint. Output rules
                    # do require both endpoints and their actual allocation work.
                    rec["raw_update_s"] = t_raw
                    rec["plugin_s"] = t_plugin
                    rec["output_rule_s"] = t_post
                    rec["latency_s"] = (0.0 if pd.kind == "raw" else t_inc) + t_raw + t_plugin + t_post
                    rec["mode"] = pd.kind.upper()
                    if pd.kind == "raw" and pd.params.get("keep_for_groups"):
                        shared["raw_lp"] = lpu
                elif pd.kind == "ft":
                    ft = self.models["ft"][pd.params["variant"]]
                    _api.cuda_sync()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        z = ft.deployed_logits(ctx.x, ctx.op, ctx=ctx)[q]
                    _api.cuda_sync()
                    t_ft = time.perf_counter() - t0
                    lp = logp_from_logits(z)
                    m = _served_metrics(ctx, logp_r, lp, y, cfg.zeta)
                    rec.update({k: v for k, v in m.items() if not k.startswith("_")})
                    rec["latency_s"] = t_ft
                    rec["mode"] = "FT"
                elif pd.kind == "oracle":
                    from ..frontier import continuous_floor_frontier
                    post = ctx.cache.get("crossfit_posterior")
                    if post is None:
                        raise _api.RTTFailure("ARTIFACT_MISSING", "oracle frontier needs the cross-fitted posterior")
                    pr = torch.exp(logp_r)
                    H_plus = pd.params.get("H_plus", cfg.H_plus)
                    H_row = pd.params.get("H_row", cfg.H_row)
                    h, ps, gain_plugin, frontier_meta = continuous_floor_frontier(
                        post, pr, ctx.w64[q], H_plus, H_row)
                    if not frontier_meta["converged"]:
                        raise _api.RTTFailure("SCIENTIFIC", "continuous posterior frontier did not meet numerical tolerances",
                                         details=frontier_meta)
                    lp = logp_from_probs(ps)
                    m = _served_metrics(ctx, logp_r, lp, y, cfg.zeta)
                    rec.update({k: v for k, v in m.items() if not k.startswith("_")})
                    rec["gain_plugin"] = gain_plugin
                    rec["frontier_metadata"] = frontier_meta
                    rec["H_plus"], rec["H_row"] = H_plus, H_row
                    rec["mode"] = "CONTINUOUS_POSTERIOR_FRONTIER_DIAGNOSTIC"
                else:
                    raise _api.RTTFailure("ORCHESTRATION", f"unknown policy kind {pd.kind}")
                rec["latency_ratio"] = rec.get("latency_s", float("nan")) / max(t_inc, 1e-12)
                if "latency_s" in rec:
                    rec["latency_with_build_s"] = rec["latency_s"] + rec["graph_build_s"]
                rec["flops_ratio"] = self.flops.get(pd.name)
                if pd.kind in ("rtt", "tr", "posterior_gate"):
                    shared[f"lat:{pd.name}"] = rec["latency_s"]
                rec["status"] = "OK"
            except _api.RTTFailure:
                raise
            except Exception as e:
                if any(p in str(e) for p in _api.INFRA_PATTERNS):
                    raise _api.RTTFailure("INFRASTRUCTURE", str(e), retriable=True)
                rec["status"] = "ERROR"
                rec["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            out.append(rec)
        if "groups" in extras:
            out.extend(self._group_records(ctx, spec, logp_r, shared))
        del ctx
        return out

    def _group_records(self, ctx, spec, logp_r, shared) -> List[dict]:
        """Figure 3(d): contributions from near-cancellation, beyond-two-hop and
        remaining rows (paired per-row gains summed with call weights)."""
        recs = []
        if "raw_lp" not in shared:
            return recs
        q = ctx.q_idx
        lpu = shared["raw_lp"]
        pr = torch.exp(logp_r)
        pu = torch.exp(lpu)
        near = (pu - pr).abs().sum(1) < 1e-3
        res_u = shared["raw"][0]
        beyond = raw_update_beyond_two_hop(self.sys, ctx, res_u)
        beyond = beyond & ~near
        rest = ~(near | beyond)
        y = ctx.y_q
        ce_r = -logp_r.gather(1, y.view(-1, 1)).squeeze(1)
        nq = float(q.shape[0])
        for key, val in shared.items():
            if key.startswith("rtt:"):
                lp = val[2]
            elif key.startswith("corr:"):
                lp = val[0]
            else:
                continue
            ce_s = -lp.gather(1, y.view(-1, 1)).squeeze(1)
            g = ce_r - ce_s
            for gname, mask in (("near_cancellation", near), ("beyond_two_hop", beyond), ("remaining", rest)):
                recs.append({"call_id": spec.call_id, "component": spec.component, "policy": key.split(":", 1)[1],
                             "kind": "group", "group": gname, "share_rows": float(mask.float().mean().item()),
                             "contribution": float(g[mask].sum().item()) / nq})
        return recs


def raw_update_beyond_two_hop(system: _api.RTTSystem, ctx: _api.CallContext, res_u: _api.ExecResult):
    """Scored rows for which more than half of the raw update's first-order influence
    on the terminal logits travels along paths with >= 3 neighbour hops
    (path-length grading; self-loop steps do not increase the grade)."""
    inc = system.inc
    cfg = system.cfg
    W = inc.weight()
    a, t = inc.alpha, inc.tau
    op = ctx.op
    selfv = (1.0 / op.deg.to(torch.float32)).unsqueeze(1)
    grades = [torch.zeros(ctx.n, system.d, device=W.device) for _ in range(4)]
    derivs = []
    for j in range(cfg.T):
        Hj = ctx.inc_states[j]
        Z = op.mm(Hj) @ W
        derivs.append(inc.act_deriv(Z))
    for j in range(cfg.open_first, cfg.T):
        # propagate existing grades one step, then inject the step at depth j+1 with grade 0
        new = [torch.zeros_like(grades[0]) for _ in range(4)]
        for gidx in range(4):
            X = grades[gidx]
            if not bool((X != 0).any().item()):
                continue
            PX = op.mm(X)
            selfpart = selfv * X
            nbrpart = PX - selfpart
            keep_self = (1.0 - a) * ((1.0 - t) * X + t * ((selfpart @ W) * derivs[j]))
            move = (1.0 - a) * t * ((nbrpart @ W) * derivs[j])
            new[gidx] = new[gidx] + keep_self
            new[min(3, gidx + 1)] = new[min(3, gidx + 1)] + move
        keep = ctx.keep(j + 1)
        if keep is not None:
            new = [X * keep.unsqueeze(1) for X in new]
        v = res_u.steps.get(j)
        if v is not None:
            new[0] = new[0] + v
        grades = new
    A = system.A32
    mags = [(X[ctx.q_idx] @ A).abs().sum(1) for X in grades]
    tot = sum(mags) + 1e-30
    return mags[3] > 0.5 * tot


def write_records(path: Path, records: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path) + ".tmp", "wt") as f:
        for r in records:
            f.write(json.dumps(r, default=_api.json_default) + "\n")
    os.replace(str(path) + ".tmp", path)


def read_records(path: Path) -> List[dict]:
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

