"""Measured reduction tables, figures, and claim ledgers."""
from __future__ import annotations

from ._imports import (
    Any,
    Dict,
    List,
    Path,
    Tuple,
    math,
    np,
)
from . import api as _api

# ----------------------------------------------------------------------------
# Paper-reported reference values, used only for side-by-side comparison
# ----------------------------------------------------------------------------
PAPER_TARGETS = {
    "table2": {
        "rtt_registered": {"gain": 6.5, "ci": [6.3, 6.7], "N_zeta": 12, "u": 0.0095, "u_node": 0.0051, "lb": 6.2, "lat": 1.42, "flops": 1.36},
        "rtt_plus_corrector": {"gain": 6.7, "ci": [6.5, 7.0], "N_zeta": 12, "u": 0.0095, "u_node": 0.0051, "lb": 6.4, "lat": 1.49, "flops": 1.40},
        "corr_2hop_rows": {"gain": 5.6, "ci": [5.4, 5.8], "N_zeta": 14, "u": 0.0107, "u_node": 0.0056, "lb": 5.3, "lat": 1.25, "flops": 1.22},
        "corr_2hop": {"gain": 5.1, "ci": [4.8, 5.3], "N_zeta": 14, "u": 0.0107, "lb": 4.8, "lat": 1.07},
        "corr_per_row": {"gain": 4.5, "ci": [4.3, 4.7], "N_zeta": 13, "u": 0.0101, "lb": 4.2, "lat": 1.02},
        "corr_4hop_rows": {"gain": 5.3, "lat": 1.32},
        "corr_8hop_rows": {"gain": 5.0, "lat": 1.46},
        "mixing_allocated": {"gain": 5.1, "N_zeta": 15, "u": 0.0113, "lb": 4.8, "lat": 2.18},
        "mixing_uniform": {"gain": 4.1, "lat": 2.18},
        "projection_allocated": {"gain": 5.2, "N_zeta": 15, "u": 0.0113, "lb": 5.0, "lat": 2.18},
        "projection_uniform": {"gain": 4.3, "lat": 2.18},
        "switch_ltt": {"gain": 4.5, "N_zeta": 17, "u": 0.0124, "lat": 2.18},
        "tr_matched": {"gain": 5.8, "ci": [5.6, 6.0], "N_zeta": 18, "u": 0.0130, "lb": 5.5, "lat": 1.41},
        "tr_e2e": {"gain": 5.5, "N_zeta": 19, "u": 0.0136, "lb": 5.2},
        "ft_same_labels": {"gain": 1.0, "N_zeta": 45, "u": 0.0281},
        "ft_kl": {"gain": 2.8, "N_zeta": 29, "u": 0.0193},
        "ft_dinf": {"gain": 3.4, "N_zeta": 27, "u": 0.0181},
        "ft_pct": {"gain": 2.9, "N_zeta": 30, "u": 0.0198},
        "ft_regaware": {"gain": 3.1, "N_zeta": 30, "u": 0.0198},
        "ft_corruptions": {"gain": 3.4, "N_zeta": 62, "u": 0.0373},
        "raw": {"gain": 3.9, "ci": [3.6, 4.3], "N_zeta": 159, "u": 0.0881, "lat": 2.18},
        "rtt_tolerance_point": {"gain": 2.0, "ci": [2.0, 2.1], "N_zeta": 0, "u": 0.0, "lb": 1.9},
        "variant_per_depth_radii": {"gain": 6.9, "N_zeta": 13, "u": 0.0101},
        "swap_no_row": {"gain": 6.7},
        "swap_spend_caps": {"gain": 6.2},
        "swap_env_only": {"gain": 6.4},
        "swap_mean_features": {"gain": 6.4},
        "swap_probe_slopes": {"gain": 6.2},
        "swap_sym_005": {"gain": 3.9},
        "swap_sym_030": {"gain": 5.5},
        "swap_global_factors": {"gain": 4.1},
        "swap_late8": {"gain": 4.8, "lat": 1.24},
        "posterior_slope_gate": {"gain": 7.9},
        "oracle_frontier": {"gain": 10.6},
    },
    "headline": {"primary_difference": 0.9, "frontier_share": 0.61, "incumbent_test_accuracy": 0.759,
                 "ledger_R1": 2035, "ledger_R2": 6, "ledger_F": 7, "transfer_u": 0.0118, "transfer_lb": 1.7},
}


def _gain_milli(x):
    return None if x is None or not isinstance(x, (int, float)) or not math.isfinite(x) else 1e3 * x


def audit_generator_component(record: dict) -> str:
    """Keep a mixture draw separate from an independent component-only audit."""
    if record.get("generator_component"):
        return str(record["generator_component"])
    fields = str(record.get("call_id", "")).rsplit(":", 2)
    return fields[-2] if len(fields) == 3 else "unknown"


def reduce_suite(suite: _api.Suite, cfg: _api.RTTConfig, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tdir = out_dir / "tables"
    fdir = out_dir / "figures"
    tdir.mkdir(exist_ok=True)
    fdir.mkdir(exist_ok=True)
    summary: Dict[str, Any] = {"version": _api.VERSION, "version_tag": _api.VERSION_TAG, "utc": _api.now_utc(),
                               "certificate_status": _api.CERTIFICATE_STATUS,
                               "scale": cfg.scale_name, "implementation_choices": _api.IMPLEMENTATION_CHOICES,
                               "not_implemented": {"proposals": _api.NOT_IMPLEMENTED_PROPOSALS,
                                                   "datasets": ["GOOD-Arxiv", "Twitch", "Facebook-100", "Elliptic",
                                                                "ogbn-products", "ogbg-molpcba", "ImageNet-C", "Folktables"],
                                                   "incumbent_families": ["GCNII", "GRAND", "GraphSAGE-2", "GCN-3"],
                                                   "experiments": ["monitoring streams (Fig. 8)", "TTA collapse streams (Table 14b)",
                                                                   "generic verifier comparison (Fig. 10c)"]}}
    # ---- incumbent ----
    inc_sum = {}
    for rd in sorted((suite.stage_dir("incumbent") / "runs").glob("*/RESULT")) if (suite.stage_dir("incumbent") / "runs").is_dir() else []:
        st, s = _api.task_status(rd)
        if st == "COMPLETE":
            inc_sum[s.get("incumbent", rd.parent.name)] = {k: s.get(k) for k in ("accuracy", "temperature", "w_norm_exact",
                                                                                  "gamma_delta", "global_factor", "flops_per_pass")}
    summary["incumbent"] = inc_sum
    sel = suite.load("selection.json", kind="json") if suite.has("selection.json") else {"status": "NOT_RUN"}
    summary["selection"] = sel
    # ---- final audit ----
    audit = _api.stage_records(suite, "audit")
    pol_recs: Dict[Tuple[str, str], List[dict]] = {}
    groups = []
    for r in audit:
        if r.get("kind") == "group":
            groups.append(r)
            continue
        key = audit_generator_component(r) if r["stage"] == "audit" else r["stage"]
        pol_recs.setdefault((key, r["policy"]), []).append(r)
    mixture = {}
    mix_calls = [r for r in audit if r.get("stage") == "audit" and r.get("kind") != "group"]
    # a call belongs to the mixture audit if its call_id says 'mixture'
    for r in mix_calls:
        if audit_generator_component(r) == "mixture":
            mixture.setdefault(r["policy"], []).append(r)
    audit_issues = []
    for generator, expected in [("mixture", cfg.audit_N_mixture)] + [(c, cfg.audit_N_component) for c in _api.ALL_COMPONENTS]:
        rr = pol_recs.get((generator, "rtt_registered"), [])
        complete = [r for r in rr if r.get("status") == "OK"]
        unique = {r["call_id"] for r in complete}
        if len(complete) != expected or len(unique) != expected or len(rr) != len(complete):
            audit_issues.append({"generator": generator, "expected": expected, "completed": len(complete),
                                 "unique_completed": len(unique), "failed": len(rr) - len(complete)})
    summary["audit_completeness"] = {"status": "INCOMPLETE" if audit_issues else "COMPLETE",
                                     "issues": audit_issues}
    node = None
    nt_dir = suite.stage_dir("node_transfer") / "runs"
    if nt_dir.is_dir():
        for p in nt_dir.glob("*_per_node/RESULT/per_node.json"):
            node = _api.read_json(p)
    summary["per_node"] = node
    t2 = []
    ref = {r["call_id"]: r for r in mixture.get("rtt_registered", []) if r.get("status") == "OK"}
    pvals = {}
    certified_kinds = {"rtt", "tr", "posterior_gate", "corrector", "mixing", "projection", "rtt_plus_corrector"}
    for pol, rr in sorted(mixture.items()):
        kind = rr[0].get("kind")
        hp = rr[0].get("H_plus")
        if hp is None:
            hp = cfg.H_plus if kind in ("mixing", "projection") else None
        s = _api.summarize_policy(rr, cfg, cfg.delta, H_plus=hp, certified=kind in certified_kinds)
        row = {"policy": pol, "kind": kind, "n": s.get("n"), "status": s.get("status"),
               "n_attempted": s.get("n_attempted"), "n_failed": s.get("n_failed"),
               "certificate_status": _api.CERTIFICATE_STATUS, "gain_milli": _gain_milli(s.get("mean_gain")),
               "ci_lo_milli": _gain_milli(s.get("ci_lo")), "ci_hi_milli": _gain_milli(s.get("ci_hi")),
               "N_zeta": s.get("N_zeta"), "u": s.get("u"), "lb_milli": _gain_milli(s.get("betting_lb")),
               "latency_ratio": s.get("latency_ratio"), "acc_change_pp": s.get("acc_change_pp"),
               "neg_flip_frac": s.get("neg_flip_frac"), "violations_damage": s.get("violations_damage"),
               "modes": s.get("modes")}
        if node and pol in node:
            row["u_node"] = node[pol]["u_node"]
        if (pol != "rtt_registered" and len(ref) == cfg.audit_N_mixture
                and s.get("status") == "COMPLETE" and len(rr) == cfg.audit_N_mixture):
            diffs = []
            for r in rr:
                if r.get("status") == "OK" and r["call_id"] in ref:
                    diffs.append(ref[r["call_id"]]["gain"] - r["gain"])
            if diffs:
                m, lo, hi = _api.paired_t_interval(np.array(diffs))
                row.update({"delta_to_rtt_milli": _gain_milli(m), "delta_ci_lo_milli": _gain_milli(lo),
                            "delta_ci_hi_milli": _gain_milli(hi)})
                pvals[pol] = _api.one_sided_p_positive(np.array(diffs))
        t2.append(row)
    secondaries = ["corr_2hop", "corr_per_row", "mixing_allocated", "projection_allocated", "switch_ltt",
                   "tr_matched", "ft_kl", "raw"]
    holm = _api.holm_adjust({k: pvals[k] for k in secondaries if k in pvals})
    for row in t2:
        if row["policy"] in holm:
            row["holm_p"] = holm[row["policy"]]
        if row["policy"] == "corr_2hop_rows":
            row["primary_p_one_sided"] = pvals.get("corr_2hop_rows")
            row["primary_pass"] = bool(pvals.get("corr_2hop_rows", 1.0) <= cfg.delta)
    _api.write_csv(tdir / "table12_complete_mixture.csv", t2)
    summary["table12"] = t2
    # ---- release ledger and the ten statements ----
    rtt_mix = mixture.get("rtt_registered", [])
    ledger = _api.release_ledger(rtt_mix) if rtt_mix else {"status": "NOT_RUN"}
    statements = []
    complete_mix = bool(rtt_mix) and all(r.get("status") == "OK" for r in rtt_mix)
    if complete_mix:
        ok = [r for r in rtt_mix if r.get("status") == "OK"]
        n = len(ok)
        statements.append({"statement": "mixture event d > zeta", **_api.rate_statement(sum(r["down"] > cfg.zeta for r in ok), n, cfg.delta)})
        statements.append({"statement": "call-accuracy drop > 0.1 point", **_api.rate_statement(sum(r["acc_change_pp"] < -0.1 for r in ok), n, cfg.delta)})
        statements.append({"statement": "negative flips on > 1% of the call", **_api.rate_statement(sum(r["neg_flip_frac"] > 0.01 for r in ok), n, cfg.delta)})
        for comp in _api.ALL_COMPONENTS:
            attempted = pol_recs.get((comp, "rtt_registered"), [])
            rr = [r for r in attempted if r.get("status") == "OK"]
            complete_component = len(rr) == len(attempted) == cfg.audit_N_component
            st = (_api.rate_statement(sum(r["down"] > cfg.zeta for r in rr), len(rr), cfg.delta)
                  if complete_component else {"events": None, "n": len(rr), "u_delta": None,
                                             "n_failed": len(attempted) - len(rr), "status": "INCOMPLETE"})
            st["mean_gain_milli"] = _gain_milli(float(np.mean([r["gain"] for r in rr]))) if rr else None
            statements.append({"statement": f"component: {comp}" + (" (separate)" if comp == "stress" else ""), **st})
        d = np.array([r["down"] for r in ok])
        hp = ok[0]["H_plus"]
        lb = -_api.betting_ucb(d, hp, cfg.delta, "tuned")
        lb_sim = -_api.betting_ucb(d, hp, cfg.delta / 10.0, "tuned")
        need = cfg.min_certified_mean_tol if hp <= cfg.zeta * 1.0001 else cfg.min_certified_mean
        statements.append({"statement": "certified mean (betting LB)", "mean_gain_milli": _gain_milli(float(-d.mean())),
                           "lb_milli": _gain_milli(lb), "lb_simultaneous_milli": _gain_milli(lb_sim),
                           "required_milli": 1e3 * need, "pass": bool(lb >= need)})
        all_pass = True
        missing = False
        for s in statements[:-1]:
            if s.get("n", 0) == 0 or s.get("u_delta") is None:
                missing = True
                s["pass"] = None
                continue
            s["pass"] = bool(s["u_delta"] <= cfg.p_star)
            all_pass = all_pass and s["pass"]
        all_pass = all_pass and statements[-1]["pass"]
        decision = "NOT_EVALUATED_MISSING_EVIDENCE" if missing else ("RELEASED" if all_pass else "NOT_RELEASED")
    else:
        decision = "NOT_EVALUATED_INCOMPLETE" if rtt_mix else "NOT_EVALUATED"
    if audit_issues:
        decision = "NOT_EVALUATED_INCOMPLETE"
        for statement in statements:
            statement["pass"] = None
            statement["evidence_status"] = "INCOMPLETE_REGISTERED_AUDIT"
    elif sel.get("status") != "SELECTED":
        decision = "NOT_RELEASED_NO_SELECTED_MEMBER"
    _api.write_csv(tdir / "table9_statements.csv", statements)
    _api.write_json(tdir / "table9_ledger.json", ledger)
    summary["release"] = {"decision": decision, "ledger": ledger, "statements": statements,
                          "selection_status": sel.get("status"), "certificate_status": _api.CERTIFICATE_STATUS,
                          "interpretation": "Numerical research audit conditional on the stated arithmetic assumptions; not formal floating-point verification."}
    # ---- betting implementations (Table 10a) ----
    t10a = []
    if complete_mix and len(rtt_mix) == cfg.audit_N_mixture:
        ok = [r for r in rtt_mix if r.get("status") == "OK"]
        d = np.array([r["down"] for r in ok])
        hp = ok[0]["H_plus"]
        mean_gain = -float(d.mean())
        for mode, label in (("tuned", "registered bets tuned to N"), ("horizon_free", "horizon-free sequence"),
                            ("kelly", "approximate Kelly"), ("hedged", "hedged capital (delta/2)")):
            ub = _api.betting_ucb(d, hp, cfg.delta, mode)
            t10a.append({"implementation": label, "certified_mean_milli": _gain_milli(-ub),
                         "half_width_milli": _gain_milli(mean_gain + ub)})
        nu = _api.normal_ucb(d, cfg.delta)
        t10a.append({"implementation": "normal interval (not finite-sample valid)", "certified_mean_milli": _gain_milli(-nu),
                     "half_width_milli": _gain_milli(mean_gain + nu)})
        t10a.append({"implementation": "Gaussian first-order width", "half_width_milli":
                     _gain_milli(math.sqrt(2 * float(d.var(ddof=1)) * math.log(1 / cfg.delta) / len(d)))})
        _, path = _api.betting_ucb(d, hp, cfg.delta, "tuned", return_path=True)
        _api.write_csv(tdir / "fig5b_running_bound.csv", [{"t": t, "lb_milli": _gain_milli(-u)} for t, u in path])
        kl = _api.kl_robust_curve(d, hp, cfg.delta, [0.0, 0.001, 0.002, 0.004, 0.008, 0.016])
        _api.write_csv(tdir / "table10c_kl_curve.csv", [{"rho": k["rho"], "certified_mean_milli": _gain_milli(k["certified_mean_gain"])} for k in kl])
        summary["kl_curve"] = kl
    _api.write_csv(tdir / "table10a_betting.csv", t10a)
    summary["table10a"] = t10a
    # ---- transfer (Table 10b) ----
    trans = _api.stage_records(suite, "node_transfer")
    t10b = []
    byp = {}
    for r in trans:
        byp.setdefault(r["policy"], []).append(r)
    for pol, rr in byp.items():
        kind = rr[0].get("kind")
        s = _api.summarize_policy(rr, cfg, cfg.delta, H_plus=rr[0].get("H_plus"), certified=kind in certified_kinds)
        t10b.append({"policy": pol, "events": s.get("N_zeta"), "n": s.get("n"), "u": s.get("u"),
                     "gain_milli": _gain_milli(s.get("mean_gain")), "lb_milli": _gain_milli(s.get("betting_lb")),
                     "rule_u_le_pstar": (s.get("u") is not None and s.get("u") <= cfg.p_star)})
    _api.write_csv(tdir / "table10b_transfer2020.csv", t10b)
    summary["table10b"] = t10b
    # ---- components and frontier (Table 11a) ----
    t11 = []
    fl_dir = suite.stage_dir("diagnostics") / "runs"
    fl = None
    if fl_dir.is_dir():
        for p in fl_dir.glob("*_frontier_levels/RESULT/frontier_levels.json"):
            fl = _api.read_json(p)
    for comp in list(_api.ALL_COMPONENTS) + ["mixture"]:
        rr = mixture.get("rtt_registered", []) if comp == "mixture" else pol_recs.get((comp, "rtt_registered"), [])
        cc = mixture.get("corr_2hop_rows", []) if comp == "mixture" else pol_recs.get((comp, "corr_2hop_rows"), [])
        oo = mixture.get("oracle_frontier", []) if comp == "mixture" else pol_recs.get((comp, "oracle_frontier"), [])
        row = {"population": comp,
               "U_RTT_realized_milli": _gain_milli(float(np.mean([r["gain"] for r in rr if r.get("status") == "OK"]))) if rr else None,
               "corrector_realized_milli": _gain_milli(float(np.mean([r["gain"] for r in cc if r.get("status") == "OK"]))) if cc else None,
               "U_out_hat_plugin_milli": _gain_milli(float(np.mean([r["gain_plugin"] for r in oo if r.get("status") == "OK"]))) if oo else None}
        if row["U_RTT_realized_milli"] is not None and row["U_out_hat_plugin_milli"]:
            row["share_realized_over_plugin"] = row["U_RTT_realized_milli"] / row["U_out_hat_plugin_milli"]
        if fl and comp in fl.get("per_population", {}):
            pp = fl["per_population"][comp]
            row.update({"diag_U_out_hat_milli": _gain_milli(pp["U_out_hat"]), "diag_U_reach_tilde_milli": _gain_milli(pp["U_reach_tilde"]),
                        "diag_U_cert_tilde_milli": _gain_milli(pp["U_cert_tilde"]),
                        "diag_U_RTT_plugin_milli": _gain_milli(pp["U_RTT_hat_plugin"])})
        t11.append(row)
    _api.write_csv(tdir / "table11a_frontier.csv", t11)
    summary["table11a"] = t11
    # ---- planted partition (Table 11b) ----
    t11b = []
    if fl_dir.is_dir():
        for p in sorted(fl_dir.glob("*_planted_g*/RESULT/*.json")):
            if p.name.startswith("planted_g"):
                o = _api.read_json(p)
                t11b.append({"gamma": o["gamma"], **{k: v for k, v in o["levels"].items()}, "note": o["decomposition_terms"]})
    _api.write_csv(tdir / "table11b_planted.csv", t11b)
    summary["table11b"] = t11b
    # ---- swaps (Table 13) ----
    reg_gain = next((r["gain_milli"] for r in t2 if r["policy"] == "rtt_registered"), None)
    reg_u = next((r["u"] for r in t2 if r["policy"] == "rtt_registered"), None)
    t13 = []
    for sw, (desc, _) in list(_api.SWAPS.items()) + [("variant_per_depth_radii", ("Per-depth radii", None))]:
        row = next((r for r in t2 if r["policy"] == sw), None)
        t13.append({"swap": sw, "description": desc, "alternative_gain_milli": row["gain_milli"] if row else None,
                    "alternative_u": row["u"] if row else None, "registered_gain_milli": reg_gain, "registered_u": reg_u,
                    "status": "MEASURED" if row else "NOT_RUN"})
    _api.write_csv(tdir / "table13_swaps.csv", t13)
    summary["table13"] = t13
    # ---- row groups (Figure 3d) ----
    grp = {}
    for r in groups:
        if ":mixture:" not in r["call_id"]:
            continue
        key = (r["policy"], r["group"])
        grp.setdefault(key, {"contrib": [], "share": []})
        grp[key]["contrib"].append(r["contribution"])
        grp[key]["share"].append(r["share_rows"])
    t3d = [{"policy": k[0], "group": k[1], "contribution_milli": _gain_milli(float(np.mean(v["contrib"]))),
            "row_share": float(np.mean(v["share"]))} for k, v in sorted(grp.items())]
    _api.write_csv(tdir / "fig3d_row_groups.csv", t3d)
    summary["row_groups"] = t3d
    # ---- breadth (Table 3) ----
    br = _api.stage_records(suite, "breadth")
    t3 = []
    for prop in _api.EXTERNAL_PROPOSALS:
        rr = [r for r in br if r["policy"].endswith("_" + prop)]
        if not rr:
            t3.append({"proposal": prop, "status": "NOT_RUN"})
            continue
        stats = {}
        for pol in ("raw", "tr", "corr", "rtt"):
            sel_ = [r for r in rr if r["policy"] == f"{pol}_{prop}"]
            kind = sel_[0].get("kind") if sel_ else None
            stats[pol] = _api.summarize_policy(sel_, cfg, cfg.delta, H_plus=sel_[0].get("H_plus") if sel_ else None,
                                          certified=kind in certified_kinds) if sel_ else {}
        rt = {r["call_id"]: r["gain"] for r in rr if r["policy"] == f"rtt_{prop}" and r.get("status") == "OK"}
        diffs = [rt[r["call_id"]] - r["gain"] for r in rr if r["policy"] == f"corr_{prop}" and r["call_id"] in rt]
        m, lo, hi = _api.paired_t_interval(np.array(diffs)) if diffs else (None, None, None)
        t3.append({"proposal": prop, "raw_milli": _gain_milli(stats["raw"].get("mean_gain")), "raw_u": stats["raw"].get("u"),
                   "tr_milli": _gain_milli(stats["tr"].get("mean_gain")), "corrector_milli": _gain_milli(stats["corr"].get("mean_gain")),
                   "rtt_milli": _gain_milli(stats["rtt"].get("mean_gain")), "rtt_u": stats["rtt"].get("u"),
                   "rtt_minus_corr_milli": _gain_milli(m), "diff_ci_lo": _gain_milli(lo), "diff_ci_hi": _gain_milli(hi),
                   "lat_rtt": stats["rtt"].get("latency_ratio"), "lat_raw_update": stats["raw"].get("latency_ratio")})
    for p in _api.NOT_IMPLEMENTED_PROPOSALS:
        t3.append({"proposal": p, "status": "NOT_IMPLEMENTED"})
    _api.write_csv(tdir / "table3_breadth.csv", t3)
    summary["table3"] = t3
    # ---- development sweeps ----
    dev = _api.stage_records(suite, "development")
    dev_rows = {}
    for r in dev:
        dev_rows.setdefault(r["policy"], []).append(r)
    dsum = []
    for pol, rr in sorted(dev_rows.items()):
        kind = rr[0].get("kind")
        s = _api.summarize_policy(rr, cfg, cfg.delta, H_plus=rr[0].get("H_plus"), certified=kind in certified_kinds)
        fb = sum(1 for r in rr if r.get("mode") == "F") / max(1, len(rr))
        dsum.append({"policy": pol, "n": s.get("n"), "gain_milli": _gain_milli(s.get("mean_gain")), "u": s.get("u"),
                     "N_zeta": s.get("N_zeta"), "latency_ratio": s.get("latency_ratio"), "fallback_rate": fb,
                     "lb_milli": _gain_milli(s.get("betting_lb"))})
    _api.write_csv(tdir / "development_sweeps.csv", dsum)
    summary["development"] = dsum
    anchor = [abs(r["down"]) for r in dev_rows.get("rtt_H25z", []) if r.get("status") == "OK" and -0.15 <= r["acc_change_pp"] <= -0.05]
    summary["zeta_anchor"] = {"median_down_given_0.1pp_drop": float(np.median(anchor)) if anchor else None,
                              "n_calls": len(anchor), "registered_zeta": cfg.zeta}
    # ---- checker tests (Table 18) ----
    t18 = []
    if fl_dir.is_dir():
        for name in ("checker_planted", "checker_hp", "checker_adversarial", "checker_underfactor",
                     "checker_rounding_off", "invariance", "depth_exact"):
            ps = list(fl_dir.glob(f"*_{name}/RESULT/{name}.json"))
            t18.append({"test": name, "result": _api.read_json(ps[0]) if ps else "NOT_RUN"})
    rtt_ok = [r for r in rtt_mix if r.get("status") == "OK" and r.get("mode") in ("R1", "R2")]
    t18.append({"test": "exact endpoint cost (offline second rollout), final-audit mixture",
                "result": {"released_calls": len(rtt_ok), "violations": int(sum(bool(r.get("violation_damage")) for r in rtt_ok)),
                           "row_violations": int(sum(bool(r.get("violation_row")) for r in rtt_ok)),
                           "certified_to_exact_median": float(np.median([r["exact_to_cert_ratio"] for r in rtt_ok
                                                                         if r.get("exact_to_cert_ratio")])) if rtt_ok else None}})
    _api.write_json(tdir / "table18_checker.json", t18)
    summary["table18"] = t18
    # ---- cost ledger (Table 17a) ----
    t17 = []
    for row in t2:
        t17.append({"policy": row["policy"], "latency_ratio_mean": row.get("latency_ratio")})
    _api.write_csv(tdir / "table17a_latency.csv", t17)
    # ---- paper target comparison ----
    comp_rows = []
    for pol, tgt in PAPER_TARGETS["table2"].items():
        row = next((r for r in t2 if r["policy"] == pol), None)
        for k, v in tgt.items():
            meas = None
            if row is not None:
                meas = {"gain": row.get("gain_milli"), "ci": [row.get("ci_lo_milli"), row.get("ci_hi_milli")],
                        "N_zeta": row.get("N_zeta"), "u": row.get("u"), "u_node": row.get("u_node"),
                        "lb": row.get("lb_milli"), "lat": row.get("latency_ratio"), "flops": row.get("flops_ratio")}.get(k)
            comp_rows.append({"table": "2/12", "row": pol, "quantity": k, "manuscript_target": v,
                              "measured": meas, "status": "MEASURED" if meas is not None else ("NOT_RUN" if row is None else "NOT_MEASURED")})
    prim = next((r for r in t2 if r["policy"] == "corr_2hop_rows"), None)
    comp_rows.append({"table": "abstract", "row": "RTT minus strongest corrector", "quantity": "gain difference (1e-3 nats)",
                      "manuscript_target": PAPER_TARGETS["headline"]["primary_difference"],
                      "measured": prim.get("delta_to_rtt_milli") if prim else None,
                      "status": "MEASURED" if prim and prim.get("delta_to_rtt_milli") is not None else "NOT_RUN"})
    share = next((r.get("share_realized_over_plugin") for r in t11 if r["population"] == "mixture"), None)
    comp_rows.append({"table": "abstract", "row": "frontier share", "quantity": "RTT / U_out (cross-fitted posterior)",
                      "manuscript_target": PAPER_TARGETS["headline"]["frontier_share"], "measured": share,
                      "status": "MEASURED" if share is not None else "NOT_RUN"})
    acc = (inc_sum.get("main") or {}).get("accuracy") or {}
    comp_rows.append({"table": "4.1", "row": "incumbent", "quantity": "official test accuracy (clean)",
                      "manuscript_target": PAPER_TARGETS["headline"]["incumbent_test_accuracy"],
                      "measured": acc.get("acc_official_test"), "status": "MEASURED" if acc.get("acc_official_test") is not None else "NOT_RUN"})
    if isinstance(ledger, dict) and "N" in ledger:
        for k, lk in (("ledger_R1", "first_pass_release"), ("ledger_R2", "recertified_release"), ("ledger_F", "fallback")):
            comp_rows.append({"table": "9a", "row": "release ledger", "quantity": lk,
                              "manuscript_target": PAPER_TARGETS["headline"][k], "measured": ledger.get(lk), "status": "MEASURED"})
    _api.write_csv(out_dir / "paper_target_comparison.csv", comp_rows)
    _api.write_json(out_dir / "paper_target_comparison.json", {"note": "manuscript values are paper-reported reference results; only the "
                                                          "'measured' column comes from this run", "rows": comp_rows})
    # ---- claim ledger (fail-closed) ----
    claims = []

    def claim(cid, text, passed, evidence):
        claims.append({"claim": cid, "statement": text,
                       "status": ("NUMERICALLY_CHECKED" if cid == "G1-G3" else "PASS") if passed is True
                                 else ("FAIL" if passed is False else "NOT_EVALUATED"),
                       "certificate_status": _api.CERTIFICATE_STATUS,
                       "evidence": evidence})
    viol_damage = sum(int(bool(r.get("violation_damage"))) for r in rtt_ok)
    viol_row = sum(int(bool(r.get("violation_row"))) for r in rtt_ok)
    viol = viol_damage + viol_row if rtt_ok else None
    claim("G1-G3", "observed released calls numerically satisfy call and row bounds against the offline endpoint; formal floating-point enclosure remains conditional",
          (viol == 0) if viol is not None and not audit_issues else None,
          {"damage_violations": viol_damage, "row_violations": viol_row,
           "nonzero_step_releases_checked": len(rtt_ok), "audit_complete": not audit_issues})
    claim("G4", "call-level regression rate below p* on the mixture, components and stress",
          all(s.get("pass") for s in statements[:-1] if s.get("n")) if statements and not audit_issues else None,
          {"statements": len(statements)})
    claim("G5", "certified mean (betting LB) above the registered minimum",
          statements[-1]["pass"] if statements and not audit_issues else None, statements[-1] if statements else None)
    claim("PRIMARY", "RTT exceeds the strongest one-pass corrector (one-sided paired test at delta)",
          prim.get("primary_pass") if prim else None, {"p": prim.get("primary_p_one_sided") if prim else None,
                                                       "delta_milli": prim.get("delta_to_rtt_milli") if prim else None})
    claim("SECONDARY", "eight Holm-corrected secondary comparisons",
          all((v is not None and v <= cfg.delta) for v in holm.values()) if holm else None, holm)
    claim("SELECTION", "a member passed the fixed-sequence selection", (sel.get("status") == "SELECTED") if sel else None,
          {"selected": sel.get("selected"), "stopped_at": sel.get("stopped_at")})
    claim("RELEASE", "all ten statements pass (release decision)", {"RELEASED": True, "NOT_RELEASED": False}.get(decision),
          {"decision": decision})
    _api.write_json(out_dir / "claim_ledger.json", claims)
    summary["claims"] = claims
    make_figures(summary, t2, dsum, fdir, cfg)
    _api.write_json(out_dir / "summary.json", summary)
    _api.atomic_write_text(out_dir / "MEASURED_REPORT.md", render_report(summary, t2, cfg))
    return summary


def make_figures(summary: dict, t2: List[dict], dsum: List[dict], fdir: Path, cfg: _api.RTTConfig) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        _api.warn(f"matplotlib unavailable ({e}); figures skipped")
        return
    # budget sweep (Fig. 2c analogue)
    fam = {"rtt": [], "corr": [], "mix": [], "proj": [], "tr": [], "sym": []}
    for r in dsum:
        p = r["policy"]
        for f in fam:
            if p.startswith(f + "_H") and p.endswith("z"):
                try:
                    mult = float(p[len(f) + 2:-1])
                except ValueError:
                    continue
                if r["gain_milli"] is not None:
                    fam[f].append((mult * cfg.zeta, r["gain_milli"]))
    if any(fam.values()):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        labels = {"rtt": "RTT", "corr": "one-pass corrector", "mix": "adaptive mixing", "proj": "floor projection",
                  "tr": "matched TR", "sym": "symmetric pricing"}
        for f, pts in fam.items():
            if pts:
                pts.sort()
                ax.plot([a for a, _ in pts], [b for _, b in pts], marker="o", label=labels[f])
        ax.set_xscale("log")
        ax.set_xlabel("H+ (nats)")
        ax.set_ylabel("-E[d] (1e-3 nats)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fdir / "fig2c_budget_sweep.png", dpi=180)
        plt.close(fig)
    # gain vs bound and latency (Fig. 4a-b analogue)
    pts = [(r["u"], r["gain_milli"], r.get("latency_ratio"), r["policy"]) for r in t2
           if r.get("u") is not None and r.get("gain_milli") is not None]
    if pts:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
        for u, gm, lat, pol in pts:
            axs[0].scatter(100 * u, gm, s=12)
            axs[0].annotate(pol, (100 * u, gm), fontsize=5)
            if lat is not None and math.isfinite(lat):
                axs[1].scatter(lat, gm, s=12)
                axs[1].annotate(pol, (lat, gm), fontsize=5)
        axs[0].axvline(100 * cfg.p_star, ls="--", lw=0.8)
        axs[0].set_xlabel("u per call (%)")
        axs[0].set_ylabel("-E[d] (1e-3 nats)")
        axs[1].set_xlabel("latency / one incumbent pass")
        fig.tight_layout()
        fig.savefig(fdir / "fig4ab_tradeoffs.png", dpi=180)
        plt.close(fig)


def render_report(summary: dict, t2: List[dict], cfg: _api.RTTConfig) -> str:
    L = [f"# RTT {_api.VERSION_TAG} measured report", "",
         "All numbers below were measured by this run. Manuscript targets are listed separately in "
         "paper_target_comparison.csv.", "",
         f"Scale: {summary.get('scale')}", "",
         f"Selection: {summary.get('selection', {}).get('status')} (selected: {summary.get('selection', {}).get('selected')})",
         f"Release decision: {summary.get('release', {}).get('decision')}", "",
         "## Final-audit mixture (gain in 1e-3 nats)", "",
         "| policy | n | gain [95% CI] | delta to RTT | N_zeta | u | LB | latency |", "|---|---|---|---|---|---|---|---|"]
    for r in t2:
        L.append(f"| {r['policy']} | {r.get('n')} | {_api.fmt(r.get('gain_milli'), 3)} [{_api.fmt(r.get('ci_lo_milli'), 3)}, "
                 f"{_api.fmt(r.get('ci_hi_milli'), 3)}] | {_api.fmt(r.get('delta_to_rtt_milli'), 3)} | {r.get('N_zeta')} | "
                 f"{_api.fmt(r.get('u'), 3)} | {_api.fmt(r.get('lb_milli'), 3)} | {_api.fmt(r.get('latency_ratio'), 3)} |")
    L += ["", "## Claim ledger", ""]
    for c in summary.get("claims", []):
        L.append(f"- [{c['status']}] {c['claim']}: {c['statement']}")
    L += ["", "## Not implemented in this recreation", ""]
    ni = summary.get("not_implemented", {})
    for k, v in ni.items():
        L.append(f"- {k}: {', '.join(v)}")
    return "\n".join(L) + "\n"

