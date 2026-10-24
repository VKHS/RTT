"""Computational stages for fitting, calibration, audit, and diagnostics."""
from __future__ import annotations

from ._imports import (
    Dict,
    List,
    Optional,
    Path,
    Sequence,
    Tuple,
    dataclasses,
    math,
    np,
    time,
    torch,
)
from . import api as _api

# ============================================================================
# Section 17: stage task runners and gates
# ============================================================================

def _gen_calls(env: _api.BaseEnv, stage_tag: str, pool: str, component: str, a: int, b: int, nq: Optional[int] = None):
    return [env.generator.make(stage_tag, pool, component, i, nq) for i in range(a, b)]


def _range_from(task: str) -> Tuple[str, int, int]:
    head, rng_ = task.split("__", 1)
    a, b = rng_.split("_")
    return head, int(a), int(b)


def resolve_overrides(cfg: _api.RTTConfig, ov: Optional[dict]) -> dict:
    ov = dict(ov or {})
    if ov.get("open_depths") == "late8":
        ov["open_depths"] = list(range(max(cfg.open_first, cfg.T - 8), cfg.T))
    return ov


def shares_restricted(shares: Dict[int, float], depths: Optional[List[int]]) -> Dict[int, float]:
    if not depths:
        return shares
    sub = {l: v for l, v in shares.items() if int(l) in depths}
    s = sum(sub.values())
    if s <= 0:
        return {l: 1.0 / len(depths) for l in depths}
    return {int(l): v / s for l, v in sub.items()}


def selected_member(suite: _api.Suite) -> Tuple[_api.MemberSpec, str]:
    if suite.has("selection.json"):
        sel = suite.load("selection.json", kind="json")
        name = sel.get("selected")
        if name:
            return _api.member_by_name(name), "SELECTED"
        return _api.member_by_name(sel.get("reporting_member", "m4_25zeta")), "NOT_SELECTED"
    return _api.member_by_name("m4_25zeta"), "SELECTION_NOT_RUN"


def registered_policy(suite: _api.Suite, cfg: _api.RTTConfig, proposal: str = "native") -> Tuple[_api.PolicySpec, str]:
    m, status = selected_member(suite)
    key = f"{proposal}__{m.name}" if proposal == "native" else f"{proposal}__m4_25zeta"
    radii = _api.load_radii(suite, key)
    if radii is None:
        raise _api.RTTFailure("ARTIFACT_MISSING", f"radii for {key} missing")
    name = "rtt_registered" if proposal == "native" else f"rtt_{proposal}"
    return _api.policy_for_member(cfg, m, radii, name=name), status


# ----------------------------------------------------------------------------

def task_incumbent(suite, args, cfg, device, task, run_dir) -> dict:
    g = _api.load_graph(args)
    roles = _api.make_data_roles(g, cfg, seed=cfg.seed)
    env = _api.BaseEnv(g, roles, cfg, device, suite_seed=cfg.seed)
    seed = cfg.seed if task == "main" else cfg.seed + 1
    _api.set_global_seeds(seed)
    inc = _api.TanhDiffusionIncumbent(g.num_features, g.num_classes, cfg, seed).to(device)
    hist = _api.train_incumbent(inc, env, roles, cfg, seed)
    temp = _api.fit_temperature(inc, env, roles.pools["temp18"], cfg)
    acc = _api.evaluate_incumbent_clean(inc, env, roles)
    gamma = inc.gamma_delta()
    wn = inc.meta.get("w_norm_exact")
    glob = (1 - cfg.alpha) * (1 - cfg.tau + cfg.tau * wn)
    inc.meta.update({"temperature": temp["temperature"], "gamma_delta": gamma, "accuracy": acc,
                     "global_factor": glob, "seed": seed, "graph": g.fingerprint()})
    man = suite.publish(f"incumbent_{task}.pt", inc.export_state())
    _api.write_csv(run_dir / "incumbent_history.csv", hist["history"])
    flops = _api.incumbent_flops_per_pass(g.n, 2 * g.num_und_edges + g.n, cfg.hidden, g.num_features, cfg.enc_hidden,
                                     g.num_classes, cfg.T)
    return {"incumbent": task, "seed": seed, "best_epoch": hist["best_epoch"],
            "best_holdout_acc": hist["best_holdout_acc"], "w_norm_exact": wn, "global_factor": glob,
            "gamma_delta": gamma, "temperature": temp, "accuracy": acc, "flops_per_pass": flops,
            "artifact": man, "data_roles": roles.summary(), "graph": g.fingerprint()}


def task_proposal(suite, args, cfg, device, task, run_dir) -> dict:
    L = _api.load_base(suite, args, cfg, device)
    seed = _api.stable_seed("proposal", task, cfg.seed) % (2 ** 31)
    if task == "native":
        prop = _api.NativeProposal(cfg.hidden, cfg, seed=cfg.seed).to(device)
        reg = {"gamma": L.inc.gamma_delta(), "open_depths": cfg.open_depths(),
               "probes": _api.make_probes(cfg.hidden, cfg.probes_per_sign, cfg.seed)}
        system = _api.RTTSystem(L.inc, prop, None, reg, cfg, device)
        info = _api.train_native_proposal(system, L.env, cfg, cfg.proposal_episodes, seed)
        man = suite.publish("proposal_native.pt", {"state_dict": {k: v.cpu() for k, v in prop.state_dict().items()},
                                                    "train": info})
        return {"proposal": "native", "train": info, "artifact": man}
    if task == "lora16":
        lora, info = _api.train_lora(L.inc, L.env, cfg, cfg.proposal_episodes, seed)
        sd = {k: v.cpu() for k, v in lora.state_dict().items() if not k.startswith("base.")}
        man = suite.publish("proposal_lora16.pt", {"state_dict": sd, "train": info})
        return {"proposal": "lora16", "train": info, "artifact": man}
    if task == "gprgnn":
        m, info = _api.train_gprgnn(L.env, L.roles, cfg, max(50, cfg.inc_epochs // 2), seed)
        man = suite.publish("proposal_gprgnn.pt", {"state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
                                                    "train": info})
        return {"proposal": "gprgnn", "train": info, "artifact": man}
    raise _api.RTTFailure("ORCHESTRATION", f"unknown proposal task {task}")


def _fresh_heads(L: _api.Loaded, cfg: _api.RTTConfig, device) -> _api.RTTHeads:
    n_open = len(cfg.open_depths())
    Fin = _api.feature_dim(L.inc.num_classes, n_open, cfg.probes_per_sign)
    return _api.RTTHeads(Fin, L.inc.num_classes, n_open, cfg).to(device)


def task_phase2(suite, args, cfg, device, task, run_dir) -> dict:
    seed = _api.stable_seed("phase2", task, cfg.seed) % (2 ** 31)
    if task.startswith("ft_"):
        if task not in _api.FT_VARIANTS:
            raise _api.RTTFailure("ORCHESTRATION", f"unknown fine-tuning variant {task}; expected one of {list(_api.FT_VARIANTS)}")
        L = _api.load_base(suite, args, cfg, device)
        ft, info = _api.finetune_incumbent(L.inc, L.env, L.roles, cfg, task, cfg.head_episodes, seed, device)
        man = suite.publish(f"{task}.pt", ft.export_state())
        return {"ft_variant": task, "train": info, "artifact": man}
    if task == "crossfit":
        L = _api.load_system(suite, args, cfg, device, "native", need_heads=False, registry_stage="phase2")
        cf, info = _api.train_crossfit_posterior(L.system, L.env, L.roles, cfg, seed)
        man = suite.publish("crossfit.pt", {"folds": [{k: v.cpu() for k, v in m.state_dict().items()} for m in cf.models],
                                            "temperature": cf.temperature, "info": info})
        return {"crossfit": info, "artifact": man}
    if task.startswith("correctors_"):
        prop = task.split("_", 1)[1]
        L = _api.load_system(suite, args, cfg, device, prop, need_heads=False, registry_stage="phase2")
        models, info = _api.train_correctors(L.system, L.env, cfg, cfg.head_episodes, seed)
        man = suite.publish(f"correctors_{prop}.pt", {k: {kk: vv.cpu() for kk, vv in m.state_dict().items()}
                                                      for k, m in models.items()})
        return {"correctors": info, "artifact": man}
    if task.startswith("heads_mean_") or task.startswith("heads_probe_"):
        variant, prop = task.split("_")[1], task.split("_", 2)[2]
        L = _api.load_system(suite, args, cfg, device, prop, need_heads=False, registry_stage="phase2")
        thr = _api.compute_role_thresholds(L.system, L.env, cfg)
        L.system.role_thr = thr
        key = f"heads_{variant}"
        L.system.extra_heads[key] = _fresh_heads(L, cfg, device)
        info = _api.train_heads(L.system, L.env, cfg, cfg.head_episodes, seed, heads_key=key,
                           feature_mode="mean" if variant == "mean" else "support",
                           slope_mode="probe" if variant == "probe" else "full")
        man = suite.publish(f"{key}_{prop}.pt", {"heads": {k: v.cpu() for k, v in L.system.extra_heads[key].state_dict().items()},
                                                  "train": info, "role_thresholds": thr})
        return {"heads_variant": key, "train": info, "artifact": man}
    if task.startswith("heads_"):
        prop = task.split("_", 1)[1]
        L = _api.load_system(suite, args, cfg, device, prop, need_heads=False, registry_stage="phase2")
        thr = _api.compute_role_thresholds(L.system, L.env, cfg)
        L.system.role_thr = thr
        L.system.reg["role_thresholds"] = thr
        L.system.heads = _fresh_heads(L, cfg, device)
        info = _api.train_heads(L.system, L.env, cfg, cfg.head_episodes, seed)
        n_open = len(cfg.open_depths())
        Fin = _api.feature_dim(L.inc.num_classes, n_open, cfg.probes_per_sign)
        L.system.tr_heads = {"tr_matched": _api.TrustRuleHead(Fin, cfg.K_roles, n_open, cfg).to(device),
                             "tr_e2e": _api.TrustRuleHead(Fin, cfg.K_roles, n_open, cfg).to(device)}
        info_tr = {m: _api.train_tr_head(L.system, L.env, cfg, cfg.head_episodes, seed + 7, m) for m in ("matched", "e2e")}
        obj = {"heads": {k: v.cpu() for k, v in L.system.heads.state_dict().items()},
               "tr_matched": {k: v.cpu() for k, v in L.system.tr_heads["tr_matched"].state_dict().items()},
               "tr_e2e": {k: v.cpu() for k, v in L.system.tr_heads["tr_e2e"].state_dict().items()},
               "train": info, "train_tr": info_tr}
        man = suite.publish(f"heads_{prop}.pt", obj)
        suite.publish(f"phase2_{prop}.json", {"role_thresholds": thr, "train": info, "train_tr": info_tr}, kind="json")
        return {"heads": prop, "train": info, "train_tr": info_tr, "role_thresholds": thr, "artifact": man}
    raise _api.RTTFailure("ORCHESTRATION", f"unknown phase2 task {task}")


def task_placement(suite, args, cfg, device, task, run_dir) -> dict:
    L = _api.load_system(suite, args, cfg, device, task, need_heads=True, registry_stage="phase2")
    pl = _api.run_placement(L.system, L.env, cfg, cfg.placement_episodes, _api.stable_seed("placement", task) % (2 ** 31))
    obj = {"caps": pl["caps"].tolist(), "caps_spend": pl["caps_spend"].tolist(), "growth": pl["growth"].tolist(),
           "trust_share": {str(k): v for k, v in pl["trust_share"].items()},
           "trust_share_spend": {str(k): v for k, v in pl["trust_share_spend"].items()},
           "admissible": {str(k): v for k, v in pl["admissible"].items()},
           "Pi": {str(k): v for k, v in pl["Pi"].items()}, "Lbar_q95": {str(k): v for k, v in pl["Lbar_q95"].items()},
           "efficiency": np.where(np.isfinite(pl["efficiency"]), pl["efficiency"], -1.0).tolist(),
           "gain": pl["gain"].tolist(), "spend": pl["spend"].tolist(), "water_level": pl["water_level"],
           "expected_spend_used": pl["expected_spend_used"], "episodes": pl["episodes"]}
    man = suite.publish(f"registry_{task}.json", obj, kind="json")
    eff_rows = []
    for l in cfg.open_depths():
        for k in range(cfg.K_roles):
            eff_rows.append({"depth": l, "role": k, "gain": pl["gain"][l, k], "spend": pl["spend"][l, k],
                             "efficiency": obj["efficiency"][l][k], "cap": pl["caps"][l, k],
                             "cap_spend": pl["caps_spend"][l, k]})
    _api.write_csv(run_dir / "placement_cells.csv", eff_rows)
    return {"proposal": task, "admissible": obj["admissible"], "trust_share": obj["trust_share"],
            "growth": obj["growth"], "artifact": man}


def task_calibration(suite, args, cfg, device, task, run_dir) -> dict:
    prop, key = task.split("__", 1)
    L = _api.load_system(suite, args, cfg, device, prop, need_heads=True)
    if key in _api.SWAPS:
        desc, ov = _api.SWAPS[key]
        ov = resolve_overrides(cfg, ov)
        hk = ov.get("heads_key", "heads")
        if hk != "heads" and hk not in L.system.extra_heads:
            return {"status_note": f"NOT_RUN: heads '{hk}' unavailable", "not_run": True}
        member = _api.member_by_name("m4_25zeta")
    else:
        member = _api.member_by_name(key)
        ov = {}
        desc = member.note
    shares = _api.trust_share_for(L.registry, ov)
    shares = shares_restricted({int(k): v for k, v in shares.items()}, ov.get("open_depths"))
    out = _api.calibrate_member(L.system, L.env, cfg, member, ov.get("caps_key", "caps"), shares, cfg.n_r,
                           stage_tag=f"calib:{task}", heads_key=ov.get("heads_key", "heads"), overrides=ov)
    out.update({"key": task, "overrides": {k: (v if not isinstance(v, float) or math.isfinite(v) else "inf")
                                           for k, v in ov.items()}, "description": desc})
    man = suite.publish(f"radii_{task}.json", out, kind="json")
    return {"calibration": task, "radii": out["radii"], "artifact": man}


def _evaluate_specs(L: _api.Loaded, cfg: _api.RTTConfig, specs: List[_api.CallSpec], policies: List[_api.PolicyDef], run_dir: Path,
                    extras: Sequence[str] = (), flops: Optional[dict] = None) -> List[dict]:
    ev = _api.Evaluator(L.system, L.checker, L.env, cfg, L.models, flops=flops)
    records = []
    t0 = time.time()
    for i, sp in enumerate(specs):
        records.extend(ev.evaluate_call(sp, policies, extras=extras))
        if (i + 1) % max(1, len(specs) // 10) == 0:
            _api.log(f"[EVAL] {i+1}/{len(specs)} calls ({time.time()-t0:.0f}s)")
    _api.write_records(run_dir / "records.jsonl.gz", records)
    return records


def _quick_stats(records: List[dict], cfg: _api.RTTConfig) -> dict:
    by = {}
    for r in records:
        if r.get("kind") == "group":
            continue
        by.setdefault(r["policy"], []).append(r)
    out = {}
    for k, v in by.items():
        ok = [r for r in v if r.get("status") == "OK"]
        if not ok:
            out[k] = {"n": 0, "errors": len(v)}
            continue
        out[k] = {"n": len(ok), "errors": len(v) - len(ok), "mean_gain": float(np.mean([r["gain"] for r in ok])),
                  "N_zeta": int(sum(r["down"] > cfg.zeta for r in ok)),
                  "violations": int(sum(bool(r.get("violation_damage")) for r in ok))}
    return out


def primary_policies(suite: _api.Suite, L: _api.Loaded, cfg: _api.RTTConfig, full: bool) -> List[_api.PolicyDef]:
    reg_spec, sel_status = registered_policy(suite, cfg)
    pols = [_api.PolicyDef("rtt_registered", "rtt", reg_spec, {"keep_for_groups": True, "selection_status": sel_status}),
            _api.PolicyDef("corr_2hop_rows", "corrector", None, {"corrector": "corr_2hop_rows", "H_plus": reg_spec.H_plus,
                                                             "H_row": cfg.H_row, "keep_for_groups": True})]
    if not full:
        pols += [_api.PolicyDef("tr_matched", "tr", dataclasses.replace(reg_spec, name="tr_matched", kind="tr", tr_key="tr_matched")),
                 _api.PolicyDef("raw", "raw", None, {}),
                 _api.PolicyDef("oracle_frontier", "oracle", None, {"H_plus": reg_spec.H_plus, "H_row": cfg.H_row})]
        return pols
    tol_radii = _api.load_radii(suite, "native__m1_tolerance_point")
    pols.append(_api.PolicyDef("rtt_plus_corrector", "rtt_plus_corrector", reg_spec, {"base": "rtt_registered"}))
    if tol_radii is not None:
        pols.append(_api.PolicyDef("rtt_tolerance_point", "rtt",
                              _api.policy_for_member(cfg, _api.member_by_name("m1_tolerance_point"), tol_radii, name="rtt_tolerance_point")))
    for name, hops, rows in _api.CORRECTOR_SPECS[1:]:
        pols.append(_api.PolicyDef(name, "corrector", None, {"corrector": name, "H_plus": reg_spec.H_plus, "H_row": cfg.H_row}))
    pols += [_api.PolicyDef("mixing_allocated", "mixing", None, {"allocation": "allocated", "H_plus": reg_spec.H_plus}),
             _api.PolicyDef("mixing_uniform", "mixing", None, {"allocation": "uniform", "H_plus": reg_spec.H_plus}),
             _api.PolicyDef("projection_allocated", "projection", None, {"allocation": "allocated", "H_plus": reg_spec.H_plus}),
             _api.PolicyDef("projection_uniform", "projection", None, {"allocation": "uniform", "H_plus": reg_spec.H_plus})]
    if suite.has("selection.json"):
        thr = suite.load("selection.json", kind="json").get("ltt_threshold")
        if thr is not None:
            pols.append(_api.PolicyDef("switch_ltt", "switch", None, {"threshold": thr}))
    pols += [_api.PolicyDef("tr_matched", "tr", dataclasses.replace(reg_spec, name="tr_matched", kind="tr", tr_key="tr_matched")),
             _api.PolicyDef("tr_e2e", "tr", dataclasses.replace(reg_spec, name="tr_e2e", kind="tr", tr_key="tr_e2e"))]
    for v in _api.FT_VARIANTS:
        if v in L.models.get("ft", {}):
            pols.append(_api.PolicyDef(v, "ft", None, {"variant": v}))
    pols.append(_api.PolicyDef("raw", "raw", None, {"keep_for_groups": True}))
    r6 = _api.load_radii(suite, "native__m6_per_depth_25zeta")
    if r6 is not None:
        pols.append(_api.PolicyDef("variant_per_depth_radii", "rtt",
                              _api.policy_for_member(cfg, _api.member_by_name("m6_per_depth_25zeta"), r6, name="variant_per_depth_radii")))
    for sw, (desc, ov) in _api.SWAPS.items():
        rr = _api.load_radii(suite, f"native__{sw}")
        if rr is None:
            continue
        ov2 = resolve_overrides(cfg, ov)
        hk = ov2.get("heads_key", "heads")
        if hk != "heads" and hk not in L.system.extra_heads:
            continue
        sp = _api.policy_for_member(cfg, _api.member_by_name("m4_25zeta"), rr, name=sw, overrides=ov2)
        pols.append(_api.PolicyDef(sw, "rtt", sp, {"description": desc}))
    if "crossfit" in L.models:
        pols.append(_api.PolicyDef("posterior_slope_gate", "posterior_gate",
                              dataclasses.replace(reg_spec, name="posterior_slope_gate", kind="posterior_gate")))
        pols.append(_api.PolicyDef("oracle_frontier", "oracle", None, {"H_plus": reg_spec.H_plus, "H_row": cfg.H_row}))
    return pols


def task_selection(suite, args, cfg, device, task, run_dir) -> dict:
    L = _api.load_system(suite, args, cfg, device, "native", need_heads=True, need_models=("correctors",))
    if task == "ltt_switch":
        corr = L.models["correctors"]["corr_2hop_rows"]
        grid = [float("inf")] + list(np.geomspace(0.5, 1e-5, 40)) + [0.0]
        rows = []
        for i in range(cfg.selection_pool_calls):
            sp = L.env.generator.make("select:ltt", "select18", "mixture", i)
            ctx = L.env.build(sp)
            L.system.prepare_call(ctx)
            res_u = L.system.run(ctx, _api.PolicySpec(name="raw", kind="raw", certify=False))
            q = ctx.q_idx
            lr = _api.logp_from_logits(ctx.inc_logits[q])
            lu = _api.logp_from_logits(res_u.logits_T[q])
            with torch.no_grad():
                H_T = ctx.inc_states[-1]
                zl = L.inc.logits(H_T)
                qd = corr(corr.features(L.system, ctx, H_T, torch.softmax(zl, 1)), zl[q]).to(torch.float64)
            s_i = (qd * (lu - lr)).sum(1)
            y = ctx.y_q
            ce_r = -lr.gather(1, y.view(-1, 1)).squeeze(1)
            ce_u = -lu.gather(1, y.view(-1, 1)).squeeze(1)
            for t_i, thr in enumerate(grid):
                use = s_i > thr
                down = float(torch.where(use, ce_u, ce_r).mean().item() - ce_r.mean().item())
                rows.append({"call": i, "threshold_index": t_i, "threshold": thr, "down": down})
        _api.write_csv(run_dir / "ltt_switch.csv", rows)
        return {"ltt": True, "grid": grid, "calls": cfg.selection_pool_calls}
    parts = task.split("__")
    member = _api.member_by_name(parts[0])
    pool = parts[1]
    nq = int(parts[2][2:]) if len(parts) > 2 else None
    radii = _api.load_radii(suite, f"native__{member.name}")
    if radii is None:
        return {"not_run": True, "status_note": f"NOT_RUN: radii native__{member.name} missing"}
    spec = _api.policy_for_member(cfg, member, radii, name=f"rtt_{member.name}")
    stag = f"select:{pool}" + (f":nq{nq}" if nq else "")
    specs = [L.env.generator.make(stag, "select18", pool, i, nq) for i in range(cfg.selection_pool_calls)]
    recs = _evaluate_specs(L, cfg, specs, [_api.PolicyDef(spec.name, "rtt", spec)], run_dir)
    return {"member": member.name, "pool": pool, "nq": nq or cfg.nq, "quick": _quick_stats(recs, cfg)}


def task_development(suite, args, cfg, device, task, run_dir) -> dict:
    grp, a, b = _range_from(task)
    L = _api.load_system(suite, args, cfg, device, "native", need_heads=True, need_models=("correctors",))
    reg_spec, _ = registered_policy(suite, cfg)
    lib = _api.default_library()
    pols: List[_api.PolicyDef] = []
    specs = _gen_calls(L.env, "dev", "dev19", "mixture", a, b)
    if grp == "budget":
        members = {m.H_plus_mult: m for m in lib if m.radii_mode == "sequential"}
        for mult in cfg.budget_sweep_multipliers:
            near = min(members, key=lambda x: abs(math.log(x) - math.log(mult)))
            m = members[near]
            radii = _api.load_radii(suite, f"native__{m.name}")
            if radii is None:
                continue
            Hp = mult * cfg.zeta
            sp = _api.policy_for_member(cfg, m, radii, name=f"rtt_H{mult:g}z")
            sp.H_plus = Hp
            pols.append(_api.PolicyDef(sp.name, "rtt", sp, {"budget_mult": mult, "radii_member": m.name}))
            pols.append(_api.PolicyDef(f"corr_H{mult:g}z", "corrector", None, {"corrector": "corr_2hop_rows", "H_plus": Hp, "H_row": cfg.H_row}))
            pols.append(_api.PolicyDef(f"mix_H{mult:g}z", "mixing", None, {"allocation": "allocated", "H_plus": Hp}))
            pols.append(_api.PolicyDef(f"proj_H{mult:g}z", "projection", None, {"allocation": "allocated", "H_plus": Hp}))
            pols.append(_api.PolicyDef(f"tr_H{mult:g}z", "tr", dataclasses.replace(sp, name=f"tr_H{mult:g}z", kind="tr")))
            pols.append(_api.PolicyDef(f"sym_H{mult:g}z", "rtt", dataclasses.replace(sp, name=f"sym_H{mult:g}z", pricing="symmetric")))
        recs = _evaluate_specs(L, cfg, specs, pols, run_dir)
    elif grp == "callsize":
        recs = []
        sel = suite.load("selection.json", kind="json") if suite.has("selection.json") else {}
        for nq in cfg.call_sizes:
            sp_calls = _gen_calls(L.env, f"dev:nq{nq}", "dev19", "mixture", a, b, nq)
            pl = [_api.PolicyDef(f"rtt_registered_nq{nq}", "rtt", dataclasses.replace(reg_spec, name=f"rtt_registered_nq{nq}")),
                  _api.PolicyDef(f"corr_nq{nq}", "corrector", None, {"corrector": "corr_2hop_rows", "H_plus": reg_spec.H_plus, "H_row": cfg.H_row})]
            rname = (sel.get("reselected_by_call_size") or {}).get(str(nq))
            if rname:
                rr = _api.load_radii(suite, f"native__{rname}")
                if rr is not None:
                    pl.append(_api.PolicyDef(f"rtt_reselected_nq{nq}", "rtt",
                                        _api.policy_for_member(cfg, _api.member_by_name(rname), rr, name=f"rtt_reselected_nq{nq}")))
            sub = run_dir / f"nq{nq}"
            sub.mkdir(parents=True, exist_ok=True)
            recs.extend(_evaluate_specs(L, cfg, sp_calls, pl, sub))
        _api.write_records(run_dir / "records.jsonl.gz", recs)
    elif grp == "rowbudget":
        for hr in cfg.row_budgets:
            tag = "inf" if not math.isfinite(hr) else f"{hr:g}"
            pols.append(_api.PolicyDef(f"rtt_Hrow{tag}", "rtt", dataclasses.replace(reg_spec, name=f"rtt_Hrow{tag}", H_row=hr),
                                  {"H_row": hr}))
            pols.append(_api.PolicyDef(f"corr_Hrow{tag}", "corrector", None, {"corrector": "corr_2hop_rows",
                                                                          "H_plus": reg_spec.H_plus, "H_row": hr}))
        recs = _evaluate_specs(L, cfg, specs, pols, run_dir)
    elif grp == "robust":
        for tr in (0.4, 0.8, 1.2):
            pols.append(_api.PolicyDef(f"rtt_trim{tr:g}", "rtt", dataclasses.replace(reg_spec, name=f"rtt_trim{tr:g}", row_trim=tr)))
        for om in (0.7, 0.8, 0.9):
            pols.append(_api.PolicyDef(f"rtt_omega{om:g}", "rtt", dataclasses.replace(reg_spec, name=f"rtt_omega{om:g}", omega=om)))
        cal = suite.load(f"radii_native__{reg_spec.member}.json", kind="json") if suite.has(f"radii_native__{reg_spec.member}.json") else None
        if cal is not None and cal.get("mode") == "sequential":
            shares = L.registry.get("trust_share", {})
            for dc in (0.05, 0.1, 0.2, 0.4):
                radii = {int(l): _api.conformal_quantile(sc, dc * float(shares.get(int(l), 0.0))) for l, sc in cal["scores"].items()}
                pols.append(_api.PolicyDef(f"rtt_dcov{dc:g}", "rtt", dataclasses.replace(reg_spec, name=f"rtt_dcov{dc:g}", radii=radii),
                                      {"approximate": "radii re-quantiled from the registered member's sequential scores"}))
        recs = _evaluate_specs(L, cfg, specs, pols, run_dir)
    else:
        raise _api.RTTFailure("ORCHESTRATION", f"unknown development group {grp}")
    return {"group": grp, "calls": b - a, "quick": _quick_stats(recs, cfg)}


def task_audit(suite, args, cfg, device, task, run_dir) -> dict:
    comp, a, b = _range_from(task)
    L = _api.load_system(suite, args, cfg, device, "native", need_heads=True, need_models=("correctors", "crossfit", "ft"))
    full = comp == "mixture"
    pols = primary_policies(suite, L, cfg, full)
    specs = _gen_calls(L.env, "audit", "final19", comp, a, b)
    recs = _evaluate_specs(L, cfg, specs, pols, run_dir, extras=("groups",) if full else ())
    return {"component": comp, "calls": b - a, "quick": _quick_stats(recs, cfg), "policies": [p.name for p in pols]}


def task_breadth(suite, args, cfg, device, task, run_dir) -> dict:
    prop, a, b = _range_from(task)
    L = _api.load_system(suite, args, cfg, device, prop, need_heads=True, need_models=("correctors",))
    radii = _api.load_radii(suite, f"{prop}__m4_25zeta")
    if radii is None:
        return {"not_run": True, "status_note": f"NOT_RUN: radii {prop}__m4_25zeta missing"}
    sp = _api.policy_for_member(cfg, _api.member_by_name("m4_25zeta"), radii, name=f"rtt_{prop}")
    pols = [_api.PolicyDef(f"raw_{prop}", "raw", None, {}),
            _api.PolicyDef(f"tr_{prop}", "tr", dataclasses.replace(sp, name=f"tr_{prop}", kind="tr")),
            _api.PolicyDef(f"corr_{prop}", "corrector", None, {"corrector": "corr_2hop_rows", "H_plus": sp.H_plus, "H_row": cfg.H_row}),
            _api.PolicyDef(f"rtt_{prop}", "rtt", sp)]
    specs = _gen_calls(L.env, "breadth", "final19", "mixture", a, b)
    recs = _evaluate_specs(L, cfg, specs, pols, run_dir)
    return {"proposal": prop, "calls": b - a, "quick": _quick_stats(recs, cfg)}


def task_node_transfer(suite, args, cfg, device, task, run_dir) -> dict:
    L = _api.load_system(suite, args, cfg, device, "native", need_heads=True, need_models=("correctors", "ft"))
    reg_spec, sel_status = registered_policy(suite, cfg)
    if task == "per_node":
        reserved = L.roles.pools["reserved19"]
        nN = len(reserved)
        nq_node = int(min(cfg.nq, max(1, nN // 2)))
        chunks = [reserved[i:i + nq_node] for i in range(0, nN, nq_node)]
        rng = np.random.default_rng(_api.stable_seed("per_node_inspect", cfg.seed))
        predictions = {name: np.zeros(nN, dtype=np.int64)
                       for name in ("incumbent", "rtt_registered", "corr_2hop_rows")}
        pos = {int(v): i for i, v in enumerate(reserved)}
        for ci, ch in enumerate(chunks):
            sp = L.env.generator.make("per_node", "reserved19", "mixture", ci, len(ch))
            sp.query = np.sort(ch)
            sp.nq = len(ch)
            ctx = L.env.build(sp, need_labels=False)
            L.system.prepare_call(ctx)
            q = ctx.q_idx
            lr = _api.logp_from_logits(ctx.inc_logits[q])
            served, res, cert, _ = _api.serve(L.system, L.checker, ctx, reg_spec)
            ls = _api.logp_from_logits(served)
            ps, h, _ = _api.corrector_serve(L.system, L.models["correctors"]["corr_2hop_rows"], ctx, reg_spec.H_plus, cfg.H_row)
            lc = _api.logp_from_probs(ps)
            positions = np.asarray([pos[int(node)] for node in sp.query], dtype=np.int64)
            for name, lp in (("incumbent", lr), ("rtt_registered", ls), ("corr_2hop_rows", lc)):
                predictions[name][positions] = lp.argmax(1).cpu().numpy()
        # Commit the population's predictions before inspecting a random label
        # sample; no full-population correctness/flip diagnostic is computed.
        pred_path = run_dir / "reserved_predictions.npz"
        np.savez(pred_path, node_ids=reserved, **predictions)
        n_cal = int(min(cfg.node_inspect, max(1, nN // 2)))
        insp = rng.choice(nN, size=n_cal, replace=False)
        label_rows = torch.from_numpy(reserved[insp]).to(L.env.device)
        inspected_labels = L.env.y[label_rows].cpu().numpy()
        correct_ref = predictions["incumbent"][insp] == inspected_labels
        out = {"n_N": nN, "n_cal": n_cal, "calls": len(chunks), "call_size": nq_node,
               "selection_status": sel_status, "certificate_status": _api.CERTIFICATE_STATUS,
               "prediction_sha256": _api.sha256_file(pred_path),
               "inspected_node_ids": reserved[insp].tolist(), "uninspected_labels_accessed": False}
        for name in ("rtt_registered", "corr_2hop_rows"):
            k = int((correct_ref & (predictions[name][insp] != inspected_labels)).sum())
            bound, Fh = _api.hypergeom_node_bound(nN, n_cal, k, cfg.delta)
            out[name] = {"inspected_flips": k, "u_node": bound, "F_hat": Fh}
        _api.write_json(run_dir / "per_node.json", out)
        return out
    _, a, b = _range_from(task)
    pols = [_api.PolicyDef("rtt_registered", "rtt", reg_spec),
            _api.PolicyDef("corr_2hop_rows", "corrector", None, {"corrector": "corr_2hop_rows", "H_plus": reg_spec.H_plus, "H_row": cfg.H_row}),
            _api.PolicyDef("tr_matched", "tr", dataclasses.replace(reg_spec, name="tr_matched", kind="tr"))]
    if "ft_kl" in L.models.get("ft", {}):
        pols.append(_api.PolicyDef("ft_kl", "ft", None, {"variant": "ft_kl"}))
    specs = _gen_calls(L.env, "transfer", "transfer20", "mixture", a, b)
    recs = _evaluate_specs(L, cfg, specs, pols, run_dir)
    return {"transfer": True, "calls": b - a, "quick": _quick_stats(recs, cfg)}


def task_diagnostics(suite, args, cfg, device, task, run_dir) -> dict:
    need = ("correctors", "crossfit") if task in ("frontier_levels",) else ("correctors",)
    if task == "checker_hp":
        out = _api.high_precision_reevaluation(seed=cfg.seed, n_cert=8 if cfg.scale_name == "preflight" else 20)
        _api.write_json(run_dir / "checker_hp.json", out)
        return out
    L = _api.load_system(suite, args, cfg, device, "native", need_heads=True, need_models=need)
    reg_spec, _ = registered_policy(suite, cfg)
    small = cfg.scale_name == "preflight"
    if task == "checker_planted":
        n_m = 100 if not small else 20
        n_t = max(1, int(math.ceil(cfg.checker_planted_violations / n_m)))
        out = _api.planted_violation_test(L.system, L.checker, L.env, cfg, n_t, n_m, cfg.seed)
    elif task == "checker_adversarial":
        out = _api.adversarial_test(L.system, L.checker, L.env, cfg, reg_spec, 4 if small else 64)
    elif task == "checker_underfactor":
        out = _api.underestimated_factor_test(L.system, L.checker, L.env, cfg, reg_spec, 4 if small else 256)
    elif task == "checker_rounding_off":
        out = _api.rounding_off_test(L.system, L.env, cfg, reg_spec, 4 if small else 256)
    elif task == "invariance":
        out = _api.invariance_test(L.system, L.env, cfg, reg_spec, 2 if small else 32)
    elif task == "frontier_levels":
        out = frontier_levels_task(L, cfg, reg_spec, run_dir)
    elif task == "depth_exact":
        out = depth_exact_task(L, cfg, reg_spec, run_dir)
    elif task.startswith("planted_g"):
        gamma = int(task[len("planted_g"):]) / 100.0
        out = planted_partition_task(suite, args, cfg, device, gamma, run_dir)
    else:
        raise _api.RTTFailure("ORCHESTRATION", f"unknown diagnostics task {task}")
    _api.write_json(run_dir / f"{task}.json", out)
    return out


def frontier_levels_task(L: _api.Loaded, cfg: _api.RTTConfig, reg_spec: _api.PolicySpec, run_dir: Path) -> dict:
    from ..frontier import continuous_floor_frontier
    rows = []
    post_spec = dataclasses.replace(reg_spec, name="posterior_slope_gate", kind="posterior_gate")
    corr = L.models["correctors"]["corr_2hop_rows"]
    n_calls = cfg.diag_exact_depth_calls
    for comp in list(_api.ALL_COMPONENTS) + ["mixture"]:
        for i in range(n_calls):
            sp = L.env.generator.make("frontier", "final19", comp, i)
            ctx = L.env.build(sp)
            L.system.prepare_call(ctx)
            q = ctx.q_idx
            post = L.models["crossfit"].predict(L.system, ctx)
            ctx.cache["crossfit_posterior"] = post
            lr = _api.logp_from_logits(ctx.inc_logits[q])
            pr = torch.exp(lr)
            _, _, u_out, frontier_meta = continuous_floor_frontier(
                post, pr, ctx.w64[q], reg_spec.H_plus, reg_spec.H_row)
            if not frontier_meta["converged"]:
                raise _api.RTTFailure("SCIENTIFIC", "continuous frontier diagnostic did not converge", details=frontier_meta)
            wq = 1.0 / float(q.shape[0])

            def plug(lp):
                return float((post * (lp - lr)).sum(1).sum().item()) * wq

            def real(lp):
                y = ctx.y_q
                return float((lp.gather(1, y.view(-1, 1)) - lr.gather(1, y.view(-1, 1))).sum().item()) * wq
            s_post, res_post, cert_post, _ = _api.serve(L.system, L.checker, ctx, post_spec)
            u_cert = plug(_api.logp_from_logits(s_post))
            u_reach, s_scale = _api.reach_level(L.system, ctx, res_post, post, reg_spec.H_plus)
            s_rtt, _, cert_rtt, _ = _api.serve(L.system, L.checker, ctx, reg_spec)
            ps, _, _ = _api.corrector_serve(L.system, corr, ctx, reg_spec.H_plus, cfg.H_row)
            rows.append({"component": comp, "index": i, "U_out_hat": u_out, "U_reach_tilde": max(u_reach, u_cert),
                         "frontier_metadata": frontier_meta,
                         "U_cert_tilde": u_cert, "U_RTT_hat_plugin": plug(_api.logp_from_logits(s_rtt)),
                         "U_RTT_realized": real(_api.logp_from_logits(s_rtt)),
                         "corrector_plugin": plug(_api.logp_from_probs(ps)), "corrector_realized": real(_api.logp_from_probs(ps)),
                         "reach_scale": s_scale, "rtt_mode": cert_rtt.mode, "posterior_mode": cert_post.mode})
    _api.write_csv(run_dir / "frontier_levels.csv", rows)
    agg = {}
    for comp in set(r["component"] for r in rows):
        rr = [r for r in rows if r["component"] == comp]
        agg[comp] = {k: float(np.mean([r[k] for r in rr])) for k in
                     ("U_out_hat", "U_reach_tilde", "U_cert_tilde", "U_RTT_hat_plugin", "U_RTT_realized",
                      "corrector_plugin", "corrector_realized")}
    return {"per_population": agg, "calls_per_population": n_calls,
            "frontier_algorithm": "continuous_active_sets_and_dual_bisection",
            "U_out_interpretation": "numerically solved optimum for supplied cross-fitted posterior; not a Bayes-frontier guarantee"}


def depth_exact_task(L: _api.Loaded, cfg: _api.RTTConfig, reg_spec: _api.PolicySpec, run_dir: Path) -> dict:
    """Figure 9(d): certified-to-exact per-depth charge ratios (exact via two tails)."""
    inc = L.inc
    rows = []
    for i in range(cfg.diag_exact_depth_calls):
        sp = L.env.generator.make("depth_exact", "final19", "mixture", i)
        ctx = L.env.build(sp)
        L.system.prepare_call(ctx)
        res = L.system.run(ctx, reg_spec)
        if res.l0 is None:
            continue
        cert = L.checker.certify_once(ctx, res, reg_spec, cfg.spectral_rank)
        W = inc.weight()
        for l, v in res.steps.items():
            Hbar = inc.transition(res.states[l], ctx.H0, ctx.op, W, ctx, l)
            st0, _ = _api.tail_forward(inc, ctx, Hbar, l + 1)
            st1, _ = _api.tail_forward(inc, ctx, Hbar + v, l + 1)
            l0p = _api.logp_from_logits(inc.logits(st0[-1])[ctx.q_idx])
            l1p = _api.logp_from_logits(inc.logits(st1[-1])[ctx.q_idx])
            exact = float((l0p - l1p).max(1).values.sum().item()) / float(ctx.q_idx.shape[0])
            ch = cert.charges.get(l)
            rows.append({"call": i, "depth": l, "kind": cfg.depth_kind(l), "certified": ch, "exact": exact,
                         "ratio": (ch / exact) if exact > 0 else None, "charge_kind": cert.charge_kind.get(l)})
    _api.write_csv(run_dir / "depth_exact.csv", rows)
    out = {}
    for l in sorted(set(r["depth"] for r in rows)):
        rr = [r["ratio"] for r in rows if r["depth"] == l and r["ratio"] is not None]
        if rr:
            out[str(l)] = {"median": float(np.median(rr)), "q05": float(np.quantile(rr, 0.05)),
                           "q95": float(np.quantile(rr, 0.95)), "n": len(rr)}
    return {"ratio_by_depth": out, "violations": int(sum(1 for r in rows if r["certified"] is not None and r["certified"] < r["exact"]))}


def planted_partition_task(suite, args, cfg, device, gamma: float, run_dir: Path) -> dict:
    from ..frontier import continuous_floor_frontier
    """Appendix E: planted community (30% of rows by degree rank) with a known label law;
    the output frontier is exact (water-filling with the known law)."""
    L = _api.load_system(suite, args, cfg, device, "native", need_heads=False, registry_stage="final")
    g = L.g
    env = L.env
    with torch.no_grad():
        z = L.inc.deployed_logits(env.x, env.clean_op).to(torch.float64)
    deg = g.degree()
    thr = np.quantile(deg, 1.0 - cfg.planted_fraction)
    comm = torch.from_numpy(deg >= thr).to(device)
    zlaw = torch.where(comm.unsqueeze(1), (1.0 - gamma) * z, z)
    pB_all = torch.softmax(zlaw, 1)
    rng = np.random.default_rng(_api.stable_seed("planted_labels", gamma, cfg.seed))
    u = torch.from_numpy(rng.random(g.n)).to(device).to(torch.float64)
    cdf = torch.cumsum(pB_all, 1)
    y_new = torch.clamp((cdf < u.unsqueeze(1)).sum(1), max=g.num_classes - 1)
    env.y = y_new
    _api.fit_temperature(L.inc, env, L.roles.pools["temp18"], cfg)
    L.system.reg["role_thresholds"] = _api.compute_role_thresholds(L.system, env, cfg)
    L.system.role_thr = L.system.reg["role_thresholds"]
    L.system.heads = _fresh_heads(L, cfg, device)
    L.system.A32 = L.inc.A_eff(torch.float32)
    L.system.A64 = L.inc.A_eff(torch.float64)
    steps = max(8, cfg.head_episodes // 8)
    _api.train_heads(L.system, env, cfg, steps, _api.stable_seed("planted", gamma) % (2 ** 31))
    L.checker = _api.Checker(L.system)
    member = _api.member_by_name("m4_25zeta")
    cal = _api.calibrate_member(L.system, env, cfg, member, "caps", L.registry.get("trust_share", {}),
                           max(16, cfg.n_r // 8), stage_tag=f"planted{gamma}")
    radii = {int(k): v for k, v in cal["radii"].items()}
    spec = _api.policy_for_member(cfg, member, radii, name="rtt_planted")
    post_spec = dataclasses.replace(spec, name="posterior_known_law", kind="posterior_gate")
    rows = []
    n_calls = max(8, cfg.diag_exact_depth_calls)
    for i in range(n_calls):
        sp = env.generator.make(f"planted{gamma}", "dev19", "clean", i)
        ctx = env.build(sp)
        L.system.prepare_call(ctx)
        q = ctx.q_idx
        pB = pB_all[q]
        ctx.cache["crossfit_posterior"] = pB
        lr = _api.logp_from_logits(ctx.inc_logits[q])
        pr = torch.exp(lr)
        _, _, u_out, frontier_meta = continuous_floor_frontier(
            pB, pr, ctx.w64[q], spec.H_plus, spec.H_row)
        if not frontier_meta["converged"]:
            raise _api.RTTFailure("SCIENTIFIC", "continuous planted-law frontier did not converge", details=frontier_meta)
        wq = 1.0 / float(q.shape[0])

        def expg(lp):
            return float((pB * (lp - lr)).sum(1).sum().item()) * wq
        s_post, res_post, _, _ = _api.serve(L.system, L.checker, ctx, post_spec)
        u_cert = expg(_api.logp_from_logits(s_post))
        u_reach, _ = _api.reach_level(L.system, ctx, res_post, pB, spec.H_plus)
        s_rtt, _, cert, _ = _api.serve(L.system, L.checker, ctx, spec)
        rows.append({"index": i, "U_out": u_out, "U_reach_tilde": max(u_reach, u_cert), "U_cert_tilde": u_cert,
                     "frontier_metadata": frontier_meta,
                     "U_RTT_hat": expg(_api.logp_from_logits(s_rtt)), "mode": cert.mode})
    _api.write_csv(run_dir / "planted_rows.csv", rows)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("U_out", "U_reach_tilde", "U_cert_tilde", "U_RTT_hat")}
    return {"gamma": gamma, "community_rule": f"degree >= {thr:.1f} (top {cfg.planted_fraction:.0%})",
            "calls": n_calls, "levels": agg, "decomposition_terms": "NOT_COMPUTED (I, E, chi, Upsilon require "
            "information-restricted conditional expectations; see Appendix E caveats)"}

