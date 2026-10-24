"""Checker, invariance, and high-precision diagnostic experiments."""
from __future__ import annotations

from ._imports import (
    Any,
    Sequence,
    Tuple,
    dataclasses,
    field,
    math,
    np,
    torch,
)
from . import api as _api

# ============================================================================
# Section 15: diagnostics and checker tests (Table 18, Figures 3, 6, 9, 10)
# ============================================================================

def _exact_endpoint(system: _api.RTTSystem, ctx: _api.CallContext, logits_T_q) -> Tuple[float, Any]:
    """Offline second rollout: exact one-sided damage of the served float32 logits
    against the float32 incumbent: S+ = sum_i w_i D_inf(p^r_i || p^s_i)."""
    lr = _api.logp_from_logits(ctx.inc_logits[ctx.q_idx])
    ls = _api.logp_from_logits(logits_T_q)
    rows = (lr - ls).max(1).values
    return float(rows.sum().item()) / float(ctx.q_idx.shape[0]), rows


def crafted_execution(system: _api.RTTSystem, ctx: _api.CallContext, depth: int, v, kind: str = "tr") -> _api.ExecResult:
    """A trajectory equal to the incumbent's up to `depth`, with step v added at depth+1."""
    inc = system.inc
    W = inc.weight()
    states = list(ctx.inc_states[: depth + 1])
    H = states[-1]
    for l in range(depth, system.T):
        Hb = inc.transition(H, ctx.H0, ctx.op, W, ctx, l)
        H = Hb + v if l == depth else Hb
        states.append(H)
    rec = _api.DepthRecord(depth=depth, kind=system.cfg.depth_kind(depth))
    rec.t = np.ones((system.cfg.K_roles, 1))
    return _api.ExecResult(policy="crafted", kind=kind, states=states, steps={depth: v}, records={depth: rec},
                      l0=depth, logits_T=inc.logits(H), B_left=0.0, time_s=0.0, start_depth=depth)


def planted_violation_test(system: _api.RTTSystem, checker: _api.Checker, env: _api.BaseEnv, cfg: _api.RTTConfig, n_traj: int,
                           n_margins: int, seed: int) -> dict:
    """Trajectories whose exact endpoint damage exceeds the budget by a hidden margin
    in [1e-7, 1e-3] nats (and rows exceeding H_row by such a margin) must be rejected."""
    rng = np.random.default_rng(_api.stable_seed("planted", seed))
    margins = np.exp(rng.uniform(math.log(1e-7), math.log(1e-3), size=n_margins))
    tested = rejected = 0
    row_tested = row_rejected = 0
    ratios = []
    depth_choices = [cfg.T - 1, max(cfg.open_first, cfg.T - cfg.b_window), cfg.open_first]
    for s in range(n_traj):
        ctx = env.build(env.generator.make("checker_planted", "final19", "mixture", s))
        system.prepare_call(ctx, time_prefix=False)
        depth = depth_choices[s % len(depth_choices)]
        with torch.no_grad():
            direction = torch.from_numpy(rng.standard_normal((ctx.n, system.d)).astype(np.float32)).to(system.device)
            mask = torch.zeros(ctx.n, dtype=torch.bool, device=system.device)
            reach = ctx.hop <= 1
            mask[reach] = True
            direction = direction * mask.unsqueeze(1).to(torch.float32)
            direction = direction / (torch.linalg.norm(direction, dim=1, keepdim=True) + 1e-30)
            scale = 0.05 * float(rng.uniform(0.2, 1.0))
            v = direction * scale
            res = crafted_execution(system, ctx, depth, v)
            exact, rows = _exact_endpoint(system, ctx, res.logits_T[ctx.q_idx])
            spec = _api.PolicySpec(name="planted", kind="tr", H_plus=10.0, H_row=float("inf"))
            cert = checker.certify_once(ctx, res, spec, cfg.spectral_rank)
        ratios.append(cert.damage_bound / max(exact, 1e-300))
        for m in margins:
            tested += 1
            budget = exact - m
            if not (cert.damage_bound <= budget):
                rejected += 1
            row_tested += 1
            rbudget = float(rows.max().item()) - m
            if not (cert.row_bound_max <= rbudget):
                row_rejected += 1
    return {"damage_planted": tested, "damage_rejected": rejected, "row_planted": row_tested,
            "row_rejected": row_rejected, "certified_to_exact_median": float(np.median(ratios)) if ratios else None,
            "passed": bool(rejected == tested and row_rejected == row_tested)}


def high_precision_reevaluation(seed: int = 3, n_cert: int = 20, digits: int = 50) -> dict:
    """Compare toy certificates with independent high-precision endpoint losses.

    The mathematical charge starts at the *shared float32 prefix*, as does the
    checker. The final damage and row bounds are separately checked against the
    two actually emitted float32 logits. No comparison tolerance is used. These
    finite endpoint checks do not constitute an interval-arithmetic proof.
    """
    import decimal
    from decimal import Decimal
    if int(n_cert) < 1 or int(digits) < 34:
        raise ValueError("high-precision diagnostics require n_cert >= 1 and digits >= 34")
    cfg = _api.RTTConfig()
    cfg.T = 6
    cfg.open_first = 3
    cfg.b_window = 2
    cfg.hidden = 5
    cfg.enc_hidden = 6
    cfg.spectral_rank = 3
    cfg.spectral_rank_tight = 5
    dev = torch.device("cpu")
    rng = np.random.default_rng(_api.stable_seed("hp", seed))
    n, Fdim, C = 24, 4, 3
    g = _api.synthetic_graph(n=n, num_classes=C, num_features=Fdim, seed=seed, avg_degree=4.0)
    roles = _api.DataRoles(pools={k: np.arange(n) for k in ("fit", "inc_train", "inc_holdout", "temp18", "calib18",
                                                       "select18", "dev19", "final19", "reserved19", "transfer20")}, notes=[])
    env = _api.BaseEnv(g, roles, cfg, dev)
    _api.set_global_seeds(seed)
    inc = _api.TanhDiffusionIncumbent(Fdim, C, cfg, seed).to(dev)
    inc.freeze()
    inc.eval()
    for p in inc.parameters():
        p.requires_grad_(False)
    reg = {"gamma": inc.gamma_delta(), "open_depths": cfg.open_depths(), "probes": _api.make_probes(cfg.hidden, 2, seed)}
    sysm = _api.RTTSystem(inc, None, None, reg, cfg, dev)
    chk = _api.Checker(sysm)

    def D(x):
        # Preserve the exact represented binary input; repr(float(x)) instead
        # interprets its short decimal rendering as a different real number.
        return Decimal.from_float(float(x))

    def dtanh(x):
        e2 = (2 * x).exp()
        return (e2 - 1) / (e2 + 1)

    def matrix(x):
        return [[D(v) for v in row] for row in x]

    def log_probs(z):
        zm = max(z)
        logsum = sum((v - zm).exp() for v in z).ln()
        return [v - zm - logsum for v in z]

    def row_damage(zr, ze):
        lr, le = log_probs(zr), log_probs(ze)
        return max(a - b for a, b in zip(lr, le))

    cases = []
    # Do not change decimal precision for callers of this diagnostic.
    with decimal.localcontext() as precision:
        precision.prec = int(digits)
        Wd = matrix(inc.weight().detach().numpy())
        Ad = matrix(inc.A_eff(torch.float64).numpy())
        bd = [D(v) for v in inc.b_eff(torch.float64).numpy()]
        alpha, abar, tau, tbar = D(inc.alpha), D(1 - inc.alpha), D(inc.tau), D(1 - inc.tau)
        for c in range(n_cert):
            spec = _api.CallSpec(call_id=f"hp{c}", stage="hp", pool="final19", component="clean", index=c, nq=8, seed=c,
                            query=np.sort(rng.choice(n, size=8, replace=False)))
            ctx = env.build(spec)
            sysm.prepare_call(ctx, time_prefix=False)
            depth = int(rng.integers(cfg.open_first, cfg.T))
            v = torch.from_numpy((rng.standard_normal((n, cfg.hidden)) * 0.05).astype(np.float32))
            res = crafted_execution(sysm, ctx, depth, v)
            cert = chk.certify_once(ctx, res, _api.PolicySpec(name="hp", kind="tr", H_plus=10.0, H_row=float("inf")), cfg.spectral_rank)
            H0d = matrix(ctx.H0.detach().numpy())
            P = torch.sparse_coo_tensor(torch.stack([ctx.op.rows, ctx.op.cols]), ctx.op.vals64, (n, n)).to_dense().numpy()
            Pd = [[(k, D(P[i, k])) for k in range(n) if P[i, k] != 0] for i in range(n)]

            def ideal_tail(H_start, start):
                Hd = matrix(H_start)
                for j in range(start, cfg.T):
                    PH = [[sum(pik * Hd[k][u] for k, pik in Pd[i]) for u in range(cfg.hidden)] for i in range(n)]
                    Z = [[sum(PH[i][a] * Wd[a][u] for a in range(cfg.hidden)) for u in range(cfg.hidden)] for i in range(n)]
                    Hd = [[alpha * H0d[i][u] + abar * (tbar * Hd[i][u] + tau * dtanh(Z[i][u]))
                           for u in range(cfg.hidden)] for i in range(n)]
                    if j + 1 in ctx.mask_rows:
                        for row, values in zip(ctx.mask_rows[j + 1].tolist(), ctx.mask_vals[j + 1].tolist()):
                            Hd[row] = [D(value) for value in values]
                return Hd

            def ideal_logits(H, i):
                return [sum(H[i][a] * Ad[a][cc] for a in range(cfg.hidden)) + bd[cc] for cc in range(C)]

            q = ctx.q_idx.numpy().tolist()
            weights = [D(ctx.w64[i].item()) for i in q]
            # Charge telescoping begins at the reused float32 prefix. Replaying
            # that prefix in real arithmetic would change the reference state.
            Hr = ideal_tail(ctx.inc_states[res.start_depth].detach().numpy(), res.start_depth)
            He = matrix(res.states[-1].detach().numpy())
            ideal_rows = [row_damage(ideal_logits(Hr, i), ideal_logits(He, i)) for i in q]
            ideal_damage = sum(w * value for w, value in zip(weights, ideal_rows))
            charge_gap = D(cert.charge_total) - ideal_damage

            # The release predicate includes incumbent float32 continuation and
            # head allowances. Check that final claim against actual emitted
            # logits, independently of the mathematical-tail comparison above.
            served_rows = [row_damage([D(v) for v in ctx.inc_logits[i].tolist()],
                                      [D(v) for v in res.logits_T[i].tolist()]) for i in q]
            served_damage = sum(w * value for w, value in zip(weights, served_rows))
            damage_gap = D(cert.damage_bound) - served_damage
            row_gap = min(D(bound.item()) - value for bound, value in zip(cert.row_bounds, served_rows))

            # Retain the old comparison to make its reference mismatch visible;
            # it is not the quantity asserted by charge_total.
            Hr_origin = ideal_tail(ctx.H0.detach().numpy(), 0)
            origin_damage = sum(w * row_damage(ideal_logits(Hr_origin, i), ideal_logits(He, i))
                                for w, i in zip(weights, q))
            origin_gap = D(cert.charge_total) - origin_damage
            cases.append({"index": c, "start_depth": res.start_depth,
                          "charge_minus_ideal_damage": float(charge_gap),
                          "damage_bound_minus_served_damage": float(damage_gap),
                          "min_row_bound_minus_served_row_damage": float(row_gap),
                          "legacy_origin_reference_gap": float(origin_gap),
                          "charge_enclosed": bool(charge_gap >= 0),
                          "served_damage_enclosed": bool(damage_gap >= 0),
                          "served_rows_enclosed": bool(row_gap >= 0)})
    failed = sum(not (c["charge_enclosed"] and c["served_damage_enclosed"] and c["served_rows_enclosed"]) for c in cases)
    min_charge = min(c["charge_minus_ideal_damage"] for c in cases)
    min_damage = min(c["damage_bound_minus_served_damage"] for c in cases)
    min_row = min(c["min_row_bound_minus_served_row_damage"] for c in cases)
    return {"certificates": n_cert, "digits": digits, "float64_below_exact_count": failed,
            "charge_below_ideal_count": sum(not c["charge_enclosed"] for c in cases),
            "damage_below_served_count": sum(not c["served_damage_enclosed"] for c in cases),
            "row_below_served_count": sum(not c["served_rows_enclosed"] for c in cases),
            "min_certified_minus_exact": min(min_charge, min_damage, min_row),
            "min_charge_minus_ideal_damage": min_charge,
            "min_damage_bound_minus_served_damage": min_damage,
            "min_row_bound_minus_served_row_damage": min_row,
            "legacy_origin_reference_below_count": sum(c["legacy_origin_reference_gap"] < 0 for c in cases),
            "reference": "shared_float32_prefix_for_charges_and_actual_float32_logits_for_release",
            "comparison_tolerance": 0.0, "cases": cases, "passed": failed == 0}


def adversarial_test(system: _api.RTTSystem, checker: _api.Checker, env: _api.BaseEnv, cfg: _api.RTTConfig, member_spec: _api.PolicySpec,
                     n_calls: int) -> dict:
    """Candidates replaced by (trimmed) ascent directions of the true loss and heads
    fooled into predicting descent: the budget bound must still hold."""
    def cands(ctx, l, Hbar, rows):
        g = _api.true_slope_field(system, ctx, l, Hbar)[rows]
        g = g / (torch.linalg.norm(g, dim=1, keepdim=True) + 1e-30)
        return torch.stack([g, g * 0.5], 0)

    def field(ctx, l, Hbar, rows):
        return -_api.true_slope_field(system, ctx, l, Hbar)[rows]
    adv = {"candidates": cands, "field": field}
    spec = dataclasses.replace(member_spec, name="adversarial", kind="adversarial")
    dmg, released, regress, maxd = [], 0, 0, 0.0
    for s in range(n_calls):
        ctx = env.build(env.generator.make("checker_adversarial", "final19", "mixture", s))
        system.prepare_call(ctx, time_prefix=False)
        served, res, cert, _ = _api.serve(system, checker, ctx, spec, adversary=adv)
        exact, _ = _exact_endpoint(system, ctx, served)
        lr = _api.logp_from_logits(ctx.inc_logits[ctx.q_idx])
        m = _api._served_metrics(ctx, lr, _api.logp_from_logits(served), ctx.y_q, cfg.zeta)
        if cert.mode in ("R1", "R2"):
            released += 1
            dmg.append(m["down"])
            maxd = max(maxd, exact)
        regress += int(m["down"] > cfg.zeta)
    return {"calls": n_calls, "released": released, "released_share": released / max(1, n_calls),
            "released_damage_mean": float(np.mean(dmg)) if dmg else 0.0,
            "released_damage_max": float(np.max(dmg)) if dmg else 0.0, "max_exact_S_plus": maxd,
            "H_plus": spec.H_plus, "bound_met": bool(maxd <= spec.H_plus * (1 + 1e-9)),
            "regression_rate": regress / max(1, n_calls)}


def underestimated_factor_test(system: _api.RTTSystem, checker: _api.Checker, env: _api.BaseEnv, cfg: _api.RTTConfig,
                               member_spec: _api.PolicySpec, n_calls: int, factor: float = 0.7) -> dict:
    spec = dataclasses.replace(member_spec, name="underestimated", provisional_inflation=factor)
    viol, caught, fails = 0, 0, 0
    for s in range(n_calls):
        ctx = env.build(env.generator.make("checker_underfactor", "final19", "mixture", s))
        system.prepare_call(ctx, time_prefix=False)
        res = system.run(ctx, spec)
        exact, rows = _exact_endpoint(system, ctx, res.logits_T[ctx.q_idx])
        cert = checker.certify(ctx, res, spec)
        truly_violating = (exact > spec.H_plus) or (math.isfinite(spec.H_row) and float(rows.max().item()) > spec.H_row)
        if cert.mode in ("R2", "F"):
            fails += 1
        if truly_violating:
            viol += 1
            if cert.mode == "F":
                caught += 1
    return {"calls": n_calls, "violating_calls": viol, "caught": caught, "first_pass_failures": fails,
            "passed": caught == viol}


def rounding_off_test(system: _api.RTTSystem, env: _api.BaseEnv, cfg: _api.RTTConfig, member_spec: _api.PolicySpec, n_calls: int) -> dict:
    raw_checker = _api.Checker(system, rounding=_api.Rounding(enabled=False), allowances=False)
    below, worst = 0, 0.0
    total = 0
    for s in range(n_calls):
        ctx = env.build(env.generator.make("checker_rounding_off", "final19", "mixture", s))
        system.prepare_call(ctx, time_prefix=False)
        res = system.run(ctx, member_spec)
        if res.l0 is None:
            continue
        exact, _ = _exact_endpoint(system, ctx, res.logits_T[ctx.q_idx])
        cert = raw_checker.certify_once(ctx, res, member_spec, cfg.spectral_rank)
        total += 1
        if cert.charge_total < exact:
            below += 1
            worst = max(worst, exact - cert.charge_total)
    return {"certificates": total, "below_exact": below, "max_shortfall": worst, "verdict": "diagnostic"}


INVARIANCE_OPS = ["duplicate", "duplicate_half", "permute", "interior", "spanning_set", "shorter"]


def _apply_candidate_op(D, op_name: str):
    if op_name == "duplicate":
        return torch.cat([D, D[:1]], 0)
    if op_name == "duplicate_half":
        return torch.cat([D, 0.5 * D[:1]], 0)
    if op_name == "permute":
        return torch.flip(D, dims=[0])
    if op_name == "interior":
        return torch.cat([D, (0.4 * D[0] + 0.4 * D[1]).unsqueeze(0)], 0)
    if op_name == "spanning_set":
        return torch.cat([D, (0.5 * D[0] + 0.5 * D[1]).unsqueeze(0)], 0)
    if op_name == "shorter":
        return torch.stack([0.5 * D[0], D[1]], 0)
    raise ValueError(op_name)


def invariance_test(system: _api.RTTSystem, env: _api.BaseEnv, cfg: _api.RTTConfig, member_spec: _api.PolicySpec, n_calls: int,
                    feature_modes: Sequence[str] = ("support", "mean")) -> dict:
    """Executed displacement change under candidate operations on the post-trim set."""
    out = {}
    depth = cfg.T - cfg.b_window - 1 if cfg.T - cfg.b_window - 1 >= cfg.open_first else cfg.open_first
    for fm in feature_modes:
        hk = "heads" if fm == "support" else ("heads_mean" if "heads_mean" in system.extra_heads else "heads")
        for opn in INVARIANCE_OPS:
            changes = []
            for s in range(n_calls):
                ctx = env.build(env.generator.make("invariance", "final19", "mixture", s))
                system.prepare_call(ctx, time_prefix=False)
                spec = dataclasses.replace(member_spec, name=f"inv_{fm}", feature_mode=fm, heads_key=hk,
                                           open_depths=[depth])
                r0 = system.run(ctx, spec, stop_after=depth)
                v0 = r0.steps.get(depth)
                adv = {"post_trim": lambda D, rows, opn=opn: _apply_candidate_op(D, opn)}
                r1 = system.run(ctx, spec, stop_after=depth, adversary=adv)
                v1 = r1.steps.get(depth)
                a = torch.zeros(1) if v0 is None else v0
                bb = torch.zeros(1) if v1 is None else v1
                if v0 is None and v1 is None:
                    changes.append(0.0)
                    continue
                if v0 is None:
                    v0 = torch.zeros_like(v1)
                if v1 is None:
                    v1 = torch.zeros_like(v0)
                den = float(torch.linalg.norm(v0).item())
                changes.append(float(torch.linalg.norm(v1 - v0).item()) / max(den, 1e-30) if den > 0 else
                               float(torch.linalg.norm(v1).item() > 0))
            out[f"{fm}:{opn}"] = {"median_rel_change": float(np.median(changes)) if changes else None,
                                  "max_rel_change": float(np.max(changes)) if changes else None,
                                  "calls": len(changes), "heads": hk}
    return out


def reach_level(system: _api.RTTSystem, ctx: _api.CallContext, res_post: _api.ExecResult, post, H_plus: float,
                steps_max: int = 10) -> Tuple[float, float]:
    """Numerical U~_reach: scale the posterior-slope trajectory's steps by s >= 1 while the
    exact endpoint damage (extra rollouts) stays within H+; returns (plug-in gain, s)."""
    inc = system.inc
    W = inc.weight()
    lr = _api.logp_from_logits(ctx.inc_logits[ctx.q_idx])
    wq = 1.0 / float(ctx.q_idx.shape[0])

    def run_scaled(s):
        H = ctx.inc_states[res_post.start_depth]
        for l in range(res_post.start_depth, system.T):
            Hb = inc.transition(H, ctx.H0, ctx.op, W, ctx, l)
            v = res_post.steps.get(l)
            H = Hb + s * v if v is not None else Hb
        ls = _api.logp_from_logits(inc.logits(H)[ctx.q_idx])
        dmg = float((lr - ls).max(1).values.sum().item()) * wq
        gain = float((post * (ls - lr)).sum(1).sum().item()) * wq
        return dmg, gain

    with torch.no_grad():
        d1, g1 = run_scaled(1.0)
        if not res_post.steps or d1 > H_plus:
            return g1, 1.0
        lo, hi = 1.0, 2.0
        while run_scaled(hi)[0] <= H_plus and hi < 64:
            lo, hi = hi, hi * 2.0
        for _ in range(steps_max):
            mid = 0.5 * (lo + hi)
            if run_scaled(mid)[0] <= H_plus:
                lo = mid
            else:
                hi = mid
        dL, gL = run_scaled(lo)
    return max(gL, g1), lo

