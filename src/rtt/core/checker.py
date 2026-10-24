"""Independent numerical checking and whole-call fallback."""
from __future__ import annotations

from ._imports import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    dataclass,
    field,
    math,
    np,
    time,
    torch,
)
from . import api as _api

# ============================================================================
# Section 9: the independent float64 checker and the release rule (Sec. 2.5,
#            Theorem 1, Appendix G)
# ============================================================================

@dataclass
class CertResult:
    passed: bool
    mode: str                       # R1 | R2 | F | NO_STEP | UNCERTIFIED
    first_fail: str = ""            # damage | row | descent (first failing predicate, first pass)
    first_fail_r2: str = ""
    damage_bound: float = float("nan")
    charge_total: float = float("nan")
    terminal_allowance: float = float("nan")
    row_bound_max: float = float("nan")
    descent_max: float = float("nan")
    charges: Dict[int, float] = field(default_factory=dict)
    charge_kind: Dict[int, str] = field(default_factory=dict)
    descent: Dict[int, float] = field(default_factory=dict)
    row_bounds: Any = None          # per scored row certified D_inf bound (float64 tensor)
    factor_mean: Dict[int, float] = field(default_factory=dict)
    lambda_growth: Dict[int, float] = field(default_factory=dict)
    alloc: Dict[str, float] = field(default_factory=dict)
    rank: int = 0
    time_s: float = 0.0


class Checker:
    """Recomputes, independently of the executor, every quantity the release
    predicates depend on.  All arithmetic is float64 with a-priori outward
    inflation; the executed float32 pass enters through its measured residuals
    and the float32 incumbent continuation through gamma_m forward-error bounds."""

    def __init__(self, system: _api.RTTSystem, rounding: Optional[_api.Rounding] = None, allowances: bool = True):
        self.sys = system
        self.inc = system.inc
        self.cfg = system.cfg
        self.dev = system.device
        self.rounding = rounding or _api.Rounding(self.cfg.rel_inflation64, self.cfg.abs_inflation64, True)
        self.allowances = allowances
        W = self.inc.weight()
        self.W64 = W.detach().to(torch.float64)
        self.absW64 = self.W64.abs()
        self.A64 = self.inc.A_eff(torch.float64)
        self.b64 = self.inc.b_eff(torch.float64)
        self.Wh_abs = self.inc.head.weight.detach().to(torch.float64).abs().t()   # d x C
        self.bh_abs = self.inc.head.bias.detach().to(torch.float64).abs()
        self.temp = float(self.inc.temperature.item())
        # Gamma_Delta recomputed here from the effective class vectors (pairwise differences,
        # no Gram cancellation); the registry value is only accepted if it is not smaller
        diffs = self.A64.t().unsqueeze(1) - self.A64.t().unsqueeze(0)          # C x C x d
        g_chk = float(torch.linalg.norm(diffs, dim=2).max().item())
        self.gamma = self.rounding.up(max(float(system.gamma), g_chk * (1.0 + 1e-12)))
        self.splits: Dict[int, _api.SpectralSplit] = {}
        self.d = self.inc.hidden
        self.alpha = self.inc.alpha
        self.tau = self.inc.tau
        self.T = self.inc.T
        self.C = self.inc.num_classes
        self.Wcol64 = torch.linalg.norm(self.W64, dim=0)                          # ||W_{:,u}||_2

    def split(self, k: int) -> _api.SpectralSplit:
        if k not in self.splits:
            self.splits[k] = _api.SpectralSplit(self.inc.weight(), k, device=self.dev)
        return self.splits[k]

    # ------------------------------------------------------------------
    def _transition64(self, ctx: _api.CallContext, H64, H0_64, j: int):
        PH = ctx.op.mm(H64)
        Z = PH @ self.W64
        actZ = torch.tanh(Z) if self.inc.activation == "tanh" else Z
        Fv = self.alpha * H0_64 + (1.0 - self.alpha) * ((1.0 - self.tau) * H64 + self.tau * actZ)
        rows = ctx.mask_rows.get(j + 1)
        if rows is not None:
            Fv = Fv.clone()
            Fv[rows] = ctx.mask_vals[j + 1].to(torch.float64)
        return Fv, Z, PH, actZ

    def _step_errors(self, ctx: _api.CallContext, H64, H0_64, PH, Z, actZ, r, j: int):
        """Per-row bounds: e64 (float64 recomputation of F) and e32 (float32
        incumbent evaluation of F anywhere in the tube row i)."""
        if not self.allowances:
            z = torch.zeros(H64.shape[0], dtype=torch.float64, device=H64.device)
            return z, z, torch.zeros_like(Z)
        m_sp = float(ctx.op.max_row_nnz)
        absH = H64.abs()
        PabsH = ctx.op.mm(absH)
        PabsH_r = PabsH + ctx.op.mm(r).unsqueeze(1)          # |X| <= |H| + r on the tube
        # float64 recomputation
        g64 = _api.gamma_m(m_sp + self.d + 2, _api.EPS64)
        EZ = g64 * (PabsH @ self.absW64)
        e_act64 = EZ + 4.0 * _api.EPS64 * actZ.abs()
        e64 = (1.0 - self.alpha) * self.tau * e_act64 + 4.0 * _api.EPS64 * (
            self.alpha * H0_64.abs() + (1.0 - self.alpha) * (1.0 - self.tau) * absH
            + (1.0 - self.alpha) * self.tau * actZ.abs())
        # float32 incumbent evaluation anywhere in the tube row
        g32 = _api.gamma_m(m_sp + self.d + 2, _api.EPS32)
        EZ32 = g32 * (PabsH_r @ self.absW64)
        # pre-activations anywhere in the tube row move by (P r)_i ||W_{:,u}|| (Lemma 1), plus the
        # float64/float32 evaluation errors; tanh is 1-Lipschitz and bounded by 1
        Zshift = ctx.op.mm(r).unsqueeze(1) * self.Wcol64.unsqueeze(0) + EZ + EZ32
        act_mag = torch.clamp(actZ.abs() + Zshift, max=1.0) if self.inc.activation == "tanh" else actZ.abs() + Zshift
        e_act32 = EZ32 + self.cfg.f32_tanh_ulps * _api.EPS32 * act_mag + 1e-45
        Xmag = absH + r.unsqueeze(1)
        # float32 combination alpha*H0 + (1-alpha)[(1-tau) X + tau act]: the Python constants are
        # rounded to float32 (<= 0.5 ulp each) before up to four roundings: gamma_6 covers both
        g6 = _api.gamma_m(6, _api.EPS32)
        e32 = (1.0 - self.alpha) * self.tau * e_act32 + g6 * (
            self.alpha * H0_64.abs() + (1.0 - self.alpha) * (1.0 - self.tau) * Xmag
            + (1.0 - self.alpha) * self.tau * act_mag)
        keep = ctx.keep(j + 1, torch.float64)
        e64n = torch.linalg.norm(e64, dim=1)
        e32n = torch.linalg.norm(e32, dim=1)
        if keep is not None:
            e64n = e64n * keep
            e32n = e32n * keep
        return self.rounding.up(e64n), self.rounding.up(e32n), EZ

    def _logit_err(self, Habs_rows):
        """max_c |fl32(z_ic) - z_ic| for float32 logits of states with |H| <= Habs."""
        g = _api.gamma_m(self.d + 2, _api.EPS32)
        mag = Habs_rows @ self.Wh_abs + self.bh_abs.unsqueeze(0)
        err = (g * mag + _api.EPS32 * mag) / self.temp
        return self.rounding.up(err.max(1).values)

    # ------------------------------------------------------------------
    def certify_once(self, ctx: _api.CallContext, res: _api.ExecResult, spec: _api.PolicySpec, rank: int) -> CertResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        rnd = self.rounding
        T = self.T
        n = ctx.n
        global_only = spec.factor_mode == "global"
        split = self.split(rank)
        # The executed and the incumbent float32 trajectories share the state at the
        # first open depth bit for bit (the executor reuses the incumbent's prefix), so
        # the telescoping identity (14) and the tube (Lemma 7) start there with r = 0.
        start = res.start_depth
        if res.states[start] is not ctx.inc_states[start] and not torch.equal(res.states[start], ctx.inc_states[start]):
            raise _api.RTTFailure("SCIENTIFIC", "executed prefix differs from the incumbent prefix")
        r = torch.zeros(n, dtype=torch.float64, device=self.dev)
        Einc = torch.zeros(n, dtype=torch.float64, device=self.dev)
        Ls: Dict[int, Any] = {}
        mus: Dict[int, Any] = {}
        vnorms: Dict[int, Any] = {}
        H0_64 = ctx.H0.to(torch.float64)
        veffs: Dict[int, Any] = {}
        e64s: Dict[int, Any] = {}
        Hbars: Dict[int, Any] = {}
        accepted = {l for l, rec in res.records.items() if rec.t is not None and np.any(np.asarray(rec.t) > 0)}
        for j in range(start, T):
            H64 = res.states[j].to(torch.float64)
            Fv, Z, PH, actZ = self._transition64(ctx, H64, H0_64, j)
            veff = res.states[j + 1].to(torch.float64) - Fv
            e64n, e32n, EZ = self._step_errors(ctx, H64, H0_64, PH, Z, actZ, r, j)
            rho = ctx.op.mm(r)
            L, mu, _ = _api.interval_factors(Z, rho, split, self.inc.activation, torch.float64, margin_Z=EZ,
                                        rounding=rnd, global_only=global_only)
            Ls[j] = L
            mus[j] = mu
            vn = rnd.up(torch.linalg.norm(veff, dim=1) + e64n)
            vnorms[j] = vn
            if j in accepted:
                veffs[j] = veff
                e64s[j] = e64n
                Hbars[j] = Fv
            keep = ctx.keep(j + 1, torch.float64)
            r = _api.tube_step(ctx.op, r, L, keep, vn, self.alpha, self.tau, extra=e32n, rounding=rnd)
            Einc = _api.tube_step(ctx.op, Einc, L, keep, None, self.alpha, self.tau, extra=e32n, rounding=rnd)
        # ---- contracts ----
        lam, xi = _api.contracts_backward(ctx.op, T, start, lambda j: Ls[j], lambda j: mus[j], lambda j: ctx.keep(j),
                                     ctx.w64, self.gamma, self.alpha, self.tau, rounding=rnd)
        # ---- charges ----
        charges: Dict[int, float] = {}
        kinds: Dict[int, str] = {}
        descent: Dict[int, float] = {}
        for j in range(start, T):
            sig = _api.sum_upper(lam[j + 1] * vnorms[j], rnd)
            ch = sig
            kd = "sigma" if j in accepted else "residual"
            if j in accepted:
                dk = cfg.depth_kind(j)
                v = veffs[j]
                e64n = e64s[j]
                if dk == "window":
                    env = self._envelope(ctx, j, Hbars[j], v, xi[j + 1], e64n, lam[j + 1])
                    if env < ch:
                        ch, kd = env, "env+"
                elif dk == "last":
                    cp = self._cplus(ctx, Hbars[j], v, e64n)
                    if cp < ch:
                        ch, kd = cp, "C+"
                rec = res.records[j]
                if res.kind in ("rtt", "posterior_gate", "adversarial") and rec.g is not None:
                    descent[j] = self._phi_cert(ctx, j, rec, v, xi[j + 1], dk, Hbars[j])
            charges[j] = ch
            kinds[j] = kd
        charge_total = rnd.up(float(sum(charges.values())))
        # ---- terminal allowances and row bounds ----
        q = ctx.q_idx
        HeT = res.states[T][q].to(torch.float64).abs()
        rT = r[q]
        if self.allowances:
            eps_pi = self._logit_err(HeT)
            eps_r = self._logit_err(HeT + rT.unsqueeze(1))
            eps_sm = rnd.up(2.0 * (self.C + 6) * _api.EPS32)
            term_rows = 2.0 * (eps_pi + eps_r) + eps_sm
        else:
            term_rows = torch.zeros_like(rT)
        row_b = rnd.up(self.gamma * rT + term_rows)
        wq = ctx.w64[q]
        inc_dev = _api.sum_upper(wq * (self.gamma * Einc[q]), rnd) if self.allowances else 0.0
        term_allow = rnd.up(_api.sum_upper(wq * term_rows, rnd) + inc_dev)
        damage = rnd.up(charge_total + term_allow)
        row_max = float(row_b.max().item())
        dvals = list(descent.values())
        desc_bad = [d for d in dvals if not (math.isfinite(d) and d <= 0.0)]
        desc_max = (float("nan") if any(not math.isfinite(d) for d in dvals) else max(dvals)) if dvals else float("-inf")
        fail = ""
        if not (damage <= spec.H_plus):
            fail = "damage"
        elif math.isfinite(spec.H_row) and not (row_max <= spec.H_row):
            fail = "row"
        elif desc_bad:
            fail = "descent"
        cr = CertResult(passed=(fail == ""), mode="", first_fail=fail, damage_bound=damage,
                        charge_total=charge_total, terminal_allowance=term_allow, row_bound_max=row_max,
                        descent_max=desc_max, charges=charges, charge_kind=kinds, descent=descent,
                        row_bounds=row_b, rank=rank)
        cr.factor_mean = {j: float(Ls[j].mean().item()) for j in Ls}
        lamT = float(lam[T].sum().item())
        cr.lambda_growth = {j: float(lam[j + 1].sum().item()) / max(lamT, 1e-300) for j in range(start, T)}
        cr.alloc = {"incumbent_f32_deviation": inc_dev, "terminal_rows": float((wq * term_rows).sum().item()),
                    "residual_charges": float(sum(v for k, v in charges.items() if kinds[k] == "residual"))}
        cr.time_s = time.perf_counter() - t0
        return cr

    # ------------------------------------------------------------------
    def _envelope(self, ctx, l: int, Hbar64, v64, xi_next, e64n, lam_next) -> float:
        rnd = self.rounding
        states, derivs = _api.tail_forward(self.inc, ctx, Hbar64, l + 1, dtype=torch.float64)
        # float64 tail: logits of the reference continuation from H-bar_{l+1}
        zT = states[-1][ctx.q_idx] @ self.A64 + self.b64
        p = torch.softmax(zT, dim=1)
        XT = _api.tail_jvp(self.inc, ctx, [dv.to(torch.float64) for dv in derivs], l + 1, v64)
        dz = XT[ctx.q_idx] @ self.A64
        wq = ctx.w64[ctx.q_idx]
        lin_rows = torch.clamp((p * dz).sum(1) - dz.min(1).values, min=0.0)
        lin = _api.sum_upper(wq * lin_rows, rnd)
        dzmag = float((wq * dz.abs().max(1).values).sum().item())
        vn = torch.linalg.norm(v64, dim=1)
        quad = 0.5 * _api.sum_upper(xi_next * vn * vn, rnd)
        # margins: float64 tail/JVP error (relative to the magnitude of dz), and the offset of the
        # float64 expansion point from the exact H-bar_{l+1} (|Fv - H-bar| <= e64 per row): first
        # order sum_i Lambda_{l+1,i} e64_i plus the second-order cross term
        off1 = _api.sum_upper(lam_next * e64n, rnd)
        off2 = 2.0 * _api.sum_upper(xi_next * (e64n + 1e-300) * vn, rnd) + _api.sum_upper(xi_next * e64n * e64n, rnd)
        marg = 1e-9 * dzmag + off1 + off2 + 1e-15
        return rnd.up(lin + quad + marg)

    def _cplus(self, ctx, Hbar64, v64, e64n) -> float:
        rnd = self.rounding
        q = ctx.q_idx
        z = Hbar64[q] @ self.A64 + self.b64
        dz = v64[q] @ self.A64
        wq = ctx.w64[q]
        val = torch.clamp(torch.logsumexp(z + dz, 1) - torch.logsumexp(z, 1) - dz.min(1).values, min=0.0)
        marg = 2.0 * self.gamma * e64n[q] + (64.0 + 2.0 * (self.C + 4)) * _api.EPS64 * (
            1.0 + z.abs().max(1).values + dz.abs().max(1).values)
        return rnd.up(_api.sum_upper(wq * (val + marg), rnd) + 1e-15)

    def _phi_cert(self, ctx, l: int, rec: _api.DepthRecord, v64, xi_next, dk: str, Hbar64) -> float:
        """Robust descent model with certified remainders at the executed step (an
        upper bound: every sum carries its gamma_m accumulation bound)."""
        rnd = self.rounding
        cfg = self.cfg
        rows = rec.rows
        vr = v64[rows]
        g = rec.g.to(torch.float64)
        gv = g * vr
        lin = _api.sum_upper(gv, rnd)
        vn2_rows = (vr * vr).sum(1)
        unc2 = _api.sum_upper(rec.sig2.to(torch.float64) * vn2_rows, rnd) + rec.reg * _api.sum_upper(v64 * v64, rnd)
        unc = rnd.up(math.sqrt(max(unc2, 0.0)))
        vn2 = (v64 * v64).sum(1)
        if dk == "last":
            q = ctx.q_idx
            z = Hbar64[q] @ self.A64 + self.b64
            dz = v64[q] @ self.A64
            p = torch.softmax(z, 1)
            rem_rows = torch.logsumexp(z + dz, 1) - torch.logsumexp(z, 1) - (p * dz).sum(1)
            rem = _api.sum_upper(ctx.w64[q] * rem_rows, rnd) + 1e-12 * float((ctx.w64[q] * dz.abs().max(1).values).sum().item())
            XiT = (self.gamma ** 2 / 4.0) * ctx.w64
            vXi = _api.sum_upper(XiT * vn2, rnd)
            b = rem
        else:
            vXi = _api.sum_upper(xi_next * vn2, rnd)
            b = 0.5 * vXi
        R = 0.5 * (cfg.rho_D * vXi + cfg.rho_x * _api.sum_upper(vn2, rnd))
        omega = rec.omega if rec.omega is not None else cfg.omega
        phi = lin + rec.kappa * unc + (b + R) / omega
        return float(rnd.up(phi))

    # ------------------------------------------------------------------
    def certify(self, ctx: _api.CallContext, res: _api.ExecResult, spec: _api.PolicySpec) -> CertResult:
        cfg = self.cfg
        stepped = any((not bool(torch.isfinite(v).all().item())) or float(v.abs().max().item()) > 0.0
                      for v in res.steps.values())
        finite = all(bool(torch.isfinite(res.states[j]).all().item()) for j in range(res.start_depth, len(res.states)))
        consistent = False
        if finite and res.logits_T is not None and len(res.states) == self.T + 1:
            # the served logits must be a float32 evaluation of the head at the certified final
            # state: any evaluation order lies within _logit_err of the exact logits
            q = ctx.q_idx
            zq = res.logits_T[q].to(torch.float64)
            zr = self.inc.logits(res.states[-1][q]).to(torch.float64)
            tol = 2.0 * self._logit_err(res.states[-1][q].to(torch.float64).abs())
            consistent = torch.equal(res.states[res.start_depth], ctx.inc_states[res.start_depth]) and \
                bool(torch.isfinite(zq).all().item()) and \
                bool(((zq - zr).abs().max(1).values <= tol).all().item())
        if not consistent:
            return CertResult(passed=False, mode="F", first_fail="integrity", first_fail_r2="integrity",
                              damage_bound=float("inf"), charge_total=float("inf"),
                              terminal_allowance=float("nan"), row_bound_max=float("inf"), descent_max=float("nan"))
        if res.l0 is None and not stepped:
            # A zero-step shortcut must not authorize a tampered state/logit record.
            # Recheck the unmodified recurrence before identifying it with the
            # incumbent. This verification cost is included in checker time.
            intact = torch.equal(res.states[res.start_depth], ctx.inc_states[res.start_depth])
            with torch.no_grad():
                W = self.inc.weight()
                for j in range(res.start_depth, self.T):
                    expected = self.inc.transition(res.states[j], ctx.H0, ctx.op, W, ctx, j)
                    intact = intact and torch.equal(expected, res.states[j + 1])
                expected_logits = self.inc.logits(res.states[-1])
                intact = intact and torch.equal(expected_logits[ctx.q_idx], res.logits_T[ctx.q_idx])
            if not intact:
                return CertResult(passed=False, mode="F", first_fail="integrity", first_fail_r2="integrity",
                                  damage_bound=float("inf"), row_bound_max=float("inf"))
            return CertResult(passed=True, mode="NO_STEP", damage_bound=0.0, charge_total=0.0,
                              terminal_allowance=0.0, row_bound_max=0.0, descent_max=float("-inf"))
        c1 = self.certify_once(ctx, res, spec, cfg.spectral_rank)
        if c1.passed:
            c1.mode = "R1"
            return c1
        c2 = self.certify_once(ctx, res, spec, cfg.spectral_rank_tight)
        if c2.passed:
            c2.mode = "R2"
            c2.first_fail = c1.first_fail
            c2.time_s += c1.time_s
            return c2
        c1.mode = "F"
        c1.first_fail_r2 = c2.first_fail
        c1.time_s += c2.time_s
        return c1


def serve(system: _api.RTTSystem, checker: Optional[Checker], ctx: _api.CallContext, spec: _api.PolicySpec,
          hook: Optional[Callable] = None, adversary: Optional[dict] = None) -> Tuple[Any, _api.ExecResult, CertResult, dict]:
    """Execute + certify + release/fallback.  Returns (served logits on scored rows,
    exec result, certificate, timing)."""
    res = system.run(ctx, spec, hook=hook, adversary=adversary)
    q = ctx.q_idx
    prefix = ctx.cache["prefix_time"]
    t_prefix = prefix[res.start_depth]
    proposal_prepare = float(ctx.timing.get("proposal_prepare_s", 0.0))
    prepare_features = float(ctx.timing.get("prepare_features_s", 0.0))
    timing = {"prefix_s": t_prefix, "exec_s": res.time_s,
              "proposal_prepare_s": proposal_prepare,
              "prepare_features_s": prepare_features,
              "build_s": float(ctx.timing.get("build_s", 0.0))}
    if spec.kind == "raw" or not spec.certify:
        timing["certify_s"] = 0.0
        timing["fallback_s"] = 0.0
        timing["total_s"] = t_prefix + res.time_s + proposal_prepare + prepare_features
        cert = CertResult(passed=True, mode="UNCERTIFIED")
        return res.logits_T[q], res, cert, timing
    _api.cuda_sync()
    tc = time.perf_counter()
    cert = checker.certify(ctx, res, spec)
    _api.cuda_sync()
    timing["certify_s"] = time.perf_counter() - tc
    if cert.mode == "F":
        # whole-call fallback: incumbent tail from the cached state preceding the first intervention
        l0 = res.l0 if res.l0 is not None else res.start_depth
        if cert.first_fail == "integrity":
            l0 = res.start_depth
        _api.cuda_sync()
        tf = time.perf_counter()
        with torch.no_grad():
            H = ctx.inc_states[l0] if l0 < len(ctx.inc_states) else res.states[l0]
            W = system.inc.weight()
            for j in range(l0, system.T):
                H = system.inc.transition(H, ctx.H0, ctx.op, W, ctx, j)
            served = system.inc.logits(H)[q]
        _api.cuda_sync()
        timing["fallback_s"] = time.perf_counter() - tf
    elif cert.mode == "NO_STEP":
        # no displacement was executed: serve the incumbent's own float32 output
        timing["fallback_s"] = 0.0
        served = res.logits_T[q]
    else:
        timing["fallback_s"] = 0.0
        served = res.logits_T[q]
    timing["total_s"] = t_prefix + res.time_s + proposal_prepare + prepare_features + timing["certify_s"] + timing["fallback_s"]
    timing["total_with_build_s"] = timing["total_s"] + timing["build_s"]
    return served, res, cert, timing

