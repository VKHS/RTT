"""Policy execution and accepted internal state displacements."""
from __future__ import annotations

from ._imports import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    dataclass,
    dataclasses,
    field,
    math,
    np,
    time,
    torch,
)
from . import api as _api

# ============================================================================
# Section 8: the executor (Algorithm 1)
# ============================================================================

@dataclass
class PolicySpec:
    name: str
    kind: str = "rtt"                    # rtt | tr | raw | posterior_gate | adversarial
    H_plus: float = 0.05
    H_row: float = 1.0
    radii: Optional[Dict[int, float]] = None
    caps_key: str = "caps"
    open_depths: Optional[List[int]] = None
    window_mode: str = "two_branch"      # two_branch | env_only | sigma_only
    pricing: str = "one_sided"           # one_sided | symmetric
    factor_mode: str = "interval"        # interval | global
    feature_mode: str = "support"        # support | mean
    heads_key: str = "heads"
    provisional_inflation: float = 1.0
    screen: bool = True
    certify: bool = True
    member: str = ""
    tr_key: str = "tr_matched"
    row_trim: Optional[float] = None
    omega: Optional[float] = None

    def brief(self) -> dict:
        d = dataclasses.asdict(self)
        d["radii"] = None if self.radii is None else {str(k): v for k, v in self.radii.items()}
        return d


@dataclass
class DepthRecord:
    depth: int
    kind: str
    rows: Any = None            # support rows (LongTensor)
    g: Any = None               # field estimate on rows [n_s, d] (float32)
    sig2: Any = None            # covariance head on rows [n_s]
    kappa: float = 0.0
    reg: float = 0.0            # physical-coordinate covariance regulariser (x-space identity)
    t: Any = None
    branch: str = ""
    charge_prov: float = 0.0
    phi_prov: float = 0.0
    B_before: float = 0.0
    status: str = ""
    n_active_blocks: int = 0
    n_support: int = 0
    step_norm: float = 0.0
    newton: int = 0
    time_s: float = 0.0
    L_gm: float = float("nan")
    omega: Optional[float] = None


@dataclass
class ExecResult:
    policy: str
    kind: str
    states: List[Any]
    steps: Dict[int, Any]
    records: Dict[int, DepthRecord]
    l0: Optional[int]
    logits_T: Any
    B_left: float
    time_s: float
    start_depth: int
    diag: Dict[str, Any] = field(default_factory=dict)


class RTTSystem:
    """Frozen incumbent + proposal + heads + registry: everything Algorithm 1 needs."""

    def __init__(self, inc: _api.TanhDiffusionIncumbent, proposal, heads: Optional[_api.RTTHeads], registry: dict,
                 cfg: _api.RTTConfig, device, tr_heads: Optional[dict] = None, extra_heads: Optional[dict] = None):
        self.inc = inc
        self.proposal = proposal
        self.heads = heads
        self.tr_heads = tr_heads or {}
        self.extra_heads = extra_heads or {}
        self.reg = registry
        self.cfg = cfg
        self.device = device
        self.T = cfg.T
        self.C = inc.num_classes
        self.d = inc.hidden
        self.gamma = float(registry.get("gamma", inc.gamma_delta()))
        self.split_prov = _api.SpectralSplit(inc.weight(), cfg.spectral_rank, device=device)
        self.w_norm = self.split_prov.w_norm
        self.open_all = list(registry.get("open_depths", cfg.open_depths()))
        self.n_open = len(cfg.open_depths())
        self.probes = registry["probes"].to(device) if "probes" in registry else _api.make_probes(self.d, cfg.probes_per_sign, cfg.seed).to(device)
        self.role_thr = list(registry.get("role_thresholds", [0.5, 0.7, 0.9]))
        self.A32 = inc.A_eff(torch.float32)
        self.A64 = inc.A_eff(torch.float64)
        self._adj_hit_time = 0.0
        self._adj_counted = False
        self._post_hook = None

    # ------------------------------------------------------------------
    def depth_index(self, l: int) -> int:
        return l - self.cfg.open_first

    def caps_for(self, spec: PolicySpec) -> np.ndarray:
        caps = self.reg.get(spec.caps_key)
        if caps is None:
            caps = np.full((self.T, self.cfg.K_roles), self.cfg.cap_init, dtype=np.float64)
            caps[: self.cfg.open_first] = 0.0
        return np.asarray(caps, dtype=np.float64)

    def growth_for(self) -> Optional[np.ndarray]:
        g = self.reg.get("growth")
        return None if g is None else np.asarray(g, dtype=np.float64)

    def heads_for(self, spec: PolicySpec) -> Optional[_api.RTTHeads]:
        if spec.heads_key == "heads":
            return self.heads
        return self.extra_heads.get(spec.heads_key)

    # ------------------------------------------------------------------
    def prepare_call(self, ctx: _api.CallContext, time_prefix: bool = True,
                     prefix_only: bool = False, prefix_depth: Optional[int] = None) -> None:
        """Prepare the shared prefix, optionally adding an OFFLINE reference run.

        ``run`` always uses ``prefix_only=True``. Evaluation/training utilities
        explicitly use the default full reference for comparisons or targets;
        that endpoint is never an input of the served RTT policy.
        """
        inc = self.inc
        target = min(self.open_all) if (prefix_only and self.open_all) else self.T
        if prefix_only and prefix_depth is not None:
            target = int(prefix_depth)
        if not 0 <= target <= self.T:
            raise ValueError("prefix depth must lie in [0, T]")
        with torch.no_grad():
            if ctx.H0 is None:
                _api.cuda_sync()
                t0 = time.perf_counter()
                ctx.H0 = inc.encode(ctx.x)
                _api.cuda_sync()
                ctx.timing["encode_s"] = time.perf_counter() - t0
                ctx.inc_states = [ctx.H0]
                ctx.cache["prefix_time"] = [ctx.timing["encode_s"]]
                ts_features = time.perf_counter()
                ctx.cache["h0_lens"] = torch.softmax(inc.logits(ctx.H0), dim=1)
                ctx.cache["x_norm"] = torch.linalg.norm(ctx.x, dim=1) / math.sqrt(max(1, ctx.x.shape[1]))
                ctx.cache["h0_norm"] = torch.linalg.norm(ctx.H0, dim=1) / math.sqrt(self.d)
                _api.cuda_sync()
                ctx.timing["prepare_features_s"] = time.perf_counter() - ts_features
            if ctx.inc_states is None:
                raise _api.RTTFailure("ORCHESTRATION", "H0 exists without its reference prefix")
            W = inc.weight()
            while len(ctx.inc_states) <= target:
                j = len(ctx.inc_states) - 1
                _api.cuda_sync()
                ts = time.perf_counter()
                H = inc.transition(ctx.inc_states[-1], ctx.H0, ctx.op, W, ctx, j)
                if time_prefix:
                    _api.cuda_sync()
                elapsed = time.perf_counter() - ts
                ctx.inc_states.append(H)
                ctx.cache["prefix_time"].append(ctx.cache["prefix_time"][-1] + elapsed)
            if not prefix_only and ctx.inc_logits is None:
                _api.cuda_sync()
                ts = time.perf_counter()
                ctx.inc_logits = inc.logits(ctx.inc_states[self.T])
                _api.cuda_sync()
                ctx.timing["incumbent_pass_s"] = ctx.cache["prefix_time"][self.T] + time.perf_counter() - ts
            if self.proposal is not None and not ctx.cache.get("proposal_prepared", False):
                _api.cuda_sync()
                ts = time.perf_counter()
                self.proposal.prepare_call(ctx)
                _api.cuda_sync()
                ctx.timing["proposal_prepare_s"] = time.perf_counter() - ts
                ctx.cache["proposal_prepared"] = True

    # ------------------------------------------------------------------
    def provisional_contracts(self, ctx: _api.CallContext, l: int, Z, r_prov, spec: PolicySpec):
        cfg = self.cfg
        seed_r = cfg.provisional_seed_frac * spec.H_row / self.gamma if math.isfinite(spec.H_row) \
            else cfg.provisional_seed_frac * 1.0 / self.gamma
        rho = ctx.op.mm(r_prov + seed_r)
        if spec.factor_mode == "global":
            L = torch.full((ctx.n,), self.w_norm, dtype=torch.float32, device=Z.device)
            mu = torch.full((ctx.n,), self.w_norm ** 2 * _api.TANH2_PEAK if self.inc.activation == "tanh" else 0.0,
                            dtype=torch.float32, device=Z.device)
        else:
            L, mu, _ = _api.interval_factors(Z, rho, self.split_prov, self.inc.activation, torch.float32)
        infl = float(spec.provisional_inflation) * float(cfg.provisional_inflation)
        if infl != 1.0:
            L = torch.clamp(L * infl, max=self.w_norm)
            mu = mu * infl
        lam, xi = _api.contracts_backward(ctx.op, self.T, l + 1, lambda j: L, lambda j: mu,
                                     lambda j: ctx.keep(j), ctx.w, self.gamma, self.inc.alpha, self.inc.tau,
                                     store=[l + 1])
        return L, mu, lam[l + 1], xi[l + 1]

    def row_allowance(self, l: int, r_prov_rows, n_left: int, spec: PolicySpec):
        if not math.isfinite(spec.H_row):
            return torch.full_like(r_prov_rows, float("inf"))
        g = self.growth_for()
        if g is not None:
            Pi_l, Pi_n = float(g[l]), float(g[l + 1])
        else:
            Pi_l, Pi_n = float(self.reg.get("growth_default", {}).get(str(l), 1.0)), \
                float(self.reg.get("growth_default", {}).get(str(l + 1), 1.0))
        Pi_n = max(Pi_n, 1e-6)
        num = torch.clamp(spec.H_row / self.gamma - Pi_l * r_prov_rows, min=0.0)
        return num / (Pi_n * max(1, n_left))

    # ------------------------------------------------------------------
    def run(self, ctx: _api.CallContext, spec: PolicySpec, hook: Optional[Callable] = None,
            stop_after: Optional[int] = None, keep_records: bool = True,
            adversary: Optional[dict] = None, post_hook: Optional[Callable] = None,
            record_tube: bool = False) -> ExecResult:
        cfg = self.cfg
        inc = self.inc
        T = self.T
        opened = sorted(spec.open_depths if spec.open_depths is not None else self.open_all)
        start = min(opened) if opened else T
        self.prepare_call(ctx, prefix_only=True, prefix_depth=start)
        caps = self.caps_for(spec)
        heads = self.heads_for(spec)
        if spec.kind in ("rtt", "tr", "adversarial") and heads is None:
            raise _api.RTTFailure("ARTIFACT_MISSING", f"policy {spec.name} requires heads '{spec.heads_key}'")
        W = inc.weight()
        H0 = ctx.H0
        states = list(ctx.inc_states[: start + 1])
        H = states[-1]
        r_prov = torch.zeros(ctx.n, dtype=torch.float32, device=H.device)
        B = float(spec.H_plus)
        l0 = None
        steps: Dict[int, Any] = {}
        records: Dict[int, DepthRecord] = {}
        _api.cuda_sync()
        t_run0 = time.perf_counter()
        L_last = None
        n_open_total = len([l for l in opened if caps[l].sum() > 0])
        opened_seen = 0
        tube_snap: Dict[int, Any] = {}
        self._post_hook = post_hook
        self._adj_hit_time = 0.0        # adjoint computed earlier on this call by another policy: charge it here too
        self._adj_counted = False
        for l in range(start, T):
            Hbar, Z = inc.transition(H, H0, ctx.op, W, ctx, l, return_pre=True)
            v = None
            is_open = (l in opened) and (spec.kind == "raw" or caps[l].sum() > 0)
            if is_open:
                opened_seen += 1
            if is_open and (B > 1e-15 or spec.kind == "raw"):
                td = time.perf_counter()
                if spec.kind == "raw":
                    cand = self.proposal.candidates(Hbar, l, ctx)
                    cand = _api.norm_trim(cand, spec.row_trim if spec.row_trim is not None else cfg.row_trim)
                    v = cand.sum(0)
                    rec = DepthRecord(depth=l, kind="raw", step_norm=float(torch.linalg.norm(v).item()))
                    records[l] = rec
                else:
                    n_left = max(1, n_open_total - opened_seen + 1)
                    v, rec, L_used = self._open_depth(ctx, spec, heads, caps, l, H, Hbar, Z, r_prov, B, n_left,
                                                      hook, adversary)
                    L_last = L_used
                    if rec is not None:
                        B -= rec.charge_prov
                        B = max(B, 0.0)
                        rec.time_s = time.perf_counter() - td
                        if keep_records:
                            records[l] = rec
                if v is not None and l0 is None and ((not bool(torch.isfinite(v).all().item()))
                                                     or float(v.abs().max().item()) > 0):
                    l0 = l
                if stop_after is not None and l >= stop_after:
                    break
            Hn = Hbar if v is None else Hbar + v
            if v is not None:
                steps[l] = v
            # provisional tube update
            if (v is not None) or bool((r_prov > 0).any().item()):
                if spec.kind != "raw":
                    if L_last is None or not is_open:
                        rho = ctx.op.mm(r_prov)
                        if spec.factor_mode == "global":
                            L_last = torch.full((ctx.n,), self.w_norm, dtype=torch.float32, device=Hbar.device)
                        else:
                            L_last, _, _ = _api.interval_factors(Z, rho, self.split_prov, inc.activation, torch.float32,
                                                            need_curvature=False)
                    sn = None if v is None else torch.linalg.norm(v, dim=1)
                    r_prov = _api.tube_step(ctx.op, r_prov, L_last, ctx.keep(l + 1), sn, inc.alpha, inc.tau)
            L_last = None
            H = Hn
            states.append(H)
            if record_tube:
                tube_snap[l + 1] = r_prov.clone()
            if stop_after is not None and l >= stop_after:
                break
        logits_T = inc.logits(H) if len(states) == T + 1 else None
        _api.cuda_sync()
        t_run = time.perf_counter() - t_run0 + self._adj_hit_time
        res = ExecResult(policy=spec.name, kind=spec.kind, states=states, steps=steps, records=records,
                         l0=l0, logits_T=logits_T, B_left=B, time_s=t_run, start_depth=start)
        res.diag["r_prov_T_max"] = float(r_prov.max().item()) if len(states) == T + 1 else None
        if record_tube:
            res.diag["tube_snap"] = tube_snap
        self._post_hook = None
        return res

    # ------------------------------------------------------------------
    def _open_depth(self, ctx, spec, heads, caps, l, H, Hbar, Z, r_prov, B, n_left, hook, adversary):
        cfg = self.cfg
        inc = self.inc
        kind_d = cfg.depth_kind(l)
        di = self.depth_index(l)
        L, mu, lam, xi = self.provisional_contracts(ctx, l, Z, r_prov, spec)
        if kind_d == "last":
            rows = ctx.q_idx
        else:
            rows = torch.nonzero(lam > 0, as_tuple=False).squeeze(1)
        n_s = int(rows.shape[0])
        rec = DepthRecord(depth=l, kind=kind_d, B_before=B, n_support=n_s)
        if n_s > 0:
            rec.L_gm = float(torch.exp(torch.log(torch.clamp(L[rows], min=1e-30)).mean()).item())
        if n_s == 0:
            rec.status = "empty_support"
            return None, rec, L
        with torch.no_grad():
            cand_full = self.proposal.candidates(Hbar, l, ctx)
        cand = cand_full[:, rows]
        del cand_full
        if adversary is not None and "candidates" in adversary:
            cand = adversary["candidates"](ctx, l, Hbar, rows)
        trim = spec.row_trim if spec.row_trim is not None else cfg.row_trim
        omega = spec.omega if spec.omega is not None else cfg.omega
        cand_nt = _api.norm_trim(cand, trim)
        lens_full = torch.softmax(inc.logits(Hbar), dim=1)
        nbr_lens = ctx.op.mm(lens_full)
        conf = lens_full[rows].max(1).values
        roles = _api.assign_roles(ctx.scored[rows], conf, self.role_thr)
        r_rows = r_prov[rows]
        rho = self.row_allowance(l, r_rows, n_left, spec)
        cap_l = torch.from_numpy(caps[l].astype(np.float32)).to(Hbar.device)
        cap_rows = cap_l[roles]
        fmode = spec.feature_mode
        s_pre = _api.support_features(cand_nt, self.probes, fmode, scale=cfg.row_trim)
        h0_lens = ctx.cache["h0_lens"]
        xn = ctx.cache["x_norm"]
        hn = ctx.cache["h0_norm"]
        rho_full = torch.zeros(ctx.n, dtype=torch.float32, device=rho.device)
        rho_full[rows] = torch.clamp(rho, max=1e6)
        h_row_feat = spec.H_row if math.isfinite(spec.H_row) else 1.0
        feat_pre = _api.build_row_features(rows, ctx, di, self.n_open, lens_full, nbr_lens, h0_lens, lam, r_prov,
                                      rho_full, s_pre, self.gamma, h_row_feat, xn, hn)
        with torch.no_grad():
            theta = heads.node_weight(feat_pre)
        D = _api.row_budget_trim(cand_nt * theta.view(1, -1, 1), rho, cap_rows)
        if adversary is not None and "post_trim" in adversary:
            D = adversary["post_trim"](D, rows)
        s_post = _api.support_features(D, self.probes, fmode, scale=cfg.row_trim)
        feat = _api.build_row_features(rows, ctx, di, self.n_open, lens_full, nbr_lens, h0_lens, lam, r_prov, rho_full,
                                  s_post, self.gamma, h_row_feat, xn, hn)
        unit2 = (self.gamma / float(ctx.spec.nq)) ** 2
        with torch.no_grad():
            pB = heads.posterior(feat, lens_full[rows])
            # field ingredients
            is_scored_row = ctx.scored[rows]
            qpos = torch.zeros(ctx.n, dtype=torch.long, device=Hbar.device)
            E0 = torch.zeros(ctx.n, self.d, dtype=torch.float32, device=Hbar.device)
            resid_rows = (lens_full[rows] - pB) @ self.A32.t()           # [n_s, d]
            srow = rows[is_scored_row]
            E0[srow] = ctx.w[srow].unsqueeze(1) * resid_rows[is_scored_row]
            E1 = ctx.op.tmm(E0)
            E2 = ctx.op.tmm(E1)
            adj_l = None
            if spec.kind == "posterior_gate":
                g_rows = self._posterior_slope(ctx, l, Hbar, rows)
                kappa = 0.0
            elif adversary is not None and "field" in adversary:
                g_rows = adversary["field"](ctx, l, Hbar, rows)
                kappa = 0.0
            else:
                g_rows = _api.field_estimate(heads, feat, di, rows, [E0, E1, E2], resid_rows, lam[rows], self.gamma,
                                        adj_rows=None)
                kappa = float((spec.radii or {}).get(l, cfg.kappa_train))
            sig2 = heads.variance(feat, di, unit2, cfg.cov_floor_rel)
        rec.kappa = kappa
        rec.omega = omega
        if hook is not None:
            hook(dict(l=l, di=di, rows=rows, feat=feat, feat_pre=feat_pre, cand_nt=cand_nt, rho=rho,
                      cap_rows=cap_rows, roles=roles, theta=theta, D=D, lens_rows=lens_full[rows],
                      E_hops=[E0, E1, E2], lam=lam, xi=xi, Hbar=Hbar, g=g_rows, sig2=sig2, pB=pB, adj=adj_l))
        if spec.kind == "rtt" and not math.isfinite(kappa):
            rec.status = "infinite_radius"
            return None, rec, L
        if not math.isfinite(kappa):
            kappa = 0.0          # TR / diagnostics: the robust model is not used
        # ---- block aggregation (float64) ----
        M = D.shape[0]
        K = cfg.K_roles
        D64 = D.to(torch.float64)
        Gi = torch.einsum("mnd,pnd->nmp", D64, D64)
        ai = torch.einsum("mnd,nd->nm", D64, g_rows.to(torch.float64))
        xi_rows = xi[rows].to(torch.float64)
        sig_rows = sig2.to(torch.float64)
        a_t = torch.zeros(K, M, dtype=torch.float64, device=D.device).index_add(0, roles, ai)
        Gt = torch.zeros(K, M, M, dtype=torch.float64, device=D.device).index_add(0, roles, Gi)
        St = torch.zeros(K, M, M, dtype=torch.float64, device=D.device).index_add(0, roles, sig_rows.view(-1, 1, 1) * Gi)
        MX = torch.zeros(K, M, M, dtype=torch.float64, device=D.device).index_add(0, roles, xi_rows.view(-1, 1, 1) * Gi)
        reg = cfg.cov_reg_rel * unit2
        rec.reg = reg
        St = St + reg * Gt
        a_np = a_t.cpu().numpy()
        Gt_np = Gt.cpu().numpy()
        St_np = St.cpu().numpy()
        MX_np = MX.cpu().numpy()
        roles_present = torch.unique(roles).cpu().numpy().tolist() if hasattr(torch, "unique") else sorted(set(roles.cpu().numpy().tolist()))
        active = []
        for k in roles_present:
            if caps[l, k] <= 0 or np.trace(Gt_np[k]) <= 0:
                continue
            if kind_d == "contract" and spec.screen and np.all(a_np[k] >= 0):
                continue
            active.append(int(k))
        rec.n_active_blocks = len(active)
        if not active:
            rec.status = "no_active_block"
            return None, rec, L
        nv = M * len(active)
        blocks = [(i * M, (i + 1) * M) for i in range(len(active))]
        a_vec = np.concatenate([a_np[k] for k in active])
        S_mat = np.zeros((nv, nv))
        Q_mat = np.zeros((nv, nv))
        MXv = np.zeros((nv, nv))
        Gv = np.zeros((nv, nv))
        for i, k in enumerate(active):
            s0, s1 = blocks[i]
            S_mat[s0:s1, s0:s1] = St_np[k]
            MXv[s0:s1, s0:s1] = MX_np[k]
            Gv[s0:s1, s0:s1] = Gt_np[k]
        Q_mat = ((1.0 + cfg.rho_D) * MXv + cfg.rho_x * Gv) / omega
        caps_act = [float(caps[l, k]) for k in active]
        base = dict(nv=nv, blocks=blocks, caps=caps_act, a_t=a_vec, S_t=S_mat, Q_t=Q_mat, kappa=kappa, B=B,
                    omega=omega)
        # direction tensors for tangents / last-depth logit maps
        results: List[_api.SolveResult] = []
        heavy = None
        gauge = None

        def get_gauge():
            As = []
            for k in active:
                sel = roles == k
                Ak = _api.build_block_gauge(Gi[sel], lam[rows][sel].to(torch.float64), a_np[k], cfg.gauge_angles)
                if Ak is None:
                    return None
                As.append(Ak)
            return As

        def beta_for(Bv):
            b = math.log(max(2, self.C)) / max(cfg.smoothing_rel * Bv, 1e-300)
            return float(min(max(b, cfg.smoothing_beta_min), cfg.smoothing_beta_max))

        if spec.kind == "tr":
            t_star, charge, br = self._trust_rule_step(ctx, spec, l, kind_d, rows, roles, D, active, blocks, caps_act,
                                                       feat, lam, xi, Hbar, B, Gi, a_np)
            res = _api.SolveResult(t=t_star, f=float("nan"), charge=charge, status="trust_rule", branch=br)
        elif kind_d == "last":
            U = self._last_depth_U(ctx, rows, roles, D, active)
            z = inc.logits(Hbar)[ctx.q_idx].to(torch.float64)
            p = torch.softmax(z, dim=1)
            heavy = _api.HeavyTerms(U, ctx.w64[ctx.q_idx], p=p, z=z, beta=beta_for(B))
            Xi_T = (self.gamma ** 2 / 4.0) * ctx.w64
            MXT = torch.zeros(K, M, M, dtype=torch.float64, device=D.device).index_add(
                0, roles, Xi_T[rows].view(-1, 1, 1) * Gi).cpu().numpy()
            Qr = np.zeros((nv, nv))
            for i, k in enumerate(active):
                s0, s1 = blocks[i]
                Qr[s0:s1, s0:s1] = cfg.rho_D * MXT[k] + cfg.rho_x * Gt_np[k]
            prob = dict(base, branch="cplus", heavy=heavy, last=True, Qr=Qr)
            res = _api.solve_block_program(prob, cfg)
            if spec.pricing == "symmetric" and res.t.any():
                res = self._symmetric_rescale_last(heavy, res, B)
        else:
            cand_res = []
            if kind_d == "contract" or spec.window_mode in ("two_branch", "sigma_only"):
                gauge = get_gauge()
                if gauge is not None:
                    r1 = _api.solve_block_program(dict(base, branch="gauge", gauge=gauge), cfg)
                    cand_res.append(r1)
            if kind_d == "window" and spec.window_mode in ("two_branch", "env_only"):
                U, p_ref = self._window_U(ctx, l, Hbar, rows, roles, D, active)
                heavy = _api.HeavyTerms(U, ctx.w64[ctx.q_idx], p=p_ref, beta=beta_for(B))
                prob = dict(base, branch="env", heavy=heavy, Qe=MXv)
                r2 = _api.solve_block_program(prob, cfg)
                if spec.pricing == "symmetric" and r2.t.any():
                    r2 = self._symmetric_rescale_env(heavy, r2, MXv, B)
                cand_res.append(r2)
            if not cand_res:
                rec.status = "no_branch"
                return None, rec, L
            res = self._choose_branch(cand_res, D, roles, active, blocks, cfg)
            # record the smaller valid charge of the chosen step (Lemma 4)
            if kind_d == "window" and res.t.any():
                chs = [res.charge]
                if gauge is not None:
                    chs.append(sum(float(np.max(gauge[i] @ res.t[s0:s1])) for i, (s0, s1) in enumerate(blocks)))
                if heavy is not None:
                    chs.append(heavy.exact_env_linear(res.t) + 0.5 * float(res.t @ MXv @ res.t))
                res.charge = float(min(chs))
        rec.status = res.status
        rec.branch = res.branch
        rec.newton = res.newton_iters
        rec.phi_prov = float(res.f) if res.f == res.f else 0.0
        t_full = np.zeros((K, M))
        for i, k in enumerate(active):
            s0, s1 = blocks[i]
            t_full[k] = res.t[s0:s1]
        rec.t = t_full
        if not np.any(res.t > 0):
            rec.charge_prov = 0.0
            return None, rec, L
        t_rows = torch.from_numpy(t_full.astype(np.float32)).to(D.device)[roles]   # [n_s, M]
        v_rows = torch.einsum("nm,mnd->nd", t_rows, D)
        v = torch.zeros(ctx.n, self.d, dtype=torch.float32, device=D.device)
        v[rows] = v_rows
        rec.charge_prov = float(max(res.charge, 0.0))
        rec.step_norm = float(torch.linalg.norm(v_rows).item())
        rec.rows = rows
        rec.g = g_rows.to(torch.float32)
        rec.sig2 = sig2.to(torch.float32)
        if getattr(self, "_post_hook", None) is not None:
            self._post_hook(dict(l=l, rows=rows, roles=roles, D=D, t=t_full, lam=lam, v_rows=v_rows,
                                 charge=rec.charge_prov, kind=kind_d, Hbar=Hbar, active=active))
        return v, rec, L

    # ------------------------------------------------------------------
    def _choose_branch(self, cands: List[_api.SolveResult], D, roles, active, blocks, cfg) -> _api.SolveResult:
        if len(cands) == 1:
            return cands[0]
        best = min(cands, key=lambda r: r.f)
        ties = [r for r in cands if abs(r.f - best.f) <= cfg.tie_tol * (1.0 + abs(best.f))]
        if len(ties) == 1:
            return best

        def phys_norm(r):
            t_rows = torch.from_numpy(np.concatenate([r.t[s0:s1] for (s0, s1) in blocks]).reshape(len(active), -1)
                                      .astype(np.float64)).to(D.device)
            tr = torch.zeros(self.cfg.K_roles, D.shape[0], dtype=torch.float64, device=D.device)
            for i, k in enumerate(active):
                tr[k] = t_rows[i]
            vr = torch.einsum("nm,mnd->nd", tr[roles], D.to(torch.float64))
            return float((vr * vr).sum().item())
        ties.sort(key=lambda r: (phys_norm(r), r.branch))
        return ties[0]

    def _window_U(self, ctx, l, Hbar, rows, roles, D, active):
        """Tangent basis U_l: first-order terminal-logit change of every coefficient,
        pushed through the reference tail from H-bar_{l+1}; and p_{l,i}."""
        inc = self.inc
        states, derivs = _api.tail_forward(inc, ctx, Hbar, l + 1)
        zT = inc.logits(states[-1])[ctx.q_idx]
        p_ref = torch.softmax(zT.to(torch.float64), dim=1)
        M = D.shape[0]
        X = torch.zeros(len(active) * M, ctx.n, self.d, dtype=torch.float32, device=D.device)
        for i, k in enumerate(active):
            sel = roles == k
            r_k = rows[sel]
            for m in range(M):
                X[i * M + m, r_k] = D[m, sel]
        XT = _api.tail_jvp(inc, ctx, derivs, l + 1, X)
        U = XT[:, ctx.q_idx] @ self.A32                                   # [nv, nq, C]
        return U.permute(1, 2, 0).contiguous().to(torch.float64), p_ref

    def _last_depth_U(self, ctx, rows, roles, D, active):
        M = D.shape[0]
        nq = rows.shape[0]
        U = torch.zeros(nq, self.C, len(active) * M, dtype=torch.float64, device=D.device)
        for i, k in enumerate(active):
            sel = roles == k
            for m in range(M):
                U[sel, :, i * M + m] = (D[m, sel] @ self.A32).to(torch.float64)
        return U

    def _symmetric_rescale_env(self, heavy: _api.HeavyTerms, res: _api.SolveResult, Qe, B) -> _api.SolveResult:
        """Symmetric pricing: enforce the mirrored envelope as well by shrinking the step."""
        Um = heavy.U
        mirrored = _api.HeavyTerms(-Um, heavy.w, p=heavy.p, beta=heavy.beta)
        t = res.t.copy()
        for _ in range(60):
            v2 = mirrored.exact_env_linear(t) + 0.5 * float(t @ Qe @ t)
            v1 = heavy.exact_env_linear(t) + 0.5 * float(t @ Qe @ t)
            if max(v1, v2) <= B:
                break
            t *= 0.8
        res.t = t
        res.charge = max(heavy.exact_env_linear(t), mirrored.exact_env_linear(t)) + 0.5 * float(t @ Qe @ t)
        return res

    def _symmetric_rescale_last(self, heavy: _api.HeavyTerms, res: _api.SolveResult, B) -> _api.SolveResult:
        t = res.t.copy()

        def osc(tt):
            y = heavy._y(tt)
            return float(torch.dot(heavy.w, y.max(1).values - y.min(1).values).item())
        for _ in range(60):
            if osc(t) <= B:
                break
            t *= 0.8
        res.t = t
        res.charge = osc(t)
        return res

    def _posterior_slope(self, ctx, l, Hbar, rows):
        """Diagnostic 'posterior-slope gate': the expected full-loss slope under the
        cross-fitted posterior, via the exact reference tail (uses labels offline)."""
        post = ctx.cache.get("crossfit_posterior")
        if post is None:
            raise _api.RTTFailure("ARTIFACT_MISSING", "posterior-slope gate needs the cross-fitted posterior")
        states, derivs = _api.tail_forward(self.inc, ctx, Hbar, l + 1)
        G_T = _api.terminal_ce_grad(self.inc, ctx, states[-1], soft_targets=post)
        G = _api.tail_vjp(self.inc, ctx, derivs, l + 1, G_T)
        return G[rows]

    def _trust_rule_step(self, ctx, spec, l, kind_d, rows, roles, D, active, blocks, caps_act, feat, lam, xi,
                         Hbar, B, Gi, a_np):
        """Matched TR: t_km = trust_k * cap_k / M, shrunk to the remaining budget under
        the same provisional charges."""
        M = D.shape[0]
        K = self.cfg.K_roles
        bf = torch.zeros(K, feat.shape[1], device=feat.device).index_add(0, roles, feat)
        cnt = torch.bincount(roles, minlength=K).to(torch.float32)
        bf = bf / torch.clamp(cnt, min=1.0).unsqueeze(1)
        ids = torch.tensor(active, dtype=torch.long, device=feat.device)
        tr_head = self.tr_heads.get(spec.tr_key)
        if tr_head is None:
            raise _api.RTTFailure("ARTIFACT_MISSING", f"trust-rule head '{spec.tr_key}' unavailable")
        with torch.no_grad():
            tr = tr_head(bf[ids], ids, cnt[ids]).cpu().numpy().astype(np.float64)
        t = np.concatenate([np.full(M, tr[i] * caps_act[i] / M) for i in range(len(active))])
        lam_rows = lam[rows].to(torch.float64)

        def sigma_of(tt):
            tf = torch.zeros(K, M, dtype=torch.float64, device=D.device)
            for i, k in enumerate(active):
                s0, s1 = blocks[i]
                tf[k] = torch.from_numpy(tt[s0:s1]).to(D.device)
            vr = torch.einsum("nm,mnd->nd", tf[roles], D.to(torch.float64))
            return float((lam_rows * torch.linalg.norm(vr, dim=1)).sum().item())

        if kind_d == "last":
            U = self._last_depth_U(ctx, rows, roles, D, active)
            z = self.inc.logits(Hbar)[ctx.q_idx].to(torch.float64)
            heavy = _api.HeavyTerms(U, ctx.w64[ctx.q_idx], p=torch.softmax(z, 1), z=z)
            charge_fn = heavy.exact_cplus
            br = "tr_cplus"
        elif kind_d == "window":
            U, p_ref = self._window_U(ctx, l, Hbar, rows, roles, D, active)
            heavy = _api.HeavyTerms(U, ctx.w64[ctx.q_idx], p=p_ref)
            nv = len(t)
            MX = np.zeros((nv, nv))
            MXt = torch.zeros(K, M, M, dtype=torch.float64, device=D.device).index_add(
                0, roles, xi[rows].to(torch.float64).view(-1, 1, 1) * Gi).cpu().numpy()
            for i, k in enumerate(active):
                s0, s1 = blocks[i]
                MX[s0:s1, s0:s1] = MXt[k]

            def charge_fn(tt):
                return min(sigma_of(tt), heavy.exact_env_linear(tt) + 0.5 * float(tt @ MX @ tt))
            br = "tr_window"
        else:
            charge_fn = sigma_of
            br = "tr_sigma"
        c = charge_fn(t)
        if c > B:
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if charge_fn(t * mid) <= B:
                    lo = mid
                else:
                    hi = mid
            t = t * lo
            c = charge_fn(t)
        return t, float(max(c, 0.0)), br

