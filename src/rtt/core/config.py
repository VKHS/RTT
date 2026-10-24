"""Method configuration, registered members, and explicit design choices."""
from __future__ import annotations

from ._imports import (
    Dict,
    List,
    Optional,
    Path,
    Tuple,
    copy,
    dataclass,
    dataclasses,
    math,
)
from . import api as _api

# ----------------------------------------------------------------------------
# Registered configuration (Table 7 and Sections 2-4 of the manuscript)
# ----------------------------------------------------------------------------

@dataclass
class RTTConfig:
    # --- incumbent (Sec. 2.2, Table 7) ---
    T: int = 32
    alpha: float = 0.1
    tau: float = 0.9
    w_spec: float = 1.6                 # ||W||_2 of the transport (registered checkpoint property)
    hidden: int = 256                   # state width d (not specified in the manuscript)
    enc_hidden: int = 512
    enc_dropout: float = 0.2
    head_dropout: float = 0.1
    activation: str = "tanh"            # 'tanh' (main family) or 'identity' (APPNP-type linear diffusion)
    inc_epochs: int = 600
    inc_lr: float = 2e-3
    inc_wd: float = 5e-4
    inc_label_smoothing: float = 0.05
    inc_patience: int = 80
    inc_holdout_frac: float = 0.05
    inc_eval_every: int = 5
    temp_min: float = 0.05
    temp_max: float = 20.0
    use_node_year: bool = False

    # --- budgets and contract (Table 7) ---
    H_plus: float = 0.05                # per-call one-sided budget (selected member 25*zeta)
    H_row: float = 1.0                  # per-node floor, nats
    zeta: float = 0.002                 # registered regression tolerance
    p_star: float = 0.02                # regression-rate target
    delta: float = 0.05                 # statistical level
    delta_cov: float = 0.1              # nominal fixed-member trajectory miscoverage
    K_roles: int = 8
    b_window: int = 4
    M_cand: int = 2
    open_first: int = 16                # open depths open_first..T-1
    row_trim: float = 0.8               # Delta-bar
    omega: float = 0.9                  # safety factor
    rho_D: float = 1e-3                 # curvature inflation
    rho_x: float = 1e-6                 # physical regulariser
    spectral_rank: int = 8
    spectral_rank_tight: int = 32
    Pi_max: float = 20.0                # admissibility cap on contract growth
    growth_quantile: float = 0.95       # growth proxy percentile
    growth_calls: int = 64              # replay calls used for the seeded growth proxy
    n_r: int = 512                      # calibration episodes per depth
    probes_per_sign: int = 16
    bank_size: int = 16
    att_dim: int = 32
    head_hidden: int = 128
    nq: int = 4096
    kappa_train: float = 2.0            # provisional radius used before calibration
    provisional_inflation: float = 1.0  # multiplies provisional factors (tests use 0.7)
    provisional_seed_frac: float = 1.0  # cap-seeded radius = frac * H_row / Gamma
    cov_floor_rel: float = 1e-6
    cov_reg_rel: float = 1e-8

    # --- solver ---
    gauge_angles: int = 64
    solver_tol: float = 1e-11
    solver_max_newton: int = 60
    smoothing_rel: float = 0.01          # log(C)/beta <= smoothing_rel * budget
    smoothing_beta_min: float = 1e3
    smoothing_beta_max: float = 1e7
    tie_tol: float = 1e-12

    # --- numerical allowances of the checker (Appendix G) ---
    rel_inflation64: float = 1e-9
    abs_inflation64: float = 1e-15
    f32_tanh_ulps: float = 4.0
    f32_exp_ulps: float = 4.0

    # --- registered perturbation generator (Sec. 4.1) ---
    degree_quantile: float = 0.25
    degree_edge_drop: float = 0.5
    cache_fraction: float = 0.5
    noise_sigma: float = 0.5
    edge_dropout: float = 0.3
    stress_fraction: float = 0.3
    stale_year_cutoff: int = 2018        # 'real cache' = historical states on the <=2018 snapshot

    # --- phase-2 training and replay ---
    proposal_episodes: int = 3072
    head_episodes: int = 6144
    lr_heads: float = 1e-3
    lr_proposal: float = 5e-4
    wd_heads: float = 1e-5
    corrector_hops: Tuple[int, ...] = (2, 4, 8)
    ft_lr: float = 2e-4
    placement_episodes: int = 2048
    cap_max: float = 1.0
    cap_init: float = 1.0
    expected_spend_frac: float = 1.0
    crossfit_folds: int = 5
    crossfit_episodes: int = 2048

    # --- evaluation (Sec. 4, App. D) ---
    selection_pool_calls: int = 1024
    audit_N_mixture: int = 2048
    audit_N_component: int = 4096
    dev_N: int = 2048
    transfer_N: int = 2048
    reserved_nodes: int = 8192
    node_inspect: int = 4096
    min_certified_mean: float = 3.0e-3
    min_certified_mean_tol: float = 1.5e-3
    frontier_grid: int = 48
    diag_exact_depth_calls: int = 256
    checker_planted_violations: int = 10000
    call_sizes: Tuple[int, ...] = (64, 256, 1024, 4096)
    row_budgets: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, float("inf"))
    budget_sweep_multipliers: Tuple[float, ...] = (1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0)
    planted_gammas: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    planted_fraction: float = 0.3

    # --- bookkeeping ---
    seed: int = 1
    scale_name: str = "full"

    def open_depths(self) -> List[int]:
        return list(range(self.open_first, self.T))

    def window_start(self) -> int:
        return self.T - self.b_window

    def depth_kind(self, l: int) -> str:
        if l == self.T - 1:
            return "last"
        if l >= self.T - self.b_window:
            return "window"
        return "contract"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


def scaled_config(scale: str, base: Optional[RTTConfig] = None) -> RTTConfig:
    """'full' uses the manuscript's episode counts; 'pilot' shrinks every episode
    count for an end-to-end dry run on the cluster; 'preflight' is the tiny
    synthetic semantic self-test used by the prepare stage."""
    cfg = copy.deepcopy(base) if base is not None else RTTConfig()
    cfg.scale_name = scale
    if scale == "full":
        return cfg
    if scale == "pilot":
        cfg.inc_epochs = 300
        cfg.proposal_episodes = 256
        cfg.head_episodes = 512
        cfg.placement_episodes = 128
        cfg.crossfit_episodes = 256
        cfg.n_r = 64
        cfg.selection_pool_calls = 96
        cfg.audit_N_mixture = 192
        cfg.audit_N_component = 96
        cfg.dev_N = 96
        cfg.transfer_N = 96
        cfg.diag_exact_depth_calls = 16
        cfg.checker_planted_violations = 500
        return cfg
    if scale == "preflight":
        cfg.hidden = 24
        cfg.enc_hidden = 32
        cfg.inc_epochs = 40
        cfg.inc_patience = 20
        cfg.inc_eval_every = 5
        cfg.nq = 96
        cfg.bank_size = 8
        cfg.att_dim = 8
        cfg.head_hidden = 32
        cfg.probes_per_sign = 4
        cfg.spectral_rank = 4
        cfg.spectral_rank_tight = 8
        cfg.proposal_episodes = 24
        cfg.head_episodes = 32
        cfg.placement_episodes = 12
        cfg.crossfit_episodes = 16
        cfg.crossfit_folds = 2
        cfg.n_r = 24
        cfg.selection_pool_calls = 20
        cfg.audit_N_mixture = 24
        cfg.audit_N_component = 12
        cfg.dev_N = 12
        cfg.transfer_N = 12
        cfg.reserved_nodes = 192
        cfg.node_inspect = 96
        cfg.diag_exact_depth_calls = 4
        cfg.checker_planted_violations = 200
        cfg.call_sizes = (32, 96)
        cfg.frontier_grid = 24
        cfg.gauge_angles = 32
        cfg.planted_gammas = (0.0, 0.3)
        return cfg
    raise ValueError(f"unknown scale {scale}")


def config_from_args(args) -> RTTConfig:
    """Scale defaults followed by explicit, frozen JSON overrides."""
    cfg = scaled_config(args.scale)
    path = getattr(args, "config", None)
    if path:
        values = _api.read_json(Path(path))
        if not isinstance(values, dict):
            raise _api.RTTFailure("PREREQUISITE", "--config must contain one JSON object")
        allowed = {f.name for f in dataclasses.fields(RTTConfig)} - {"scale_name", "seed"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise _api.RTTFailure("PREREQUISITE", f"Unknown or reserved config keys: {unknown}")
        for name, value in values.items():
            current = getattr(cfg, name)
            if isinstance(current, tuple):
                if not isinstance(value, list):
                    raise _api.RTTFailure("PREREQUISITE", f"{name} must be a JSON list")
                value = tuple(value)
            elif isinstance(current, bool):
                if not isinstance(value, bool):
                    raise _api.RTTFailure("PREREQUISITE", f"{name} must be Boolean")
            elif isinstance(current, int):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise _api.RTTFailure("PREREQUISITE", f"{name} must be an integer")
            elif isinstance(current, float):
                if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
                    raise _api.RTTFailure("PREREQUISITE", f"{name} must be finite numeric")
                value = float(value)
            setattr(cfg, name, value)
    cfg.seed = int(args.seed)
    if not (cfg.T > 0 and 0 <= cfg.open_first < cfg.T and 1 <= cfg.b_window <= cfg.T):
        raise _api.RTTFailure("PREREQUISITE", "Require T>0, 0<=open_first<T and 1<=b_window<=T")
    if cfg.K_roles != 8 or cfg.M_cand != 2:
        raise _api.RTTFailure("PREREQUISITE", "The native role/candidate construction requires K_roles=8, M_cand=2")
    if not (0 <= cfg.alpha <= 1 and 0 <= cfg.tau <= 1 and 0 < cfg.omega < 1):
        raise _api.RTTFailure("PREREQUISITE", "Invalid diffusion coefficients or safety factor")
    if not (cfg.H_plus >= 0 and cfg.H_row > 0 and cfg.rho_x > 0 and cfg.rho_D >= 0):
        raise _api.RTTFailure("PREREQUISITE", "Invalid budget or regularizer")
    for name in ("hidden", "enc_hidden", "head_hidden", "nq", "n_r", "bank_size", "att_dim", "probes_per_sign"):
        if getattr(cfg, name) <= 0:
            raise _api.RTTFailure("PREREQUISITE", f"{name} must be positive")
    for name in ("delta", "delta_cov", "p_star"):
        if not 0 < getattr(cfg, name) < 1:
            raise _api.RTTFailure("PREREQUISITE", f"{name} must be in (0,1)")
    if cfg.activation not in ("tanh", "identity"):
        raise _api.RTTFailure("PREREQUISITE", "Only tanh and the declared linear ablation are implemented")
    return cfg


# ----------------------------------------------------------------------------
# Library of members (Appendix D, 'Selection'): ordered least -> most aggressive
# ----------------------------------------------------------------------------

@dataclass
class MemberSpec:
    name: str
    H_plus_mult: float           # H+ = mult * zeta
    radii_mode: str = "sequential"   # 'sequential' (trajectory-valid) or 'per_depth'
    delta_cov: float = 0.1
    per_depth_q: float = 0.1
    selectable: bool = True
    note: str = ""

    def H_plus(self, cfg: RTTConfig) -> float:
        return float(self.H_plus_mult) * cfg.zeta


def default_library() -> List[MemberSpec]:
    return [
        MemberSpec("m1_tolerance_point", 1.0, note="H+ = zeta: regression event ruled out deterministically"),
        MemberSpec("m2_5zeta", 5.0),
        MemberSpec("m3_10zeta", 10.0),
        MemberSpec("m4_25zeta", 25.0, note="manuscript's selected member"),
        MemberSpec("m5_50zeta", 50.0),
        MemberSpec("m6_per_depth_25zeta", 25.0, radii_mode="per_depth", selectable=False,
                   note="per-depth radii (q=0.1 per depth); Theorem 1(iii) not claimed; placed last"),
    ]


# ----------------------------------------------------------------------------
# Implementation choices where the manuscript leaves a detail unspecified.
# Emitted with every run so that the recreation is auditable.
# ----------------------------------------------------------------------------
IMPLEMENTATION_CHOICES: List[Dict[str, str]] = [{'id': 'IC01',
  'topic': 'state width and encoder',
  'manuscript': 'H0 = Enc(X); width not stated',
  'choice': 'd=256 state, 2-layer ReLU MLP encoder (512 hidden), features standardised with official-train '
            'statistics'},
 {'id': 'IC02',
  'topic': 'transport normalisation',
  'manuscript': 'P row-stochastic',
  'choice': 'P = D^-1 (A + I) on the symmetrised citation graph; W shared across depths and rescaled using '
            'float64 SVD at export to target spectral norm 1.6. The measured exported norm is logged.'},
 {'id': 'IC03',
  'topic': 'role blocks',
  'manuscript': 'K = 8 role blocks (assignment rule not stated)',
  'choice': 'label-free roles: {scored row, support row} x quartile of logit-lens confidence at H-bar_{l+1}; '
            'quartile thresholds registered on fitting calls before head training'},
 {'id': 'IC04',
  'topic': 'field head',
  'manuscript': 'field head g-hat_l on the executed support',
  'choice': 'A state-local field g_hat = sum_{h=0}^2 beta_h (P^T)^h E0 + beta_s (Lambda_provisional/Gamma) '
            'E_self. E0=w*(p_lens-posterior)*A_eff^T on scored rows; E_self is the residual on support rows. '
            'Four softplus coefficient heads read current-state/support-function features. No incumbent '
            'endpoint or reference-trajectory adjoint is a deployment input. Trained with full reference-tail '
            'slope targets.'},
 {'id': 'IC05',
  'topic': 'covariance head',
  'manuscript': 'block-diagonal covariance head Sigma-hat_x',
  'choice': 'per-row isotropic variance sigma_i^2 (softplus MLP) mapped to Sigma_x,k = Q_k^T diag(sigma^2) Q_k '
            '+ reg; trained by block Gaussian NLL of slope errors'},
 {'id': 'IC06',
  'topic': 'node-weight head',
  'manuscript': "theta_i in [0,1] scales a row's candidates",
  'choice': 'sigmoid MLP on features of the norm-trimmed hull; trained by importance-weighted BCE on the sign '
            "of the true full-loss descent of the row's candidates"},
 {'id': 'IC07',
  'topic': 'solver',
  'manuscript': 'bisection on the budget multiplier with a coupled convex solve',
  'choice': 'Primal log-barrier Newton solver over auxiliary candidate coefficients, with physical '
            'displacement objective. A polygonal majorant approximates the contract norm, and log-sum-exp '
            'upper approximations handle classwise maxima. These are conservative numerical solver '
            'approximations, not an exact implementation of the manuscript multiplier-bisection solver. Both '
            'convex window branches are solved independently; the final checker recomputes charges. Finite '
            'solver tolerance does not establish exact representation invariance.'},
 {'id': 'IC08',
  'topic': 'provisional contracts',
  'manuscript': 'provisional tube with cap-seeded provisional factors',
  'choice': 'at each open depth the rank-k interval factor is computed at the current pre-activations with '
            'radius P(r~ + H_row/Gamma) and reused for all remaining depths to form Lambda~, Xi~; the same '
            'factor updates r~'},
 {'id': 'IC09',
  'topic': 'directed rounding',
  'manuscript': 'float64 checker with directed rounding',
  'choice': 'Research float64 checker with explicit float32 residual/continuation error models and '
            'conservative numeric inflation. This is not operation-level validated directed rounding; '
            'eigensolver and tanh/JVP error assumptions are not universally proved. CERTIFICATE_STATUS '
            'accompanies outputs. Tests establish only tested-instance behavior, not universal floating-point '
            'soundness.'},
 {'id': 'IC10',
  'topic': 'perturbation generator',
  'manuscript': 'clean, degree-shift, real-cache, feature-noise, edge-dropout; severe stress rewrites 30% of '
                'rows at T/2',
  'choice': 'degree: queries from the lowest-degree quartile of the pool, incident edges dropped w.p. 0.5; '
            'cache: 50% of non-query rows replaced at every depth by historical states computed on the <=2018 '
            'snapshot; noise: N(0, 0.5^2) on standardised features; dropout: 30% of undirected edges; stress: '
            '30% of rows at depth 16 overwritten by clean depth-16 states of random other nodes'},
 {'id': 'IC11',
  'topic': 'proposal training',
  'manuscript': 'phase two trains the proposal through the executed gate',
  'choice': 'Native proposal warm-start: raw norm-trimmed update minimizes training-label CE. Heads are then '
            'fitted on trajectories generated by the current executed gate with true reference-tail slope '
            'supervision. This staged optimization is executable but is not the joint implicit-KKT training in '
            'Appendix B; no equivalence of learned weights or reported gains is claimed.'},
 {'id': 'IC12',
  'topic': 'external proposals',
  'manuscript': 'GPR-GNN, Co-GNN, AMP, GTrans, Matcha, TSA, STEM, retrained incumbent, LoRA-16',
  'choice': 'implemented: retrained incumbent (v2), LoRA-16 weight delta, GPR-GNN adapter, and a generic '
            'exported-state interface; Co-GNN, AMP, GTrans, Matcha, TSA, STEM require upstream code and are '
            'reported NOT_IMPLEMENTED'},
 {'id': 'IC13',
  'topic': 'path-length influence',
  'manuscript': 'beyond-two-hop rows via the tangent basis',
  'choice': 'influence graded by neighbour-hop count along tangent paths (self-loop steps do not increase the '
            'grade); reported as path-length grading'},
 {'id': 'IC15',
  'topic': 'posterior head',
  'manuscript': "posterior head p-hat_B estimates p^B for the field's inputs",
  'choice': 'A per-depth posterior softmax(log lens + MLP(current-state/support features)) feeds the '
            'state-local field. It is trained using fitting-label CE. The deployment path does not compute a '
            'second full incumbent trajectory.'},
 {'id': 'IC16',
  'topic': 'growth proxy Pi-hat_l',
  'manuscript': '95th percentile over rows and development calls of the ratio of terminal to depth-l '
                'provisional radius',
  'choice': 'the provisional tube with cap-seeded provisional factors along the incumbent trajectory is seeded '
            'with a unit radius at each state depth j and propagated to T; Pi-hat_j is the 95th percentile of '
            'r_T / r_j over rows and the first 64 placement-replay calls (defined even when the replay policy '
            'takes no step)'},
 {'id': 'IC17',
  'topic': 'trust rule and the radius',
  'manuscript': 'matched TR replaces the field, robust model and solver by a learned trust scale',
  'choice': 'TR steps do not depend on the calibrated radius (an infinite radius closes a depth only for the '
            'learned robust RTT step)'},
 {'id': 'IC18',
  'topic': 'certified-efficiency placement',
  'manuscript': 'caps placed by certified efficiency (replacing spend-based caps adds about +0.3)',
  'choice': 'on replayed <=2017 calls every (depth, role) cell is scored by the best certified-quadratic net '
            'gain of a scaled step along one of its candidates (full-loss slopes, Xi-curvature with the '
            '(1+rho_D)/omega inflation), divided by its contract charge sum Lambda~||d||; caps are '
            'water-filled in decreasing efficiency until the expected per-call charge reaches H+; a depth is '
            'admissible when its scalar contract growth with the 95th-percentile interval factor is <= Pi_max; '
            'trust shares pi_l are the resulting expected-charge shares'},
 {'id': 'IC14',
  'topic': 'latency and FLOPs',
  'manuscript': 'A100 80GB, 41.6 ms incumbent pass',
  'choice': 'Actual synchronized wall-clock measurements on the executing hardware; includes prefix, '
            'current-state feature preparation, proposal preparation, adapted execution, numerical checking '
            'and measured fallback. Cold environment/cache construction is outside per-call timing and must be '
            'reported separately. FLOPs are analytic estimates; A100 numbers are not assumed on GH200.'},
 {'id': 'IC19',
  'topic': 'deployment and evaluation separation',
  'manuscript': 'Accepted adapted calls do not need a separate full incumbent rollout',
  'choice': 'Native serve() prepares only the required shared prefix. Window computations use short tails from '
            'the actual current Hbar. Whole-call fallback executes the cached unmodified tail. A zero-step '
            'integrity check replays the unmodified tail. Offline evaluation separately computes the reference '
            'endpoint for metrics.'},
 {'id': 'IC20',
  'topic': 'statistics and selection scope',
  'manuscript': 'Fixed-member sequential calibration; independent final audit',
  'choice': 'Fresh depth-specific draws, fixed predictable betting prior and independent component samples. '
            'Selected-member nominal coverage is not claimed without an additional selection-valid '
            'construction. Errors, missing calls or failed selection prevent statistical release. '
            'Reserved-node predictions are committed before uniform label inspection.'}]

