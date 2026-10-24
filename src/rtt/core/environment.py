"""Sparse transport, episode generation, and call-local state."""
from __future__ import annotations

from ._imports import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    TORCH_AVAILABLE,
    contextlib,
    dataclass,
    np,
    os,
    time,
    torch,
)
from . import api as _api

# ============================================================================
# Section 3: sparse transports and the registered call generator (Sec. 4.1)
# ============================================================================

def configure_torch_numerics(deterministic: bool = True, threads: Optional[int] = None) -> dict:
    """TF32 must be off: the checker's float32 forward-error allowances assume
    IEEE float32 arithmetic in every matmul of the executed and incumbent passes."""
    info = {"torch": None}
    if not TORCH_AVAILABLE:
        return info
    info["torch"] = torch.__version__
    with contextlib.suppress(Exception):
        torch.backends.cuda.matmul.allow_tf32 = False
    with contextlib.suppress(Exception):
        torch.backends.cudnn.allow_tf32 = False
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision("highest")
    if threads:
        with contextlib.suppress(Exception):
            torch.set_num_threads(int(threads))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        with contextlib.suppress(Exception):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    info["tf32_matmul"] = bool(getattr(getattr(torch.backends, "cuda", None), "matmul", None) and
                               torch.backends.cuda.matmul.allow_tf32)
    return info


def pick_device(require_cuda: bool = False, force_cpu: bool = False):
    if force_cpu or not TORCH_AVAILABLE:
        return torch.device("cpu") if TORCH_AVAILABLE else None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise _api.RTTFailure("INFRASTRUCTURE", "CUDA required but not available", retriable=True)
    return torch.device("cpu")


class GraphOp:
    """Row-stochastic transport P = D^{-1}(A + I) of one call in CSR form.
    Rows/cols given as directed edges (both directions), without self loops."""

    def __init__(self, n: int, rows, cols, device):
        self.n = int(n)
        self.device = device
        loops = torch.arange(self.n, device=device, dtype=torch.long)
        r = torch.cat([rows.to(device=device, dtype=torch.long), loops])
        c = torch.cat([cols.to(device=device, dtype=torch.long), loops])
        key = r * self.n + c
        order = torch.argsort(key)
        r = r[order]
        c = c[order]
        deg = torch.bincount(r, minlength=self.n)
        self.deg = deg                                   # includes self loop
        vals64 = (1.0 / deg.to(torch.float64))[r]
        self.rows = r
        self.cols = c
        self.vals64 = vals64
        counts = deg
        indptr = torch.zeros(self.n + 1, dtype=torch.long, device=device)
        indptr[1:] = torch.cumsum(counts, 0)
        self.indptr = indptr
        self.P32 = torch.sparse_csr_tensor(indptr, c, vals64.to(torch.float32), size=(self.n, self.n))
        key_t = c * self.n + r
        order_t = torch.argsort(key_t)
        rt = c[order_t]
        ct = r[order_t]
        vt = vals64[order_t]
        counts_t = torch.bincount(rt, minlength=self.n)
        indptr_t = torch.zeros(self.n + 1, dtype=torch.long, device=device)
        indptr_t[1:] = torch.cumsum(counts_t, 0)
        self.col_nnz = counts_t
        self.indptr_t = indptr_t
        self.cols_t = ct
        self.vals64_t = vt
        self.PT32 = torch.sparse_csr_tensor(indptr_t, ct, vt.to(torch.float32), size=(self.n, self.n))
        self._P64 = None
        self._PT64 = None
        self.nnz = int(r.shape[0])
        self.max_row_nnz = int(deg.max().item())

    # -- float64 operators (checker) are built lazily --
    @property
    def P64(self):
        if self._P64 is None:
            self._P64 = torch.sparse_csr_tensor(self.indptr, self.cols, self.vals64, size=(self.n, self.n))
        return self._P64

    @property
    def PT64(self):
        if self._PT64 is None:
            self._PT64 = torch.sparse_csr_tensor(self.indptr_t, self.cols_t, self.vals64_t, size=(self.n, self.n))
        return self._PT64

    def _op(self, X, transpose: bool):
        if X.dtype == torch.float64:
            M = self.PT64 if transpose else self.P64
        else:
            M = self.PT32 if transpose else self.P32
        if X.dim() == 1:
            return torch.sparse.mm(M, X.unsqueeze(1)).squeeze(1)
        return torch.sparse.mm(M, X)

    def mm(self, X):
        return self._op(X, False)

    def tmm(self, X):
        return self._op(X, True)

    def neighbor_mask(self, m):
        """Boolean dilation: rows with a nonzero P entry to a row in m (incl. self)."""
        return self.mm(m.to(torch.float32)) > 0

    def edge_values_mm(self, values, X):
        """(P ⊙ E) X with per-edge values (autograd through values and X).
        values align with (self.rows, self.cols)."""
        idx = torch.stack([self.rows, self.cols], 0)
        S = torch.sparse_coo_tensor(idx, values, (self.n, self.n))
        return torch.sparse.mm(S, X)


def directed_from_undirected(u: np.ndarray, v: np.ndarray, keep: Optional[np.ndarray] = None):
    if keep is not None:
        u = u[keep]
        v = v[keep]
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    return rows, cols


@dataclass
class CallSpec:
    call_id: str
    stage: str
    pool: str
    component: str
    index: int
    nq: int
    seed: int
    query: np.ndarray
    edge_keep: Optional[np.ndarray] = None
    noise_seed: Optional[int] = None
    cache_rows: Optional[np.ndarray] = None
    stress_rows: Optional[np.ndarray] = None
    stress_src: Optional[np.ndarray] = None

    def brief(self) -> dict:
        return {"call_id": self.call_id, "stage": self.stage, "pool": self.pool,
                "component": self.component, "index": self.index, "nq": self.nq}


MIXTURE_COMPONENTS = ("clean", "degree", "cache", "noise", "dropout")
ALL_COMPONENTS = MIXTURE_COMPONENTS + ("stress",)


class CallGenerator:
    """The registered deployment generator: a call = query set of nq scored rows
    with weights 1/nq and one draw of the registered perturbation."""

    def __init__(self, g: _api.GraphData, roles: _api.DataRoles, cfg: _api.RTTConfig, suite_seed: int = 1):
        self.g = g
        self.roles = roles
        self.cfg = cfg
        self.suite_seed = int(suite_seed)
        self.deg = g.degree()

    def _rng(self, stage: str, pool: str, component: str, index: int, salt: str = "") -> np.random.Generator:
        return _api.np_rng("call", self.suite_seed, stage, pool, component, int(index), salt)

    def make(self, stage: str, pool: str, component: str, index: int, nq: Optional[int] = None) -> CallSpec:
        cfg = self.cfg
        nodes = self.roles.pools[pool]
        rng = self._rng(stage, pool, component, index)
        comp = component
        if component == "mixture":
            comp = MIXTURE_COMPONENTS[int(rng.integers(0, len(MIXTURE_COMPONENTS)))]
        k = int(min(nq or cfg.nq, len(nodes)))
        seed = int(rng.integers(0, 2 ** 62))
        spec = CallSpec(call_id=f"{stage}:{pool}:{component}:{index}", stage=stage, pool=pool,
                        component=comp, index=int(index), nq=k, seed=seed, query=np.zeros(0, np.int64))
        if comp == "degree":
            d = self.deg[nodes]
            thr = np.quantile(d, cfg.degree_quantile)
            low = nodes[d <= thr]
            if len(low) < k:
                low = nodes[np.argsort(d, kind="stable")[:max(k, 1)]]
            spec.query = np.sort(rng.choice(low, size=k, replace=False))
            inq = np.zeros(self.g.n, dtype=bool)
            inq[spec.query] = True
            incident = inq[self.g.und_u] | inq[self.g.und_v]
            drop = incident & (rng.random(self.g.num_und_edges) < cfg.degree_edge_drop)
            spec.edge_keep = ~drop
        else:
            spec.query = np.sort(rng.choice(nodes, size=k, replace=False))
        if comp == "dropout":
            spec.edge_keep = rng.random(self.g.num_und_edges) >= cfg.edge_dropout
        elif comp == "noise":
            spec.noise_seed = int(rng.integers(0, 2 ** 62))
        elif comp == "cache":
            nonq = np.setdiff1d(np.arange(self.g.n), spec.query, assume_unique=True)
            m = int(round(cfg.cache_fraction * len(nonq)))
            spec.cache_rows = np.sort(rng.choice(nonq, size=m, replace=False))
        elif comp == "stress":
            m = int(round(cfg.stress_fraction * self.g.n))
            spec.stress_rows = np.sort(rng.choice(self.g.n, size=m, replace=False))
            spec.stress_src = rng.integers(0, self.g.n, size=m)
        return spec

    def batch(self, stage: str, pool: str, component: str, indices: Iterable[int], nq: Optional[int] = None) -> List[CallSpec]:
        return [self.make(stage, pool, component, i, nq) for i in indices]


class CallContext:
    """Device-side realisation of one call.  Exogenous replacements (cache,
    stress) are part of both continuation maps (Sec. 2.1)."""

    def __init__(self):
        self.spec: Optional[CallSpec] = None
        self.op: Optional[GraphOp] = None
        self.x = None
        self.w = None
        self.w64 = None
        self.q_idx = None
        self.y_q = None
        self.mask_rows: Dict[int, Any] = {}
        self.mask_vals: Dict[int, Any] = {}
        self.mask_bool: Dict[int, Any] = {}
        self.hop = None
        self.H0 = None
        self.inc_states: Optional[List[Any]] = None
        self.inc_logits = None
        self.timing: Dict[str, float] = {}
        self.cache: Dict[str, Any] = {}
        self.n = 0
        self.T = 0

    def keep(self, j: int, dtype=None):
        """1 - mask at depth j as a float vector, or None when no row is replaced."""
        m = self.mask_bool.get(j)
        if m is None:
            return None
        k = (~m).to(dtype or torch.float32)
        return k

    def apply_mask(self, Hn, j: int, differentiable: bool = False):
        rows = self.mask_rows.get(j)
        if rows is None:
            return Hn
        vals = self.mask_vals[j].to(Hn.dtype)
        if differentiable:
            full = torch.zeros_like(Hn)
            full = full.index_put((rows,), vals)
            return torch.where(self.mask_bool[j].unsqueeze(1), full, Hn)
        out = Hn.clone()
        out[rows] = vals
        return out


class BaseEnv:
    """Task-level environment: graph on device, clean transport, historical cache
    states on the <=2018 snapshot and clean stress states (both computed once
    from the frozen incumbent)."""

    def __init__(self, g: _api.GraphData, roles: _api.DataRoles, cfg: _api.RTTConfig, device, suite_seed: int = 1):
        self.g = g
        self.roles = roles
        self.cfg = cfg
        self.device = device
        self.x = torch.from_numpy(g.x).to(device)
        self.y = torch.from_numpy(g.y).to(device)
        self.u_np = g.und_u
        self.v_np = g.und_v
        rows, cols = directed_from_undirected(g.und_u, g.und_v)
        self.clean_op = GraphOp(g.n, torch.from_numpy(rows), torch.from_numpy(cols), device)
        self.deg_np = g.degree()
        self.log_deg0 = torch.from_numpy(np.log1p(self.deg_np).astype(np.float32)).to(device)
        self.generator = CallGenerator(g, roles, cfg, suite_seed)
        self.stale_states = None
        self.clean_stress_state = None
        self.incumbent = None
        self._x_norm = torch.linalg.norm(self.x, dim=1)

    def attach_incumbent(self, inc: "TanhDiffusionIncumbent") -> None:
        self.incumbent = inc
        g = self.g
        cfg = self.cfg
        keep = (g.year[g.und_u] <= cfg.stale_year_cutoff) & (g.year[g.und_v] <= cfg.stale_year_cutoff)
        rows, cols = directed_from_undirected(g.und_u, g.und_v, keep)
        stale_op = GraphOp(g.n, torch.from_numpy(rows), torch.from_numpy(cols), self.device)
        with torch.no_grad():
            H0 = inc.encode(self.x)
            states = _api.incumbent_rollout(inc, H0, stale_op, masks=None)
            self.stale_states = torch.stack([s.to(torch.float16) for s in states], 0)
            clean_states = _api.incumbent_rollout(inc, H0, self.clean_op, masks=None)
            sd = min(max(1, cfg.T // 2), cfg.T)
            self.clean_stress_state = clean_states[sd].clone()
            self.clean_H0 = H0
            self.clean_terminal = clean_states[-1].clone()
        del states, clean_states

    def build(self, spec: CallSpec, need_labels: bool = True) -> CallContext:
        cfg = self.cfg
        dev = self.device
        ctx = CallContext()
        ctx.spec = spec
        ctx.n = self.g.n
        ctx.T = cfg.T
        t0 = time.perf_counter()
        if spec.edge_keep is None:
            ctx.op = self.clean_op
        else:
            rows, cols = directed_from_undirected(self.u_np, self.v_np, spec.edge_keep)
            ctx.op = GraphOp(self.g.n, torch.from_numpy(rows), torch.from_numpy(cols), dev)
        if spec.noise_seed is not None:
            nrng = np.random.default_rng(spec.noise_seed)
            noise = nrng.standard_normal(size=self.g.x.shape).astype(np.float32) * np.float32(cfg.noise_sigma)
            ctx.x = self.x + torch.from_numpy(noise).to(dev)
        else:
            ctx.x = self.x
        q = torch.from_numpy(spec.query.astype(np.int64)).to(dev)
        ctx.q_idx = q
        w = torch.zeros(self.g.n, dtype=torch.float32, device=dev)
        w[q] = 1.0 / float(spec.nq)
        ctx.w = w
        w64 = torch.zeros(self.g.n, dtype=torch.float64, device=dev)
        w64[q] = 1.0 / float(spec.nq)
        ctx.w64 = w64
        if need_labels:
            ctx.y_q = self.y[q]
        if spec.cache_rows is not None:
            if self.stale_states is None:
                raise _api.RTTFailure("ORCHESTRATION", "cache call requires stale states (attach_incumbent)")
            rows_t = torch.from_numpy(spec.cache_rows.astype(np.int64)).to(dev)
            mb = torch.zeros(self.g.n, dtype=torch.bool, device=dev)
            mb[rows_t] = True
            for j in range(1, cfg.T + 1):
                ctx.mask_rows[j] = rows_t
                ctx.mask_vals[j] = self.stale_states[j][rows_t].to(torch.float32)
                ctx.mask_bool[j] = mb
        if spec.stress_rows is not None:
            if self.clean_stress_state is None:
                raise _api.RTTFailure("ORCHESTRATION", "stress call requires clean stress state")
            sd = min(max(1, cfg.T // 2), cfg.T)
            rows_t = torch.from_numpy(spec.stress_rows.astype(np.int64)).to(dev)
            src_t = torch.from_numpy(spec.stress_src.astype(np.int64)).to(dev)
            mb = torch.zeros(self.g.n, dtype=torch.bool, device=dev)
            mb[rows_t] = True
            ctx.mask_rows[sd] = rows_t
            ctx.mask_vals[sd] = self.clean_stress_state[src_t].to(torch.float32)
            ctx.mask_bool[sd] = mb
        # hop classes to the scored set: 0 scored, 1, 2, 3 (>=3)
        m0 = torch.zeros(self.g.n, dtype=torch.bool, device=dev)
        m0[q] = True
        m1 = ctx.op.neighbor_mask(m0)
        m2 = ctx.op.neighbor_mask(m1)
        hop = torch.full((self.g.n,), 3, dtype=torch.long, device=dev)
        hop[m2] = 2
        hop[m1] = 1
        hop[m0] = 0
        ctx.hop = hop
        ctx.scored = m0
        ctx.log_deg_call = torch.log(ctx.op.deg.to(torch.float32))
        ctx.env_log_deg0 = self.log_deg0
        ctx.timing["build_s"] = time.perf_counter() - t0
        return ctx

