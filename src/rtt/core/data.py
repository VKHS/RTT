"""Graph representation, native arxiv loading, and data-role allocation."""
from __future__ import annotations

from ._imports import (
    Dict,
    List,
    Optional,
    Path,
    TORCH_AVAILABLE,
    Tuple,
    argparse,
    dataclass,
    hashlib,
    np,
    os,
    torch,
)
from . import api as _api

# ============================================================================
# Section 2: data, synthetic graphs and year-based data roles (Appendix D)
# ============================================================================

@dataclass
class GraphData:
    name: str
    x: np.ndarray                 # float32 [n, F], standardised
    y: np.ndarray                 # int64 [n]
    year: np.ndarray              # int64 [n]
    und_u: np.ndarray             # int64 [E], undirected unique edges u < v, no self loops
    und_v: np.ndarray
    num_classes: int
    split: Dict[str, np.ndarray]  # official train / valid / test indices
    source: str = ""
    feature_note: str = ""

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_und_edges(self) -> int:
        return int(self.und_u.shape[0])

    def degree(self) -> np.ndarray:
        deg = np.bincount(self.und_u, minlength=self.n) + np.bincount(self.und_v, minlength=self.n)
        return deg.astype(np.int64)

    def fingerprint(self) -> dict:
        h = hashlib.sha256()
        for key, value in [("x", self.x), ("y", self.y), ("year", self.year),
                           ("und_u", self.und_u), ("und_v", self.und_v)] + [
                ("split_" + k, self.split[k]) for k in sorted(self.split)]:
            arr = np.ascontiguousarray(value)
            h.update(key.encode("ascii"))
            h.update(str(arr.shape).encode("ascii"))
            h.update(arr.dtype.str.encode("ascii"))
            h.update(memoryview(arr).cast("B"))
        return {"name": self.name, "n": self.n, "F": self.num_features,
                "E_und": self.num_und_edges, "C": self.num_classes,
                "sha256": h.hexdigest(), "source": self.source,
                "feature_note": self.feature_note}


def _undirected_unique(src: np.ndarray, dst: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    a = np.minimum(src[keep], dst[keep])
    b = np.maximum(src[keep], dst[keep])
    key = a * np.int64(n) + b
    key = np.unique(key)
    return (key // n).astype(np.int64), (key % n).astype(np.int64)


def _standardise(x: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, dict]:
    x = x.astype(np.float64)
    mu = x[train_idx].mean(axis=0)
    sd = x[train_idx].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    out = ((x - mu) / sd).astype(np.float32)
    return out, {"mean_abs_mu": float(np.abs(mu).mean()), "mean_sd": float(sd.mean())}


def _year_rbf(year: np.ndarray, bins: int = 16, lo: int = 1990, hi: int = 2020) -> np.ndarray:
    centers = np.linspace(lo, hi, bins)
    width = (hi - lo) / max(1, bins - 1)
    z = (year[:, None].astype(np.float64) - centers[None, :]) / width
    return np.exp(-0.5 * z * z).astype(np.float32)


def _load_ogb_raw(dataset_root: str) -> dict:
    """Load ogbn-arxiv with the framework-free OGB loader (no PyG needed)."""
    try:
        from ogb.nodeproppred import NodePropPredDataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise _api.RTTFailure("PREREQUISITE", f"ogb package unavailable: {e}")
    ds = NodePropPredDataset(name="ogbn-arxiv", root=dataset_root)
    graph, label = ds[0]
    split = ds.get_idx_split()
    return {
        "x": np.asarray(graph["node_feat"], dtype=np.float32),
        "edge_index": np.asarray(graph["edge_index"], dtype=np.int64),
        "y": np.asarray(label, dtype=np.int64).reshape(-1),
        "year": np.asarray(graph["node_year"], dtype=np.int64).reshape(-1),
        "split": {k: np.asarray(v, dtype=np.int64) for k, v in split.items()},
        "num_classes": int(np.asarray(label).max()) + 1,
    }


def _load_legacy_graph_pt(path: str) -> dict:
    if not TORCH_AVAILABLE:
        raise _api.RTTFailure("PREREQUISITE", "torch needed to read legacy graph.pt")
    o = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(o, dict):
        raise _api.RTTFailure("PREREQUISITE", f"legacy graph cache {path} is not a dict")

    def arr(key_opts, dtype):
        for k in key_opts:
            if k in o and o[k] is not None:
                v = o[k]
                if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                return np.asarray(v).astype(dtype)
        return None

    x = arr(["x", "node_feat"], np.float32)
    ei = arr(["edge_index"], np.int64)
    y = arr(["y", "label", "labels"], np.int64)
    year = arr(["node_year", "year"], np.int64)
    if x is None or ei is None or y is None or year is None:
        raise _api.RTTFailure("PREREQUISITE", f"legacy graph cache {path} lacks x/edge_index/y/node_year")
    split = None
    for k in ("split_idx", "split", "splits"):
        if k in o and isinstance(o[k], dict):
            split = {}
            for kk, vv in o[k].items():
                if TORCH_AVAILABLE and isinstance(vv, torch.Tensor):
                    vv = vv.detach().cpu().numpy()
                split[kk] = np.asarray(vv, dtype=np.int64).reshape(-1)
    y = y.reshape(-1)
    year = year.reshape(-1)
    if split is None:
        split = {"train": np.where(year <= 2017)[0], "valid": np.where(year == 2018)[0],
                 "test": np.where(year >= 2019)[0]}
    return {"x": x, "edge_index": ei, "y": y, "year": year, "split": split,
            "num_classes": int(y.max()) + 1}


def build_graph_from_raw(raw: dict, name: str, source: str, use_year: bool = False,
                         external_x: Optional[np.ndarray] = None,
                         feature_normalization: str = "standard") -> GraphData:
    n = raw["x"].shape[0]
    ei = raw["edge_index"]
    u, v = _undirected_unique(ei[0], ei[1], n)
    aliases = {"train": ("train", "train_idx"), "valid": ("valid", "val", "validation", "valid_idx"),
               "test": ("test", "test_idx")}
    split = {}
    for k, opts in aliases.items():
        key = next((a for a in opts if a in raw["split"]), None)
        if key is None:
            raise _api.RTTFailure("PREREQUISITE", f"graph split lacks '{k}' (keys: {sorted(raw['split'])})")
        split[k] = np.asarray(raw["split"][key], dtype=np.int64).reshape(-1)
    note = "native OGB features"
    if external_x is not None:
        if external_x.shape[0] != n:
            raise _api.RTTFailure("PREREQUISITE", f"external features have {external_x.shape[0]} rows, graph has {n}")
        x_raw = external_x.astype(np.float32)
        note = "registered external backbone features"
    else:
        x_raw = raw["x"].astype(np.float32)
    if feature_normalization == "standard":
        x, st = _standardise(x_raw, split["train"])
        note += f"; standardised on official train (mean|mu|={st['mean_abs_mu']:.3g})"
    else:
        x = x_raw.astype(np.float32)
        note += "; no normalisation"
    if use_year:
        x = np.concatenate([x, _year_rbf(raw["year"])], axis=1)
        note += "; + 16 year-RBF features"
    return GraphData(name=name, x=x, y=raw["y"].astype(np.int64), year=raw["year"].astype(np.int64),
                     und_u=u, und_v=v, num_classes=int(raw["num_classes"]), split=split,
                     source=source, feature_note=note)


def save_graph_npz(g: GraphData, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez(tmp, x=g.x, y=g.y, year=g.year, und_u=g.und_u, und_v=g.und_v,
             num_classes=np.int64(g.num_classes), split_train=g.split["train"],
             split_valid=g.split["valid"], split_test=g.split["test"],
             name=np.array(g.name), source=np.array(g.source), feature_note=np.array(g.feature_note))
    os.replace(tmp, path)


def load_graph_npz(path: Path) -> GraphData:
    z = np.load(path, allow_pickle=False)
    return GraphData(name=str(z["name"]), x=z["x"].astype(np.float32), y=z["y"].astype(np.int64),
                     year=z["year"].astype(np.int64), und_u=z["und_u"].astype(np.int64),
                     und_v=z["und_v"].astype(np.int64), num_classes=int(z["num_classes"]),
                     split={"train": z["split_train"], "valid": z["split_valid"], "test": z["split_test"]},
                     source=str(z["source"]), feature_note=str(z["feature_note"]))


def synthetic_graph(n: int = 1600, num_classes: int = 6, num_features: int = 16, seed: int = 7,
                    avg_degree: float = 8.0, homophily: float = 0.72, noise: float = 1.4) -> GraphData:
    """Temporal SBM-like citation graph used by the preflight self-test.
    Year shares mimic arxiv's temporal split: <=2017 / 2018 / 2019 / 2020."""
    rng = np.random.default_rng(seed)
    years = rng.choice([2014, 2015, 2016, 2017, 2018, 2019, 2020], size=n,
                       p=[0.12, 0.13, 0.14, 0.16, 0.17, 0.16, 0.12]).astype(np.int64)
    y = rng.integers(0, num_classes, size=n).astype(np.int64)
    mu = rng.normal(0.0, 1.0, size=(num_classes, num_features))
    drift = rng.normal(0.0, 0.15, size=(num_classes, num_features))
    x = mu[y] + (years - 2017)[:, None] * drift[y] + rng.normal(0.0, noise, size=(n, num_features))
    m = int(n * avg_degree / 2)
    src = rng.integers(0, n, size=m)
    same = rng.random(m) < homophily
    by_class = [np.where(y == c)[0] for c in range(num_classes)]
    dst = rng.integers(0, n, size=m)
    for c in range(num_classes):
        idx = np.where(same & (y[src] == c))[0]
        if len(idx):
            dst[idx] = rng.choice(by_class[c], size=len(idx))
    # a few hubs
    hubs = rng.choice(n, size=max(3, n // 200), replace=False)
    hs = rng.choice(hubs, size=m // 20)
    hd = rng.integers(0, n, size=m // 20)
    src = np.concatenate([src, hs])
    dst = np.concatenate([dst, hd])
    u, v = _undirected_unique(src, dst, n)
    split = {"train": np.where(years <= 2017)[0], "valid": np.where(years == 2018)[0],
             "test": np.where(years >= 2019)[0]}
    xs, _ = _standardise(x.astype(np.float32), split["train"])
    return GraphData(name="synthetic_temporal_sbm", x=xs, y=y, year=years, und_u=u, und_v=v,
                     num_classes=num_classes, split=split, source=f"synthetic(seed={seed})",
                     feature_note="synthetic class-conditional Gaussian features")


def load_graph(args: argparse.Namespace, allow_download: bool = False) -> GraphData:
    """Resolution order: RTT npz cache -> OGB raw loader -> legacy graph.pt."""
    if getattr(args, "dataset_name", "ogbn-arxiv") != "ogbn-arxiv":
        raise _api.RTTFailure("PREREQUISITE", "This runner implements ogbn-arxiv; other datasets need their own protocol")
    kind = getattr(args, "dataset_kind", "cached")
    if kind == "synthetic":
        return synthetic_graph(n=int(getattr(args, "synthetic_nodes", 1600)),
                               num_classes=int(getattr(args, "synthetic_classes", 6)),
                               num_features=int(getattr(args, "synthetic_features", 16)),
                               seed=int(getattr(args, "synthetic_seed", 7)))
    npz = Path(args.rtt_graph_npz)
    ext = getattr(args, "external_feature_pt", None)
    if npz.is_file() and not ext:
        g = load_graph_npz(npz)
        wants_year = bool(getattr(args, "use_node_year", False))
        wants_standard = getattr(args, "feature_normalization", "standard") == "standard"
        if g.name != "ogbn-arxiv" or wants_year != ("year-RBF" in g.feature_note) or wants_standard != ("standardised" in g.feature_note):
            raise _api.RTTFailure("PREREQUISITE", "Graph cache preprocessing differs from the requested configuration; choose a new --rtt_graph_npz")
        return g
    raw = None
    errors = []
    try:
        if allow_download or (Path(args.dataset_root) / "ogbn_arxiv").exists():
            raw = _load_ogb_raw(args.dataset_root)
            source = f"ogb:{args.dataset_root}"
    except Exception as e:
        errors.append(f"ogb loader: {e}")
    if raw is None and getattr(args, "graph_pt", None) and Path(args.graph_pt).is_file():
        try:
            raw = _load_legacy_graph_pt(args.graph_pt)
            source = f"legacy:{args.graph_pt}"
        except Exception as e:
            errors.append(f"legacy graph.pt: {e}")
    if raw is None:
        raise _api.RTTFailure("PREREQUISITE", "could not load ogbn-arxiv; " + " | ".join(errors))
    external_x = None
    if ext:
        if not TORCH_AVAILABLE:
            raise _api.RTTFailure("PREREQUISITE", "torch required to read external features")
        o = torch.load(ext, map_location="cpu", weights_only=False)
        if isinstance(o, dict):
            o = o.get("x", o.get("features", o.get("emb")))
        external_x = np.asarray(o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else o, dtype=np.float32)
    g = build_graph_from_raw(raw, name="ogbn-arxiv", source=source,
                             use_year=bool(getattr(args, "use_node_year", False)),
                             external_x=external_x,
                             feature_normalization=getattr(args, "feature_normalization", "standard"))
    if not ext:
        save_graph_npz(g, npz)
    return g


# ----------------------------------------------------------------------------
# Data roles: the four uses of labels (Appendix D)
# ----------------------------------------------------------------------------

@dataclass
class DataRoles:
    pools: Dict[str, np.ndarray]
    notes: List[str]

    def summary(self) -> dict:
        return {"sizes": {k: int(len(v)) for k, v in self.pools.items()}, "notes": self.notes}


def make_data_roles(g: GraphData, cfg: _api.RTTConfig, seed: int = 20260926) -> DataRoles:
    rng = np.random.default_rng(_api.stable_seed("roles", seed, g.name, g.n))
    notes: List[str] = []
    year = g.year
    fit = np.sort(g.split["train"])                     # labels <= 2017
    perm = rng.permutation(fit)
    n_hold = max(1, int(round(cfg.inc_holdout_frac * len(fit))))
    inc_holdout = np.sort(perm[:n_hold])
    inc_train = np.sort(perm[n_hold:])
    y18 = np.sort(g.split["valid"])                     # 2018
    p18 = rng.permutation(y18)
    n_t = int(round(0.2 * len(p18)))
    n_c = int(round(0.4 * len(p18)))
    temp18 = np.sort(p18[:n_t])
    calib18 = np.sort(p18[n_t:n_t + n_c])
    select18 = np.sort(p18[n_t + n_c:])
    test = np.sort(g.split["test"])
    y19 = test[year[test] == 2019]
    y20 = test[year[test] >= 2020]
    if len(y19) == 0 and len(y20) == 0:
        raise _api.RTTFailure("PREREQUISITE", "no test-period nodes for audit pools")
    p19 = rng.permutation(y19)
    n_res = min(int(cfg.reserved_nodes), len(p19) // 3)
    if n_res < cfg.reserved_nodes:
        notes.append(f"reserved node set reduced from {cfg.reserved_nodes} to {n_res} (2019 cohort size {len(p19)})")
    reserved = np.sort(p19[:n_res])
    rest = p19[n_res:]
    half = len(rest) // 2
    dev19 = np.sort(rest[:half])
    final19 = np.sort(rest[half:])
    pools = {
        "fit": fit, "inc_train": inc_train, "inc_holdout": inc_holdout,
        "temp18": temp18, "calib18": calib18, "select18": select18,
        "dev19": dev19, "final19": final19, "reserved19": reserved, "transfer20": np.sort(y20),
    }
    for k, v in pools.items():
        if len(v) < 1:
            raise _api.RTTFailure("PREREQUISITE", f"empty data role {k}")
    # disjointness assertions (fail closed)
    names = ["inc_train", "inc_holdout", "temp18", "calib18", "select18", "dev19", "final19", "reserved19", "transfer20"]
    seen = np.zeros(g.n, dtype=np.int64)
    for k in names:
        seen[pools[k]] += 1
    if seen.max() > 1:
        raise _api.RTTFailure("ORCHESTRATION", "data roles overlap")
    for k in ("dev19", "final19", "select18", "calib18", "transfer20"):
        if len(pools[k]) < cfg.nq:
            notes.append(f"pool {k} has {len(pools[k])} < nq={cfg.nq} nodes; calls use the whole pool")
    return DataRoles(pools=pools, notes=notes)

