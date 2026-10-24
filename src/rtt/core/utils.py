"""Serialization, hashing, deterministic seeds, and timers."""
from __future__ import annotations

from ._imports import (
    Any,
    List,
    Optional,
    Path,
    Sequence,
    TORCH_AVAILABLE,
    _dt,
    contextlib,
    csv,
    dataclasses,
    hashlib,
    io,
    json,
    math,
    np,
    os,
    random,
    time,
    torch,
)

# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


_LOG_T0 = time.time()


def log(*parts: Any) -> None:
    msg = " ".join(str(p) for p in parts)
    elapsed = time.time() - _LOG_T0
    print(f"[{_dt.datetime.now().strftime('%H:%M:%S')} +{elapsed:8.1f}s] {msg}", flush=True)


def warn(*parts: Any) -> None:
    log("[WARN]", *parts)


def json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if TORCH_AVAILABLE and isinstance(o, torch.Tensor):
        return o.detach().cpu().numpy().tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def to_json(obj: Any, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent, sort_keys=True, default=json_default)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        with contextlib.suppress(Exception):
            os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(Path(path), text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(Path(path), to_json(obj) + "\n")


def read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_csv(path: Path, rows: Sequence[dict], fields: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        keys: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fields = keys
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(fields), extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: _csv_cell(r.get(k)) for k in fields})
    atomic_write_text(path, buf.getvalue())


def _csv_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        v = float(v)          # np.float64 subclasses float; repr() would print 'np.float64(...)'
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, default=json_default)
    return v


def read_csv(path: Path) -> List[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(to_json(obj, indent=None).encode())


def stable_seed(*parts: Any) -> int:
    """Deterministic 63-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def np_rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


class Timer:
    def __init__(self, sync: bool = True):
        self.sync = sync
        self.t = 0.0

    def __enter__(self):
        if self.sync:
            cuda_sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync:
            cuda_sync()
        self.t = time.perf_counter() - self._t0
        return False


def cuda_sync() -> None:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()


def finite_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return str(float(x))
        return f"{float(x):.{nd}g}"
    return str(x)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


