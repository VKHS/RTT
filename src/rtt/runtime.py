"""Small reproducibility and atomic-artifact utilities."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch


NUMERICAL_STATUS = "research_float64_output_checks_not_formally_validated_arithmetic"


def progress(stage, completed, total, **fields):
    """Flush bounded progress messages so long-running batch jobs are observable."""
    interval = max(1, total // 10)
    if completed not in {1, total} and completed % interval:
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[RTT {timestamp}] {stage} {completed}/{total} {details}".rstrip(), file=sys.stderr, flush=True)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else ("Infinity" if value > 0 else "-Infinity" if value < 0 else None)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2 ** 31)


def configure(seed=1, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Some graph index_add kernels lack a deterministic CUDA implementation.
    # The endpoint checker independently replays outputs; backend assumptions
    # and hardware are recorded, without claiming bitwise GPU determinism.
    return environment()


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": torch.__version__, "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tf32": False, "numerical_status": NUMERICAL_STATUS}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        synchronize(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *args):
        synchronize(self.device)
        self.seconds = time.perf_counter() - self.started


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)
    write_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": sha256(path)})


def load_checkpoint(path):
    path = Path(path)
    expected = read_json(path.with_suffix(path.suffix + ".sha256.json"))["sha256"]
    if sha256(path) != expected:
        raise ValueError(f"Checkpoint content changed: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)

