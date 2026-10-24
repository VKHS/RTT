"""Deterministic experiment task manifests."""
from __future__ import annotations

from ._imports import (
    List,
    Optional,
    Path,
    Tuple,
)
from . import api as _api

# ----------------------------------------------------------------------------
# task lists
# ----------------------------------------------------------------------------

def _shards(n: int, per: int) -> List[Tuple[int, int]]:
    out = []
    s = 0
    while s < n:
        out.append((s, min(n, s + per)))
        s += per
    return out


def list_tasks(stage: str, cfg: _api.RTTConfig, suite: Optional[_api.Suite] = None, opts: Optional[dict] = None) -> List[dict]:
    opts = opts or {}
    breadth = bool(opts.get("breadth", True))
    T = []
    if stage == "incumbent":
        T = [{"task": "main", "seed": cfg.seed}, {"task": "v2", "seed": cfg.seed + 1}]
    elif stage == "proposal":
        T = [{"task": "native"}]
        if breadth:
            T += [{"task": "lora16"}, {"task": "gprgnn"}]
    elif stage == "phase2":
        T = [{"task": "heads_native"}, {"task": "correctors_native"}, {"task": "crossfit"}]
        T += [{"task": v} for v in _api.FT_VARIANTS]          # 'ft_same_labels', 'ft_kl', 'ft_dinf'
        if opts.get("swap_heads", True):
            T += [{"task": "heads_mean_native"}, {"task": "heads_probe_native"}]
        if breadth:
            T += [{"task": f"heads_{p}"} for p in _api.EXTERNAL_PROPOSALS]
            T += [{"task": f"correctors_{p}"} for p in _api.EXTERNAL_PROPOSALS]
    elif stage == "placement":
        T = [{"task": "native"}] + ([{"task": p} for p in _api.EXTERNAL_PROPOSALS] if breadth else [])
    elif stage == "calibration":
        T = [{"task": f"native__{m.name}"} for m in _api.default_library()]
        T += [{"task": f"native__{s}"} for s in _api.SWAPS]
        if breadth:
            T += [{"task": f"{p}__m4_25zeta"} for p in _api.EXTERNAL_PROPOSALS]
    elif stage == "selection":
        for m in _api.default_library():
            for pool in ("mixture", "clean", "cache", "noise"):
                T.append({"task": f"{m.name}__{pool}"})
        T.append({"task": "ltt_switch"})
        for nq in cfg.call_sizes:
            if nq == cfg.nq:
                continue
            for m in _api.default_library():
                if m.selectable:
                    T.append({"task": f"{m.name}__mixture__nq{nq}"})
    elif stage == "development":
        per = max(1, int(opts.get("dev_shard", 128)))
        for grp in ("budget", "callsize", "rowbudget", "robust"):
            n = int(cfg.dev_N if grp != "budget" else max(1, cfg.dev_N // 2))
            for a, b in _shards(n, per):
                T.append({"task": f"{grp}__{a:05d}_{b:05d}"})
    elif stage == "audit":
        per = max(1, int(opts.get("audit_shard", 128)))
        for a, b in _shards(cfg.audit_N_mixture, per):
            T.append({"task": f"mixture__{a:05d}_{b:05d}"})
        per_c = max(1, int(opts.get("audit_component_shard", 512)))
        for comp in _api.ALL_COMPONENTS:
            for a, b in _shards(cfg.audit_N_component, per_c):
                T.append({"task": f"{comp}__{a:05d}_{b:05d}"})
    elif stage == "breadth":
        per = max(1, int(opts.get("audit_shard", 128)))
        if breadth:
            for p in _api.EXTERNAL_PROPOSALS:
                for a, b in _shards(cfg.audit_N_mixture, per):
                    T.append({"task": f"{p}__{a:05d}_{b:05d}"})
    elif stage == "node_transfer":
        T = [{"task": "per_node"}]
        per = max(1, int(opts.get("audit_shard", 128)))
        for a, b in _shards(cfg.transfer_N, per):
            T.append({"task": f"transfer__{a:05d}_{b:05d}"})
    elif stage == "diagnostics":
        T = [{"task": "checker_planted"}, {"task": "checker_hp"}, {"task": "checker_adversarial"},
             {"task": "checker_underfactor"}, {"task": "checker_rounding_off"}, {"task": "invariance"},
             {"task": "frontier_levels"}, {"task": "depth_exact"}]
        T += [{"task": f"planted_g{int(round(g * 100)):03d}"} for g in cfg.planted_gammas]
    elif stage == "reduce":
        T = [{"task": "reduce"}]
    else:
        raise _api.RTTFailure("ORCHESTRATION", f"unknown stage {stage}")
    for i, t in enumerate(T):
        t["ordinal"] = f"{i + 1:04d}"
        t["stage"] = stage
    return T


def write_task_manifest(path: Path, tasks: List[dict]) -> None:
    lines = ["ordinal\ttask\tstage"]
    for t in tasks:
        lines.append(f"{t['ordinal']}\t{t['task']}\t{t['stage']}")
    _api.atomic_write_text(path, "\n".join(lines) + "\n")

