"""Stage completeness gates and fixed-sequence selection."""
from __future__ import annotations

from ._imports import (
    List,
    Optional,
    Path,
    Tuple,
    contextlib,
    np,
)
from . import api as _api

# ============================================================================
# Section 18: gates (fail-closed stage validation, fixed-sequence selection)
#             and the reduce stage (tables, figures, ledgers)
# ============================================================================

REQUIRED_TASKS = {
    "incumbent": {"main"},
    "proposal": {"native"},
    "phase2": {"heads_native", "correctors_native"},
    "placement": {"native"},
    "calibration": {f"native__{m.name}" for m in _api.default_library() if m.radii_mode == "sequential"},
    "selection": {f"{m.name}__{p}" for m in _api.default_library() for p in ("mixture", "clean", "cache", "noise")},
    "development": set(),
    "audit": set(),     # filled dynamically: all mixture shards
    "breadth": set(),
    "node_transfer": set(),
    "diagnostics": set(),
    "reduce": {"reduce"},
}


def task_status(run_dir: Path) -> Tuple[str, dict]:
    sp = run_dir / "summary.json"
    if sp.is_file():
        try:
            s = _api.read_json(sp)
            return str(s.get("status", "UNKNOWN")), s
        except Exception:
            return "UNPARSEABLE", {}
    fp = run_dir / "failure_summary.json"
    if fp.is_file():
        try:
            f = _api.read_json(fp)
            return f"FAILED_{str(f.get('failure_class', 'SCIENTIFIC')).upper()}", f
        except Exception:
            return "FAILED_UNPARSEABLE", {}
    return "MISSING", {}


def run_gate(suite: _api.Suite, stage: str, cfg: _api.RTTConfig) -> int:
    tasks = read_task_manifest(suite.manifest / f"tasks_{stage}.tsv")
    gdir = suite.stage_dir(stage) / "GATE"
    gdir.mkdir(parents=True, exist_ok=True)
    required = set(REQUIRED_TASKS.get(stage, set()))
    if stage == "audit":
        required = {t["task"] for t in tasks if t["task"].startswith("mixture__")}
    statuses = {}
    retry, fatal, optional_failed = [], [], []
    for t in tasks:
        rd = suite.run_dir(stage, f"{t['ordinal']}_{t['task']}")
        st, info = task_status(rd)
        statuses[t["task"]] = st
        if st == "COMPLETE":
            continue
        retriable = st in ("MISSING", "FAILED_INFRASTRUCTURE", "UNPARSEABLE", "FAILED_UNPARSEABLE") or bool(info.get("retriable"))
        if retriable:
            retry.append(t)
        elif t["task"] in required:
            fatal.append({"task": t["task"], "status": st, "detail": info.get("message", "")})
        else:
            optional_failed.append({"task": t["task"], "status": st, "detail": info.get("message", "")})
    _api.write_task_manifest(gdir / "missing_tasks.tsv", retry)
    gate = {"stage": stage, "version": _api.VERSION, "statuses": statuses, "retry": [t["task"] for t in retry],
            "fatal": fatal, "optional_failed": optional_failed, "utc": _api.now_utc()}
    code = 0
    if fatal:
        gate["decision"] = "FAIL"
        code = 20
    elif retry:
        gate["decision"] = "RETRY"
        code = 10
    else:
        gate["decision"] = "PASS" if not optional_failed else "PASS_WITH_OPTIONAL_FAILURES"
        if stage == "selection":
            gate["selection"] = apply_selection(suite, cfg)
    _api.write_json(gdir / "gate.json", gate)
    for f in ("PASS", "FAIL", "RETRY"):
        with contextlib.suppress(FileNotFoundError):
            (gdir / f).unlink()
    (gdir / gate["decision"].split("_")[0]).write_text(gate["decision"] + "\n")
    lines = [f"RTT {_api.VERSION_TAG} gate: {stage}", f"decision: {gate['decision']}"]
    lines += [f"  {k}: {v}" for k, v in statuses.items()]
    if fatal:
        lines += ["fatal:"] + [f"  {x}" for x in fatal]
    if optional_failed:
        lines += ["optional failures:"] + [f"  {x}" for x in optional_failed]
    _api.atomic_write_text(gdir / "gate.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return code


def read_task_manifest(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(dict(zip(hdr, line.split("\t"))))
    return rows


def stage_records(suite: _api.Suite, stage: str, prefix: Optional[str] = None) -> List[dict]:
    out = []
    base = suite.stage_dir(stage) / "runs"
    if not base.is_dir():
        return out
    for rd in sorted(base.glob("*/RESULT")):
        name = rd.parent.name.split("_", 1)[1] if "_" in rd.parent.name else rd.parent.name
        if prefix is not None and not name.startswith(prefix):
            continue
        st, _ = task_status(rd)
        if st != "COMPLETE":
            continue
        for p in [rd / "records.jsonl.gz"] + sorted(rd.glob("nq*/records.jsonl.gz")):
            if p.is_file():
                if p.parent != rd and (rd / "records.jsonl.gz").is_file():
                    continue
                out.extend(_api.read_records(p))
    return out


def apply_selection(suite: _api.Suite, cfg: _api.RTTConfig) -> dict:
    """Fixed-sequence selection over the pre-ordered library (Appendix D)."""
    lib = _api.default_library()
    recs = stage_records(suite, "selection")
    failed_records = [r for r in recs if r.get("kind") != "group" and r.get("status") != "OK"]
    if failed_records:
        out = {"selected": None, "path": [], "stopped_at": None,
               "reporting_member": "m4_25zeta", "status": "INCOMPLETE",
               "ltt_threshold": None, "ltt_path": [], "reselected_by_call_size": {},
               "failed_calls": len(failed_records), "certificate_status": _api.CERTIFICATE_STATUS,
               "note": "Selection has failed calls; reporting_member is diagnostic only and is not selected."}
        suite.publish("selection.json", out, kind="json")
        return out
    by = {}
    for r in recs:
        if r.get("status") != "OK":
            continue
        nq = int(r.get("nq", cfg.nq))
        key = (r["policy"].replace("rtt_", ""), r["component"] if r["stage"].startswith("select:") else "?",
               nq, r["stage"])
        by.setdefault((r["policy"].replace("rtt_", ""), r["stage"]), []).append(r)
    path = []
    passing = []
    stopped = None
    for m in lib:
        if not m.selectable:
            path.append({"member": m.name, "status": "NOT_SELECTABLE_VARIANT"})
            continue
        res = {"member": m.name, "pools": {}}
        ok_all = True
        for pool in ("mixture", "clean", "cache", "noise"):
            rr = by.get((m.name, f"select:{pool}"), [])
            n = len(rr)
            if n != cfg.selection_pool_calls or len({r["call_id"] for r in rr}) != n:
                res["pools"][pool] = {"n": n, "expected_n": cfg.selection_pool_calls,
                                      "status": "INCOMPLETE"}
                ok_all = False
                continue
            nz = sum(1 for r in rr if r["down"] > cfg.zeta)
            u = _api.clopper_pearson_upper(nz, n, cfg.delta)
            res["pools"][pool] = {"n": n, "N_zeta": nz, "u": u, "pass": u <= cfg.p_star}
            ok_all = ok_all and u <= cfg.p_star
        mix = by.get((m.name, "select:mixture"), [])
        lb = None
        if ok_all and mix:
            d = np.array([r["down"] for r in mix])
            lb = -_api.betting_ucb(d, m.H_plus(cfg), cfg.delta, "tuned")
        need = cfg.min_certified_mean_tol if m.H_plus_mult <= 1.0 else cfg.min_certified_mean
        res["certified_mean"] = lb
        res["min_certified_mean"] = need
        res["pass"] = bool(ok_all and lb is not None and lb >= need)
        path.append(res)
        if res["pass"]:
            passing.append((lb, m.name))
        else:
            stopped = m.name
            break
    selected = max(passing)[1] if passing else None
    # learn-then-test threshold of the per-node switch
    ltt = None
    ltt_path = []
    lt_dir = suite.stage_dir("selection") / "runs"
    lt_files = list(lt_dir.glob("*_ltt_switch/RESULT/ltt_switch.csv")) if lt_dir.is_dir() else []
    if lt_files:
        rows = _api.read_csv(lt_files[0])
        by_t = {}
        for r in rows:
            by_t.setdefault(int(r["threshold_index"]), []).append(r)
        for ti in sorted(by_t):
            rr = by_t[ti]
            n = len(rr)
            nz = sum(1 for r in rr if float(r["down"]) > cfg.zeta)
            u = _api.clopper_pearson_upper(nz, n, cfg.delta)
            thr = float(rr[0]["threshold"])
            ltt_path.append({"threshold": thr, "n": n, "N_zeta": nz, "u": u})
            if u <= cfg.p_star:
                ltt = thr
            else:
                break
    # reselection per call size (mixture pool only)
    resel = {}
    for nq in cfg.call_sizes:
        if nq == cfg.nq:
            continue
        cands = []
        for m in lib:
            if not m.selectable:
                continue
            rr = by.get((m.name, f"select:mixture:nq{nq}"), [])
            if not rr:
                break
            nz = sum(1 for r in rr if r["down"] > cfg.zeta)
            u = _api.clopper_pearson_upper(nz, len(rr), cfg.delta)
            if u > cfg.p_star:
                break
            lb = -_api.betting_ucb(np.array([r["down"] for r in rr]), m.H_plus(cfg), cfg.delta, "tuned")
            cands.append((lb, m.name))
        if cands:
            resel[str(nq)] = max(cands)[1]
    out = {"selected": selected, "path": path, "stopped_at": stopped, "reporting_member": selected or "m4_25zeta",
           "status": "SELECTED" if selected else "NO_MEMBER_PASSED",
           "certificate_status": _api.CERTIFICATE_STATUS,
           "ltt_threshold": ltt, "ltt_path": ltt_path, "reselected_by_call_size": resel,
           "note": "fixed-member coverage (Lemma 6) does not transfer to the selected member without "
                   "selection-valid calibration; the independent final audit is the release evidence"}
    suite.publish("selection.json", out, kind="json")
    return out


