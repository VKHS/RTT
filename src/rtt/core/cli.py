"""Stage command-line interface, preflight, and device diagnostics."""
from __future__ import annotations

from ._imports import (
    List,
    Optional,
    Path,
    TORCH_AVAILABLE,
    argparse,
    json,
    math,
    np,
    os,
    platform,
    socket,
    sys,
    time,
    torch,
    traceback,
)
from . import api as _api

# ============================================================================
# Section 19: command line, identity, GPU health, preflight, task wrapper
# ============================================================================

RUNNERS = {
    "incumbent": _api.task_incumbent, "proposal": _api.task_proposal, "phase2": _api.task_phase2, "placement": _api.task_placement,
    "calibration": _api.task_calibration, "selection": _api.task_selection, "development": _api.task_development,
    "audit": _api.task_audit, "breadth": _api.task_breadth, "node_transfer": _api.task_node_transfer,
    "diagnostics": _api.task_diagnostics,
}


def identity() -> dict:
    return {"version": _api.VERSION, "version_tag": _api.VERSION_TAG, "expected_runner_basename": _api.EXPECTED_RUNNER_BASENAME,
            "actual_runner_basename": Path(sys.argv[0]).name, "expected_shell_basename": _api.EXPECTED_SHELL_BASENAME}


def system_info(device=None) -> dict:
    info = {"host": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version.split()[0], "numpy": np.__version__, "utc": _api.now_utc(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
    if TORCH_AVAILABLE:
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        info["tf32_disabled"] = not bool(torch.backends.cuda.matmul.allow_tf32)
    return info


def gpu_health(out: Optional[str], repeats: int, size: int) -> int:
    rec = {"version": _api.VERSION, "utc": _api.now_utc(), "host": socket.gethostname()}
    try:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        _api.configure_torch_numerics()
        g = torch.Generator(device="cuda").manual_seed(1234)
        A = torch.randn(size, size, device="cuda", generator=g)
        B = torch.randn(size, size, device="cuda", generator=g)
        ref = None
        for r in range(max(1, repeats)):
            C = A @ B
            torch.cuda.synchronize()
            if not torch.isfinite(C).all():
                raise RuntimeError("non-finite matmul result")
            s = float(C.double().sum().item())
            if ref is None:
                ref = s
            elif abs(s - ref) > 1e-6 * max(1.0, abs(ref)):
                raise RuntimeError("non-reproducible matmul")
        rec.update({"status": "PASS", "gpu": torch.cuda.get_device_name(0)})
        code = 0
    except Exception as e:
        rec.update({"status": "FAIL", "error": str(e)})
        code = _api.EXIT_INFRASTRUCTURE
    if out:
        _api.write_json(Path(out), rec)
    print(_api.to_json(rec))
    return code


def run_stage_task(args) -> int:
    suite = _api.Suite(args.suite)
    cfg, _ = suite.config()
    stage, task, ordinal = args.stage, args.task, args.ordinal
    run_dir = Path(args.run_dir) if args.run_dir else suite.run_dir(stage, f"{ordinal}_{task}")
    run_dir.mkdir(parents=True, exist_ok=True)
    _api.atomic_write_text(run_dir / "command_line.txt", " ".join(sys.argv) + "\n")
    t0 = time.time()
    try:
        info = _api.configure_torch_numerics(deterministic=True, threads=args.threads)
        device = _api.pick_device(require_cuda=args.require_cuda, force_cpu=args.cpu)
        _api.write_json(run_dir / "system.json", {**system_info(device), **info, "device": str(device)})
        _api.set_global_seeds(_api.stable_seed(cfg.seed, stage, task) % (2 ** 31))
        _api.log(f"[TASK] stage={stage} task={task} ordinal={ordinal} device={device} run_dir={run_dir}")
        if stage == "reduce":
            out = _api.reduce_suite(suite, cfg, suite.stage_dir("reduce"))
            summary = {"reduce": True, "release_decision": out.get("release", {}).get("decision")}
        else:
            summary = RUNNERS[stage](suite, args, cfg, device, task, run_dir)
        status = "NOT_RUN" if summary.get("not_run") else "COMPLETE"
        summary.update({"status": "COMPLETE" if status != "NOT_RUN" else "COMPLETE", "task_outcome": status,
                        "version": _api.VERSION, "stage": stage, "task": task, "elapsed_s": time.time() - t0,
                        "scale": cfg.scale_name})
        _api.write_json(run_dir / "summary.json", summary)
        _api.log(f"[TASK] complete in {time.time() - t0:.1f}s")
        return 0
    except _api.RTTFailure as e:
        fc = e.failure_class
        rec = {"status": f"FAILED_{fc}", "failure_class": fc, "retriable": e.retriable, "message": str(e),
               "stage": stage, "task": task, "version": _api.VERSION, "details": e.details,
               "traceback": traceback.format_exc(), "elapsed_s": time.time() - t0}
        _api.write_json(run_dir / "failure_summary.json", rec)
        _api.log(f"[FATAL] {fc}: {e}")
        return _api.exit_code_for(fc)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        infra = any(p in msg for p in _api.INFRA_PATTERNS) or "out of memory" in msg.lower()
        fc = "INFRASTRUCTURE" if infra else "SCIENTIFIC"
        rec = {"status": f"FAILED_{fc}", "failure_class": fc, "retriable": infra, "message": msg,
               "stage": stage, "task": task, "version": _api.VERSION, "traceback": traceback.format_exc(),
               "elapsed_s": time.time() - t0}
        _api.write_json(run_dir / "failure_summary.json", rec)
        traceback.print_exc()
        return _api.exit_code_for(fc)


def init_suite(args) -> int:
    suite = _api.Suite(args.suite)
    suite.manifest.mkdir(parents=True, exist_ok=True)
    if (suite.manifest / "rtt_config.json").exists():
        raise _api.RTTFailure("PREREQUISITE", "Suite already initialized; resume it or choose a new --suite path")
    cfg = _api.config_from_args(args)
    _api.write_suite_config(suite, cfg, args)
    opts = {"breadth": bool(args.breadth), "swap_heads": bool(args.swap_heads), "dev_shard": args.dev_shard,
            "audit_shard": args.audit_shard, "audit_component_shard": args.audit_component_shard}
    _api.write_json(suite.manifest / "task_options.json", opts)
    for st in _api.STAGES:
        tasks = _api.list_tasks(st, cfg, suite, opts)
        _api.write_task_manifest(suite.manifest / f"tasks_{st}.tsv", tasks)
    print(f"[INIT] suite {suite.root} scale={cfg.scale_name}")
    return 0


def semantic_preflight(args) -> int:
    """Tiny synthetic end-to-end run of every stage type in-process.  Fails closed on
    any exception, on any certificate violation, and on failed checker tests."""
    root = Path(args.out_dir or (Path.cwd() / "rtt_preflight")).resolve()
    if root.exists() and any(root.iterdir()):
        raise _api.RTTFailure("PREREQUISITE", "Preflight output directory is not empty; choose a new --out_dir")
    root.mkdir(parents=True, exist_ok=True)
    pa = argparse.Namespace(**vars(args))
    pa.suite = str(root)
    pa.scale = "preflight"
    pa.dataset_kind = "synthetic"
    full_cov = bool(getattr(args, "preflight_full_coverage", False))
    pa.breadth = 1 if full_cov else int(getattr(args, "preflight_breadth", 0))
    pa.swap_heads = 1 if full_cov else 0
    pa.dev_shard = 6
    pa.audit_shard = 12
    pa.audit_component_shard = 12
    pa.require_cuda = False
    init_suite(pa)
    suite = _api.Suite(str(root))
    cfg, _ = suite.config()
    plan = [("incumbent", "main"), ("incumbent", "v2"), ("proposal", "native"),
            ("phase2", "heads_native"), ("phase2", "correctors_native"), ("phase2", "crossfit"),
            ("phase2", "ft_kl"), ("placement", "native")]
    plan += [("calibration", f"native__{m.name}") for m in _api.default_library()]
    plan += [("calibration", "native__swap_no_row")]
    plan += [("selection", f"{m.name}__{p}") for m in _api.default_library() for p in ("mixture", "clean", "cache", "noise")]
    plan += [("selection", "ltt_switch")]
    plan += [("audit", "mixture__00000_00012")] + [(("audit", f"{c}__00000_00012")) for c in _api.ALL_COMPONENTS]
    plan += [("development", "budget__00000_00006"), ("node_transfer", "per_node"),
             ("node_transfer", "transfer__00000_00012"),
             ("diagnostics", "checker_planted"), ("diagnostics", "checker_hp"), ("diagnostics", "checker_underfactor"),
             ("diagnostics", "checker_adversarial"), ("diagnostics", "checker_rounding_off"),
             ("diagnostics", "invariance"), ("diagnostics", "frontier_levels"), ("diagnostics", "depth_exact"),
             ("diagnostics", "planted_g030")]
    manifests = {st: {t["task"]: t["ordinal"] for t in _api.read_task_manifest(suite.manifest / f"tasks_{st}.tsv")}
                 for st in _api.STAGES}
    if full_cov:
        # every task of the training chain, and one shard of every task group of the
        # evaluation stages: exercises breadth proposals, head swaps, every mechanism
        # swap, every fine-tuning variant, call-size variants and every diagnostic
        plan = []
        for st in ("incumbent", "proposal", "phase2", "placement", "calibration", "selection"):
            plan += [(st, t) for t in manifests[st]]
        for st in ("audit", "development", "breadth", "node_transfer", "diagnostics"):
            seen = set()
            for t in manifests[st]:
                grp = t.split("__")[0]
                if "__" in t and grp in seen:
                    continue
                seen.add(grp)
                plan.append((st, t))
    else:
        # the selection gate needs every selection task (incl. call-size variants) to PASS
        extra = [("selection", t) for t in manifests["selection"] if ("selection", t) not in plan]
        last_sel = max(i for i, (st, _) in enumerate(plan) if st == "selection")
        plan = plan[:last_sel + 1] + extra + plan[last_sel + 1:]
    results = []
    t0 = time.time()
    for st, task in plan:
        if st in ("audit", "development", "breadth", "node_transfer", "diagnostics") and ("selection", "gate") not in results:
            rc_gate = _api.run_gate(suite, "selection", cfg)
            if rc_gate != 0:
                raise _api.RTTFailure("PREREQUISITE", f"preflight selection gate did not pass (rc={rc_gate})")
            results.append(("selection", "gate"))
        task_name = task
        ordinal = manifests[st].get(task_name, "9999")
        a = argparse.Namespace(**vars(pa))
        a.stage, a.task, a.ordinal, a.run_dir = st, task_name, ordinal, None
        _api.log(f"[PREFLIGHT] {st}:{task_name}")
        rc = run_stage_task(a)
        if rc != 0:
            fs = suite.run_dir(st, f"{ordinal}_{task_name}") / "failure_summary.json"
            print(fs.read_text() if fs.is_file() else "no failure summary")
            raise _api.RTTFailure("PREREQUISITE", f"preflight task {st}:{task_name} failed rc={rc}")
        results.append((st, task_name))
    a = argparse.Namespace(**vars(pa))
    a.stage, a.task, a.ordinal, a.run_dir = "reduce", "reduce", "0001", None
    if run_stage_task(a) != 0:
        raise _api.RTTFailure("PREREQUISITE", "preflight reduce failed")
    summ = _api.read_json(suite.stage_dir("reduce") / "summary.json")
    checks = []
    audit = _api.stage_records(suite, "audit")
    evald = audit + _api.stage_records(suite, "development") + _api.stage_records(suite, "breadth") + \
        _api.stage_records(suite, "node_transfer") + _api.stage_records(suite, "selection")
    viol = [r for r in evald if r.get("violation_damage") or r.get("violation_row")]
    checks.append(("no certificate violations in any evaluated record", len(viol) == 0,
                   [{k: r.get(k) for k in ("call_id", "policy", "mode", "exact_S_plus", "cert_damage")} for r in viol[:5]]))
    errs = [r for r in evald if r.get("status") == "ERROR"]
    checks.append(("no policy errors in any evaluated record", len(errs) == 0,
                   [(r.get("policy"), r.get("error")) for r in errs[:5]]))
    n_rel = sum(1 for r in evald if r.get("mode") in ("R1", "R2"))
    checks.append(("the evaluation exercised released internal steps (R1/R2)", n_rel > 0, n_rel))
    diag = suite.stage_dir("diagnostics") / "runs"
    for name in ("checker_planted", "checker_hp", "checker_underfactor"):
        ps = list(diag.glob(f"*_{name}/RESULT/{name}.json"))
        ok = bool(ps) and bool(_api.read_json(ps[0]).get("passed"))
        checks.append((f"{name} passed", ok, _api.read_json(ps[0]) if ps else None))
    ps = list(diag.glob("*_checker_adversarial/RESULT/checker_adversarial.json"))
    adv = _api.read_json(ps[0]) if ps else None
    checks.append(("adversarial fields/candidates: released damage within the budget",
                   bool(adv) and bool(adv.get("bound_met")), adv))
    # statistics sanity against closed forms
    checks.append(("Clopper-Pearson(0 of 2048) == 1 - 0.05^(1/2048)",
                   abs(_api.clopper_pearson_upper(0, 2048, 0.05) - (1 - 0.05 ** (1 / 2048))) < 1e-9, None))
    q = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
    pr = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
    pp = _api.floor_project(q, pr, torch.tensor([0.45], dtype=torch.float64))
    floor_ok = bool((pp >= math.exp(-0.45) * pr * (1 - 1e-9)).all().item()) and abs(float(pp.sum().item()) - 1) < 1e-12
    checks.append(("floor projection satisfies the floor and normalisation", floor_ok, pp.tolist()))
    ok_all = all(c[1] for c in checks)
    rep = {"passed": ok_all, "checks": [{"check": c[0], "pass": c[1], "detail": c[2]} for c in checks],
           "elapsed_s": time.time() - t0, "suite": str(root), "release_decision": summ.get("release", {}).get("decision")}
    _api.write_json(root / "PREFLIGHT_REPORT.json", rep)
    print(_api.to_json(rep))
    if not ok_all:
        raise _api.RTTFailure("PREREQUISITE", "semantic preflight checks failed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reference-Tail Trust (RTT) runner")
    p.add_argument("--print_version", action="store_true")
    p.add_argument("--print_identity", action="store_true")
    p.add_argument("--gpu_health_check_only", action="store_true")
    p.add_argument("--gpu_health_out", default=None)
    p.add_argument("--gpu_health_repeats", type=int, default=2)
    p.add_argument("--gpu_health_matrix_size", type=int, default=1024)
    p.add_argument("--semantic_preflight_only", action="store_true")
    p.add_argument("--preflight_full_coverage", action="store_true",
                   help="semantic preflight over every task type (breadth, swaps, all diagnostics); slower")
    p.add_argument("--prepare_dataset_only", action="store_true")
    p.add_argument("--init_suite", action="store_true")
    p.add_argument("--stage", default=None, choices=_api.STAGES + [None])
    p.add_argument("--task", default=None)
    p.add_argument("--ordinal", default="0000")
    p.add_argument("--run_dir", default=None)
    p.add_argument("--stage_gate", default=None)
    p.add_argument("--suite", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--scale", default="full", choices=["full", "pilot", "preflight"])
    p.add_argument("--config", default=None, help="JSON overrides, frozen at suite initialization")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--breadth", type=int, default=1)
    p.add_argument("--swap_heads", type=int, default=1)
    p.add_argument("--dev_shard", type=int, default=128)
    p.add_argument("--audit_shard", type=int, default=128)
    p.add_argument("--audit_component_shard", type=int, default=512)
    p.add_argument("--dataset_kind", default="cached", choices=["cached", "synthetic"])
    p.add_argument("--dataset_name", default="ogbn-arxiv")
    p.add_argument("--dataset_root", default="./datasets/ogb_cache")
    p.add_argument("--graph_pt", default=None)
    p.add_argument("--rtt_graph_npz", default="./datasets/graph/rtt_ogbn_arxiv_graph.npz")
    p.add_argument("--external_feature_pt", default=None)
    p.add_argument("--feature_normalization", default="standard", choices=["standard", "none"])
    p.add_argument("--use_node_year", action="store_true")
    p.add_argument("--synthetic_nodes", type=int, default=1600)
    p.add_argument("--synthetic_classes", type=int, default=6)
    p.add_argument("--synthetic_features", type=int, default=16)
    p.add_argument("--synthetic_seed", type=int, default=7)
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_version:
        print(_api.VERSION)
        return 0
    if args.print_identity:
        print(json.dumps(identity(), sort_keys=True))
        return 0
    if args.gpu_health_check_only:
        return gpu_health(args.gpu_health_out, args.gpu_health_repeats, args.gpu_health_matrix_size)
    try:
        if args.semantic_preflight_only:
            return semantic_preflight(args)
        if args.prepare_dataset_only:
            g = _api.load_graph(args, allow_download=True)
            fp = g.fingerprint()
            cfg = _api.config_from_args(args)
            roles = _api.make_data_roles(g, cfg, seed=args.seed)
            out = {"graph": fp, "roles": roles.summary(), "npz": args.rtt_graph_npz, "utc": _api.now_utc()}
            if args.out_dir:
                _api.write_json(Path(args.out_dir) / "dataset_prep.json", out)
            print(_api.to_json(out))
            return 0
        if args.init_suite:
            return init_suite(args)
        if args.stage_gate:
            suite = _api.Suite(args.suite)
            cfg, _ = suite.config()
            return _api.run_gate(suite, args.stage_gate, cfg)
        if args.stage:
            return run_stage_task(args)
    except _api.RTTFailure as e:
        print(f"[FATAL] {e.failure_class}: {e}", file=sys.stderr)
        traceback.print_exc()
        return _api.exit_code_for(e.failure_class)
    build_parser().print_help()
    return _api.EXIT_USAGE


