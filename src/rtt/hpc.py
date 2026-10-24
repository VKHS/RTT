"""Immutable Slurm orchestration for the separate RTT dataset experiments.

This module uses only the Python standard library. Numerical work belongs to
``rtt.cli`` and runs exclusively inside the submitted compute jobs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


DATASETS = (
    "ogbn_arxiv", "good_arxiv_degree", "good_arxiv_time", "twitch", "facebook100",
    "elliptic", "ogbn_products", "imagenet_c", "acs_income", "ogbg_molpcba",
)
STAGES = ("train", "calibrate", "select", "audit", "report")
DEFAULT_BASE = "/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls"
TIME_DEFAULTS = {"prepare": "03:00:00", "train": "20:00:00", "calibrate": "16:00:00",
                 "select": "12:00:00", "audit": "12:00:00", "report": "02:00:00"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def orchestration(run_dir):
    return Path(run_dir) / "orchestration"


def snapshot(repository, run_dir):
    """Copy executable sources and configurations, excluding data and checkpoints."""
    repository = Path(repository).resolve()
    destination = orchestration(run_dir) / "code"
    destination.mkdir(parents=True, exist_ok=False)
    allowed_directories = {"src", "scripts", "experiments", "configs", "requirements"}
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".DS_Store")
    for item in sorted(repository.iterdir()):
        if item.is_dir() and item.name in allowed_directories:
            shutil.copytree(item, destination / item.name, ignore=ignore)
        elif item.is_file() and (item.suffix in {".py", ".sh", ".toml", ".json"}
                                 or item.name.startswith("requirements")):
            shutil.copy2(item, destination / item.name)
    required = ("src/rtt/cli.py", "src/rtt/hpc.py", "scripts/arrhenius.sh")
    for name in required:
        if not (destination / name).is_file():
            raise RuntimeError(f"Repository is missing required executable: {name}")
    hashes = {p.relative_to(destination).as_posix(): sha256(p)
              for p in sorted(destination.rglob("*")) if p.is_file()}
    write_json(orchestration(run_dir) / "source_sha256.json", hashes)
    return destination


def verify_frozen(run_dir, plan):
    state = orchestration(run_dir)
    plan_path = state / "plan.json"
    if sha256(plan_path) != read_json(state / "plan_sha256.json")["sha256"]:
        raise RuntimeError("Frozen Slurm plan changed; create a new run for changed settings")
    if sha256(plan["config"]) != plan["config_sha256"]:
        raise RuntimeError("Frozen dataset configuration changed; create a new run")
    if sha256(state / "source_sha256.json") != plan["source_manifest_sha256"]:
        raise RuntimeError("Frozen source inventory changed; create a new run")
    source = Path(plan["code_dir"])
    hashes = read_json(state / "source_sha256.json")
    actual_files = {item.relative_to(source).as_posix() for item in source.rglob("*")
                    if item.is_file() and "__pycache__" not in item.parts
                    and ".pytest_cache" not in item.parts and item.suffix != ".pyc"
                    and item.name != ".DS_Store"}
    if actual_files != set(hashes):
        raise RuntimeError("Frozen source file inventory changed; create a new run")
    for name, expected in hashes.items():
        item = source / name
        if not item.is_file() or sha256(item) != expected:
            raise RuntimeError(f"Frozen source changed or disappeared: {name}")


def expected_artifacts(stage, index=0):
    if stage == "prepare":
        return ["data_manifest.json"]
    if stage == "train":
        return ["manifest.json", "protocol.json", "checkpoints/model.pt",
                "checkpoints/adaptation.pt", "training.json"]
    if stage == "calibrate":
        return ["calibration.json"]
    if stage == "select":
        return ["selection.json"]
    if stage == "audit":
        prefix = f"records/audit_shard_{index:04d}"
        return [prefix + ".jsonl", prefix + ".meta.json"]
    if stage == "report":
        return ["summary.json", "metrics.csv", "REPORT.md"]
    raise ValueError(f"Unknown stage: {stage}")


def receipt_path(run_dir, stage, index=0):
    return orchestration(run_dir) / "completed" / f"{stage}_{index:04d}.json"


def failure_path(run_dir, stage, index=0):
    return orchestration(run_dir) / "failures" / f"{stage}_{index:04d}.json"


def complete(run_dir, plan, stage, index=0):
    path = receipt_path(run_dir, stage, index)
    if not path.is_file():
        return False
    record = read_json(path)
    if record.get("status") != "COMPLETE" or record.get("config_sha256") != plan["config_sha256"]:
        raise RuntimeError(f"Invalid stage completion receipt: {path}")
    expected = set(expected_artifacts(stage, index))
    if set(record.get("artifacts", {})) != expected:
        raise RuntimeError(f"Incomplete artifact ledger: {path}")
    for name, digest in record["artifacts"].items():
        artifact = Path(run_dir) / name
        if not artifact.is_file() or sha256(artifact) != digest:
            raise RuntimeError(f"Completed stage artifact changed or disappeared: {artifact}")
    return True


def indices_for(plan, stage):
    return list(range(plan["audit_shards"])) if stage == "audit" else [0]


def missing_indices(run_dir, plan, stage):
    missing = []
    for index in indices_for(plan, stage):
        if complete(run_dir, plan, stage, index):
            continue
        failure = failure_path(run_dir, stage, index)
        if failure.is_file() and not read_json(failure).get("retriable", False):
            raise RuntimeError(f"Non-retriable stage failure: {failure}; review it before starting a new run")
        missing.append(index)
    return missing


def job_records(run_dir):
    path = orchestration(run_dir) / "jobs.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assert_no_active_jobs(run_dir):
    known = {str(record["job_id"]) for record in job_records(run_dir)}
    if not known:
        return
    result = subprocess.run(["squeue", "--noheader", "--format=%F"], check=True,
                            text=True, capture_output=True)
    active = known.intersection(line.strip() for line in result.stdout.splitlines())
    if active:
        raise RuntimeError("Run still has active jobs: " + ", ".join(sorted(active)))


def record_submission(run_dir, record):
    path = orchestration(run_dir) / "jobs.jsonl"
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps({**record, "utc": now()}, sort_keys=True) + "\n")


def stage_resources(plan, stage):
    memory = "180G" if plan["dataset"] == "ogbn_products" and stage != "report" else "110G"
    if stage == "prepare":
        memory = "64G"
    if stage == "report":
        memory = "64G"
    walltime = TIME_DEFAULTS[stage]
    if plan["profile"] == "smoke":
        walltime = "02:00:00"
    elif plan["profile"] == "pilot":
        walltime = "04:00:00"
    return {"cpus": plan["cpus"] if stage != "report" else min(4, plan["cpus"]),
            "memory": plan["resources"].get(stage, {}).get("memory", memory),
            "walltime": plan["resources"].get(stage, {}).get("walltime", walltime)}


def submit_job(run_dir, plan, stage, indices, dependencies=()):
    logs = Path(run_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    resources = stage_resources(plan, stage)
    command = ["sbatch", "--parsable", f"--account={plan['account']}", "--partition=gpu",
               "--ntasks=1", "--ntasks-per-node=1", "--gpus=1",
               f"--cpus-per-task={resources['cpus']}", f"--mem={resources['memory']}",
               f"--time={resources['walltime']}", "--kill-on-invalid-dep=yes",
               f"--job-name=rtt_{plan['dataset']}_{stage}", "--export=ALL",
               f"--chdir={plan['code_dir']}",
               f"--output={logs / (stage + '_%A_%a.out')}",
               f"--error={logs / (stage + '_%A_%a.err')}"]
    if stage == "audit":
        command.append("--array=" + ",".join(map(str, indices)) + f"%{plan['max_concurrent']}")
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(map(str, dependencies)))
    command += [str(Path(plan["code_dir"]) / "scripts" / "arrhenius.sh"),
                "_worker", str(run_dir), stage]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    identifier = result.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", identifier):
        raise RuntimeError(f"Invalid sbatch job identifier: {result.stdout!r}")
    record_submission(run_dir, {"job_id": identifier, "stage": stage, "indices": list(indices),
                                "dependencies": list(dependencies), "resources": resources})
    print(f"Submitted {stage}: job {identifier}; task indices {indices}", flush=True)
    return identifier


def submit_pipeline(run_dir, plan):
    verify_frozen(run_dir, plan)
    lock_path = orchestration(run_dir) / "submission.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert_no_active_jobs(run_dir)
        # Reject invalid previous outcomes before adding any new jobs to Slurm.
        pending = {stage: missing_indices(run_dir, plan, stage) for stage in plan["stages"]}
        previous = []
        for stage in plan["stages"]:
            if pending[stage]:
                previous = [submit_job(run_dir, plan, stage, pending[stage], previous)]
    print(f"Run directory: {run_dir}")
    print(f"Status: bash experiments/{plan['dataset']}/run.sh status {run_dir}")


def initialize(args):
    run_dir = Path(args.run_dir).resolve()
    state = orchestration(run_dir)
    if state.exists() and any(state.iterdir()):
        raise RuntimeError("Run already has an orchestration plan; use resume or a new directory")
    if (run_dir / "manifest.json").exists():
        raise RuntimeError("Run already contains a numerical experiment; choose a new Slurm run directory")
    state.mkdir(parents=True, exist_ok=True)
    config = Path(args.config).resolve()
    configured = read_json(config).get("dataset")
    configured_name = configured.get("name") if isinstance(configured, dict) else configured
    if str(configured_name).lower().replace("-", "_") != args.dataset:
        raise RuntimeError("The selected dataset and configuration dataset.name differ")
    code = snapshot(args.repo, run_dir)
    frozen_config = state / "config.json"
    shutil.copy2(config, frozen_config)
    profile = args.profile
    shards = int(os.environ.get("RTT_AUDIT_SHARDS", "1" if profile == "smoke" else "2" if profile == "pilot" else "8"))
    cpus = int(os.environ.get("RTT_CPUS", "16"))
    concurrency = int(os.environ.get("RTT_MAX_CONCURRENT", "2"))
    if min(shards, cpus, concurrency) < 1:
        raise ValueError("Audit shards, CPUs, and concurrency must all be positive")
    base = os.environ.get("RTT_BASE", DEFAULT_BASE)
    plan = {
        "schema": 1, "created_utc": now(), "dataset": args.dataset, "profile": profile,
        "stages": ["prepare"] if args.prepare_only else list(STAGES),
        "run_dir": str(run_dir), "code_dir": str(code), "config": str(frozen_config),
        "config_sha256": sha256(frozen_config), "data_root": str(Path(args.data_root).resolve()),
        "source_manifest_sha256": sha256(state / "source_sha256.json"),
        "download": bool(args.download), "device": "cuda", "base": base,
        "venv": os.environ.get("RTT_VENV", str(Path(base) / "code/venv_arrhenius_gh200")),
        "python_module": os.environ.get("RTT_PYTHON_MODULE", "GPU/Python/3.13.5-bare-gcc-2025b-eb"),
        "build_module": os.environ.get("RTT_BUILD_MODULE", "GPU/buildenv-nvhpc/25.9-cu13.0"),
        "account": os.environ.get("RTT_ACCOUNT", "naiss2025-5-643-gpu"),
        "storage_project": "naiss2025-6-428", "partition": "gpu", "cpus": cpus,
        "max_concurrent": concurrency, "audit_shards": shards,
        "resources": {stage: {field: os.environ[key] for field, key in (
            ("memory", f"RTT_MEM_{stage.upper()}"), ("walltime", f"RTT_TIME_{stage.upper()}"))
            if key in os.environ} for stage in ("prepare",) + STAGES},
        "sampling_note": "A completed job is an execution result; statistical acceptance is decided by the numerical audit.",
    }
    if profile == "smoke":
        plan["download"] = False
    write_json(state / "plan.json", plan)
    write_json(state / "plan_sha256.json", {"sha256": sha256(state / "plan.json")})
    submit_pipeline(run_dir, plan)


def numerical_command(run_dir, plan, stage, index=0):
    command = [sys.executable, "-u", "-m", "rtt.cli", stage,
               "--config", plan["config"], "--run-dir", str(run_dir),
               "--profile", plan["profile"], "--data-root", plan["data_root"],
               "--device", plan["device"]]
    if plan.get("download") and stage in {"prepare", "train"}:
        command.append("--download")
    if stage == "audit":
        command += ["--shard-index", str(index), "--shards", str(plan["audit_shards"])]
    return command


def worker(args):
    run_dir = Path(args.run_dir).resolve()
    plan = read_json(orchestration(run_dir) / "plan.json")
    verify_frozen(run_dir, plan)
    stage = args.stage
    if stage not in plan["stages"]:
        raise RuntimeError("Worker stage is not in the frozen plan")
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if stage == "audit" else 0
    if index not in indices_for(plan, stage):
        raise RuntimeError("Audit task index is outside the frozen shard range")
    if complete(run_dir, plan, stage, index):
        print("Stage already complete with verified artifacts")
        return
    failure = failure_path(run_dir, stage, index)
    if failure.is_file() and not read_json(failure).get("retriable", False):
        raise RuntimeError(f"Non-retriable previous failure: {failure}")
    prior_stages = plan["stages"][:plan["stages"].index(stage)]
    for previous in prior_stages:
        if not all(complete(run_dir, plan, previous, i) for i in indices_for(plan, previous)):
            raise RuntimeError(f"Prerequisite stage is incomplete: {previous}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(plan["code_dir"]) / "src")
    command = numerical_command(run_dir, plan, stage, index)
    started = now()
    try:
        subprocess.run(command, check=True, cwd=plan["code_dir"], env=environment)
        artifacts = {}
        for name in expected_artifacts(stage, index):
            path = run_dir / name
            if not path.is_file():
                raise RuntimeError(f"Numerical stage did not produce its required artifact: {path}")
            artifacts[name] = sha256(path)
        write_json(receipt_path(run_dir, stage, index), {
            "status": "COMPLETE", "stage": stage, "index": index,
            "started_utc": started, "completed_utc": now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "config_sha256": plan["config_sha256"], "artifacts": artifacts,
        })
        if failure.exists():
            failure.unlink()
    except BaseException as error:
        code = error.returncode if isinstance(error, subprocess.CalledProcessError) else None
        write_json(failure, {"stage": stage, "index": index, "utc": now(),
                             "error": str(error), "returncode": code,
                             "retriable": code in {75, 137, 143, -9, -15}})
        raise


def status(run_dir):
    run_dir = Path(run_dir).resolve()
    plan = read_json(orchestration(run_dir) / "plan.json")
    print(f"Dataset: {plan['dataset']} | Profile: {plan['profile']} | Run: {run_dir}")
    verify_frozen(run_dir, plan)
    for stage in plan["stages"]:
        indices = indices_for(plan, stage)
        count = sum(complete(run_dir, plan, stage, index) for index in indices)
        failed = sum(failure_path(run_dir, stage, index).exists() for index in indices)
        print(f"{stage:12s} {count}/{len(indices)} complete; {failed} failure records")
    summary = run_dir / "summary.json"
    if summary.is_file():
        print("Numerical/statistical summary:", summary)
    print(plan["sampling_note"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit", "resume", "status", "worker"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--config")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="full")
    parser.add_argument("--data-root")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--stage", choices=("prepare",) + STAGES)
    args = parser.parse_args(argv)
    if args.command == "submit":
        if not (args.dataset and args.config and args.data_root):
            parser.error("submit requires --dataset, --config, and --data-root")
        initialize(args)
    elif args.command == "worker":
        if args.stage is None:
            parser.error("worker requires --stage")
        worker(args)
    elif args.command == "status":
        status(args.run_dir)
    else:
        plan = read_json(orchestration(args.run_dir) / "plan.json")
        submit_pipeline(Path(args.run_dir).resolve(), plan)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"RTT submission error: {error}", file=sys.stderr)
        raise SystemExit(1)
