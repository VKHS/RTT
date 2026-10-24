#!/usr/bin/env python3
"""Small, standard-library-only Slurm driver for the native ogbn-arxiv study.

The numerical runner owns experiments. This file only freezes a core task plan,
submits dependency-linked jobs, checks completion, and resumes identical tasks.
Run through run_rtt_iclr2027.sh on Arrhenius; ``status`` also works locally.
"""
from __future__ import annotations

import argparse
import csv
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


STAGE_DIRS = {
    "incumbent": "010_INCUMBENT", "proposal": "020_PROPOSAL",
    "phase2": "030_PHASE2", "placement": "040_PLACEMENT",
    "calibration": "050_CALIBRATION", "selection": "060_SELECTION",
    "development": "070_DEVELOPMENT", "audit": "080_AUDIT",
    "breadth": "085_BREADTH", "node_transfer": "090_NODE_TRANSFER",
    "diagnostics": "095_DIAGNOSTICS", "reduce": "900_REDUCE",
}
CHAIN = ("incumbent", "proposal", "phase2", "placement", "calibration", "selection")
EVALUATION = ("audit", "node_transfer", "diagnostics")
CORE = CHAIN + EVALUATION + ("reduce",)
RUNNER = "run_rtt_iclr2027_arrhenius.py"
TIME_DEFAULTS = {
    "incumbent": "04:00:00", "proposal": "08:00:00", "phase2": "20:00:00",
    "placement": "08:00:00", "calibration": "16:00:00", "selection": "08:00:00",
    "audit": "12:00:00", "node_transfer": "12:00:00",
    "diagnostics": "20:00:00", "reduce": "04:00:00",
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tasks_for(suite, stage):
    path = suite / "000_MANIFEST" / f"tasks_{stage}.tsv"
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def result_dir(suite, stage, task):
    return suite / STAGE_DIRS[stage] / "runs" / f"{task['ordinal']}_{task['task']}" / "RESULT"


def complete(suite, stage, task):
    path = result_dir(suite, stage, task) / "summary.json"
    if not path.exists():
        return False
    info = read_json(path)
    if info.get("status") != "COMPLETE" or info.get("task_outcome") == "NOT_RUN":
        return False
    if info.get("passed") is False or info.get("bound_met") is False:
        return False
    for stats in info.get("quick", {}).values():
        if stats.get("errors", 0) or stats.get("violations", 0):
            return False
    return True


def choose_core(stage, task):
    """Exclude breadth/sweep tasks; retain the method and its matched comparisons."""
    name = task["task"]
    if stage == "incumbent":
        return name == "main"
    if stage == "proposal":
        return name == "native"
    if stage == "phase2":
        return name in {"heads_native", "correctors_native", "crossfit"}
    if stage == "placement":
        return name == "native"
    if stage == "calibration":
        return name.startswith("native__m")
    if stage == "selection":
        return "__nq" not in name and name != "ltt_switch"
    if stage == "node_transfer":
        return name == "per_node"
    if stage == "diagnostics":
        return name in {"checker_planted", "checker_hp", "checker_underfactor", "checker_adversarial"}
    return stage in {"audit", "reduce"}


def freeze_core_manifests(suite):
    manifest = suite / "000_MANIFEST"
    full = manifest / "available_tasks"
    full.mkdir()
    counts, skipped = {}, {}
    for stage in STAGE_DIRS:
        source = manifest / f"tasks_{stage}.tsv"
        shutil.copy2(source, full / source.name)
        original = tasks_for(suite, stage)
        selected = [task for task in original if choose_core(stage, task)]
        skipped[stage] = [task["task"] for task in original if task not in selected]
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["ordinal", "task", "stage"], delimiter="\t")
            writer.writeheader()
            writer.writerows(selected)
        counts[stage] = len(selected)
    return counts, skipped


def runner_command(plan, *arguments):
    return [sys.executable, "-u", str(Path(plan["code_dir"]) / RUNNER), *map(str, arguments)]


def dataset_arguments(plan):
    args = ["--dataset_kind", plan["dataset_kind"], "--dataset_name", "ogbn-arxiv",
            "--dataset_root", plan["dataset_root"], "--rtt_graph_npz", plan["graph_npz"],
            "--feature_normalization", "standard", "--threads", str(plan["cpus"])]
    # Loading an old graph.pt is an explicit user choice, never an automatic fallback.
    if plan.get("graph_pt"):
        args += ["--graph_pt", plan["graph_pt"]]
    return args


def config_arguments(plan):
    return ["--config", plan["config_override"]] if plan.get("config_override") else []


def run_checked(command, **kwargs):
    print("[RUN] " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, **kwargs)


def snapshot(source, suite):
    destination = suite / "000_MANIFEST" / "code"
    destination.mkdir(parents=True)
    hashes = {}
    for file in sorted(source.iterdir()):
        if file.is_file() and (file.suffix in {".py", ".sh", ".json", ".toml"}
                               or file.name.startswith("requirements")):
            shutil.copy2(file, destination / file.name)
            hashes[file.name] = sha256(destination / file.name)
        elif file.is_dir() and file.name in {"rtt", "src", "configs", "experiments", "requirements", "scripts"}:
            shutil.copytree(file, destination / file.name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for child in sorted((destination / file.name).rglob("*")):
                if child.is_file():
                    hashes[str(child.relative_to(destination))] = sha256(child)
    for name in (RUNNER, "rtt_pipeline.py", "run_rtt_iclr2027.sh"):
        if name not in hashes:
            raise RuntimeError(f"Required source file absent: {name}")
    write_json(suite / "000_MANIFEST" / "source_sha256.json", hashes)
    return destination


def verify_source(suite, plan):
    for name, expected in read_json(suite / "000_MANIFEST" / "source_sha256.json").items():
        if sha256(Path(plan["code_dir"]) / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}; start a new suite for changed code")
    protocol = suite / "000_MANIFEST" / "protocol_sha256.json"
    if protocol.exists():
        for name, expected in read_json(protocol).items():
            if sha256(suite / name) != expected:
                raise RuntimeError(f"Frozen protocol changed: {name}; use a new suite for a new protocol")
    cache = plan.get("graph_fingerprint")
    if cache:
        stat = Path(plan["graph_npz"]).stat()
        if stat.st_size != cache["size"] or stat.st_mtime_ns != cache["mtime_ns"]:
            if sha256(plan["graph_npz"]) != cache["sha256"]:
                raise RuntimeError("Prepared dataset changed since this suite was initialized")


def verify_artifacts(suite):
    """Check retained stage checkpoints before deciding which tasks can be skipped."""
    artifacts = suite / "ARTIFACTS"
    if not artifacts.is_dir():
        return
    for file in sorted(artifacts.glob("*.manifest.json")):
        record = read_json(file)
        name = record["artifact"]
        if Path(name).name != name:
            raise RuntimeError(f"Invalid artifact name in {file}")
        artifact = artifacts / name
        if not artifact.is_file() or sha256(artifact) != record["sha256"]:
            raise RuntimeError(f"Retained artifact missing or altered: {artifact}")


def initialize(args):
    suite = Path(args.suite).resolve()
    suite.mkdir(parents=True, exist_ok=True)
    manifest = suite / "000_MANIFEST"
    if (manifest / "rtt_config.json").exists() or (manifest / "pipeline.json").exists():
        raise RuntimeError("Suite already initialized; use resume rather than overwriting its protocol")
    source = Path(args.code_dir).resolve()
    code = snapshot(source, suite)
    mode = args.mode
    plan = {
        "created_utc": now(), "profile": "native_arxiv_core", "mode": mode,
        "scale": {"smoke": "preflight", "pilot": "pilot", "full": "full"}[mode],
        "dataset_kind": "synthetic" if mode == "smoke" else "cached",
        "dataset_root": str(Path(os.environ["RTT_DATA_ROOT"]).resolve()),
        "graph_npz": str(Path(os.environ["RTT_GRAPH_NPZ"]).resolve()),
        "graph_pt": os.environ.get("RTT_GRAPH_PT"), "code_dir": str(code),
        "seed": int(os.environ.get("RTT_SEED", "1")),
        "cpus": int(os.environ.get("RTT_CPUS", "16")),
        "max_concurrent": int(os.environ.get("RTT_MAX_CONCURRENT", "2")),
        "account": os.environ.get("RTT_ACCOUNT", "naiss2025-5-643-gpu"),
        "partition": "gpu", "venv": os.environ["RTT_VENV"],
        "base": os.environ["RTT_BASE"],
        "python_module": os.environ["RTT_PYTHON_MODULE"],
        "build_module": os.environ["RTT_BUILD_MODULE"],
        "stages": list(CORE),
    }
    if plan["cpus"] < 1 or plan["max_concurrent"] < 1:
        raise ValueError("CPU count and concurrency must be positive")
    if os.environ.get("RTT_CONFIG"):
        original = Path(os.environ["RTT_CONFIG"]).resolve()
        # Parse now for a clear failure; the numerical runner validates field names.
        read_json(original)
        frozen = code / "submitted_config.json"
        shutil.copy2(original, frozen)
        plan["config_override"] = str(frozen)
        sources = read_json(manifest / "source_sha256.json")
        sources[frozen.name] = sha256(frozen)
        write_json(manifest / "source_sha256.json", sources)
    if mode != "smoke":
        graph = Path(plan["graph_npz"])
        graph.parent.mkdir(parents=True, exist_ok=True)
        with graph.with_suffix(graph.suffix + ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run_checked(runner_command(plan, "--prepare_dataset_only", "--scale", plan["scale"],
                                       "--seed", plan["seed"], "--out_dir", manifest,
                                       *dataset_arguments(plan), *config_arguments(plan)))
        if not graph.is_file():
            raise RuntimeError(f"Data preparation did not create the expected NPZ: {graph}")
        st = graph.stat()
        plan["graph_fingerprint"] = {"sha256": sha256(graph), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    run_checked(runner_command(plan, "--init_suite", "--suite", suite,
                               "--scale", plan["scale"], "--seed", plan["seed"],
                               "--breadth", "0", "--swap_heads", "0",
                               "--audit_shard", os.environ.get("RTT_AUDIT_SHARD", "128"),
                               "--audit_component_shard", os.environ.get("RTT_COMPONENT_SHARD", "512"),
                               *dataset_arguments(plan), *config_arguments(plan)))
    plan["task_counts"], plan["excluded_tasks"] = freeze_core_manifests(suite)
    plan["scope_note"] = (
        "Native ogbn-arxiv training, matched correctors, calibration, selection, all registered "
        "audit components, reserved-node audit and four checker diagnostics. Development sweeps, "
        "external proposals, fine-tuning baselines, call-size reselection, temporal transfer and "
        "other diagnostics are excluded. Smoke uses a synthetic graph; pilot/full use ogbn-arxiv. "
        "Completion of jobs does not imply a statistical release or reproduce published values."
    )
    write_json(manifest / "pipeline.json", plan)
    frozen_paths = [manifest / name for name in ("pipeline.json", "rtt_config.json", "task_options.json")]
    frozen_paths += sorted(manifest.glob("tasks_*.tsv"))
    if (manifest / "dataset_prep.json").exists():
        frozen_paths.append(manifest / "dataset_prep.json")
    write_json(manifest / "protocol_sha256.json",
               {str(path.relative_to(suite)): sha256(path) for path in frozen_paths})
    submit_pipeline(suite, plan)


def record_submission(suite, entry):
    file = suite / "000_MANIFEST" / "jobs.jsonl"
    with file.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps({**entry, "utc": now()}) + "\n")


def assert_no_active_jobs(suite):
    path = suite / "000_MANIFEST" / "jobs.jsonl"
    if not path.exists():
        return
    previous = {str(json.loads(line)["job_id"]) for line in path.read_text().splitlines() if line.strip()}
    result = subprocess.run(["squeue", "--noheader", "--format=%F"], check=True,
                            text=True, capture_output=True)
    active = previous & {line.strip() for line in result.stdout.splitlines()}
    if active:
        raise RuntimeError("This suite still has active jobs: " + ", ".join(sorted(active)) +
                           ". Let them finish or cancel them before resuming.")


def submit_job(suite, plan, role, stage, dependencies=(), indices=()):
    logs = suite / "logs"
    logs.mkdir(exist_ok=True)
    time_default = TIME_DEFAULTS.get(stage, "01:00:00")
    if plan["mode"] == "smoke":
        time_default = "02:00:00"
    elif plan["mode"] == "pilot":
        time_default = "04:00:00"
    cpus = plan["cpus"] if role == "worker" else 4
    mem_default = "64G" if stage == "incumbent" else "110G"
    if role == "gate":
        mem_default, time_default = "24G", "01:00:00"
    if stage == "reduce":
        mem_default = "180G"
    mem = os.environ.get(f"RTT_MEM_{stage.upper()}", mem_default)
    walltime = os.environ.get(f"RTT_TIME_{stage.upper()}", time_default)
    code = Path(plan["code_dir"])
    command = ["sbatch", "--parsable", "--account=" + plan["account"], "--partition=gpu",
               "--ntasks=1", "--ntasks-per-node=1", "--gpus=1", f"--cpus-per-task={cpus}",
               "--mem=" + mem, "--time=" + walltime, "--kill-on-invalid-dep=yes",
               "--job-name=rtt_" + stage + ("_gate" if role == "gate" else ""),
               "--chdir=" + str(code), "--export=ALL",
               "--output=" + str(logs / f"{stage}_{role}_%A_%a.out"),
               "--error=" + str(logs / f"{stage}_{role}_%A_%a.err")]
    if indices:
        command += ["--array=" + ",".join(map(str, indices)) + f"%{plan['max_concurrent']}"]
    if dependencies:
        command += ["--dependency=afterok:" + ":".join(dependencies)]
    command += [str(code / "run_rtt_iclr2027.sh"), "_" + role, str(suite), stage]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    job = result.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", job):
        raise RuntimeError(f"Unexpected sbatch response: {result.stdout!r}")
    record_submission(suite, {"job_id": job, "stage": stage, "role": role,
                              "indices": list(indices), "dependencies": list(dependencies),
                              "resources": {"cpus": cpus, "mem": mem, "walltime": walltime}})
    print(f"[SUBMITTED] {stage}:{role} job={job}", flush=True)
    return job


def unfinished_indices(suite, stage):
    tasks = tasks_for(suite, stage)
    missing = [i for i, task in enumerate(tasks) if not complete(suite, stage, task)]
    for i in missing:
        rd = result_dir(suite, stage, tasks[i])
        failure = rd / "failure_summary.json"
        if failure.exists() and not read_json(failure).get("retriable", False):
            raise RuntimeError(f"Non-retriable task failure: {failure}. Review it; changed code needs a new suite.")
        if (rd / "summary.json").exists():
            raise RuntimeError(f"Task completed with a failed/missing scientific check: {rd}. It is not retried.")
    return missing


def submit_stage(suite, plan, stage, dependencies):
    missing = unfinished_indices(suite, stage)
    if missing:
        dependencies = [submit_job(suite, plan, "worker", stage, dependencies, missing)]
    return submit_job(suite, plan, "gate", stage, dependencies)


def submit_pipeline(suite, plan):
    verify_source(suite, plan)
    assert_no_active_jobs(suite)
    # A lock prevents two simultaneous resume requests from duplicating the audit.
    with (suite / "000_MANIFEST" / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert_no_active_jobs(suite)
        verify_artifacts(suite)
        # Validate all previous outcomes before submitting any new dependency jobs.
        for stage in CORE:
            unfinished_indices(suite, stage)
        dependency = []
        for stage in CHAIN:
            dependency = [submit_stage(suite, plan, stage, dependency)]
        endpoints = [submit_stage(suite, plan, stage, dependency) for stage in EVALUATION]
        submit_stage(suite, plan, "reduce", endpoints)
    print(f"[SUITE] {suite}\n[STATUS] bash run_rtt_iclr2027.sh status {suite}")


def worker(args, plan):
    suite = Path(args.suite).resolve()
    verify_source(suite, plan)
    tasks = tasks_for(suite, args.stage)
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if index < 0 or index >= len(tasks):
        raise RuntimeError(f"Array index {index} outside manifest for {args.stage}")
    task = tasks[index]
    if complete(suite, args.stage, task):
        print("[SKIP] Task already complete with the same frozen protocol")
        return
    result = result_dir(suite, args.stage, task)
    result.mkdir(parents=True, exist_ok=True)
    write_json(result / "launch.json", {"utc": now(), "job_id": os.environ.get("SLURM_JOB_ID"),
                                        "array_task": index, "mode": plan["mode"], "task": task})
    try:
        run_checked(runner_command(plan, "--stage", args.stage, "--task", task["task"],
                                   "--ordinal", task["ordinal"], "--suite", suite,
                                   "--run_dir", result, "--require_cuda", *dataset_arguments(plan)))
        if not complete(suite, args.stage, task):
            raise RuntimeError(f"Task did not pass its execution/checker checks: {result}")
    except BaseException as error:
        write_json(result / "launcher_failure.json", {"utc": now(), "error": str(error)})
        raise


def gate(args, plan):
    suite = Path(args.suite).resolve()
    verify_source(suite, plan)
    missing = [task["task"] for task in tasks_for(suite, args.stage) if not complete(suite, args.stage, task)]
    if missing:
        raise RuntimeError("Unsuccessful tasks in stage: " + ", ".join(missing))
    run_checked(runner_command(plan, "--stage_gate", args.stage, "--suite", suite))


def status(suite):
    plan = read_json(suite / "000_MANIFEST" / "pipeline.json")
    print(f"Suite: {suite}\nMode: {plan['mode']} | profile: {plan['profile']}")
    for stage in CORE:
        tasks = tasks_for(suite, stage)
        count = sum(complete(suite, stage, task) for task in tasks)
        gate_path = suite / STAGE_DIRS[stage] / "GATE" / "gate.json"
        decision = read_json(gate_path).get("decision", "UNKNOWN") if gate_path.exists() else "PENDING"
        print(f"{stage:15s} {count:4d}/{len(tasks):<4d} tasks complete; gate={decision}")
    report = suite / STAGE_DIRS["reduce"] / "summary.json"
    if report.exists():
        summary = read_json(report)
        print("Selection:", summary.get("selection", {}).get("status"))
        print("Statistical release:", summary.get("release", {}).get("decision"))
    print(plan["scope_note"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["initialize", "resume", "worker", "gate", "status", "reduce"])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--code-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument("--stage", choices=list(STAGE_DIRS))
    args = parser.parse_args(argv)
    suite = Path(args.suite).resolve()
    if args.command == "initialize":
        initialize(args)
    elif args.command == "status":
        status(suite)
    else:
        plan = read_json(suite / "000_MANIFEST" / "pipeline.json")
        if args.command == "resume":
            submit_pipeline(suite, plan)
        elif args.command == "reduce":
            assert_no_active_jobs(suite)
            for stage in CHAIN + EVALUATION:
                if not all(complete(suite, stage, task) for task in tasks_for(suite, stage)):
                    raise RuntimeError(f"Stage {stage} is incomplete; use status/resume before reduce")
            submit_job(suite, plan, "worker", "reduce", indices=[0])
        elif args.command == "worker":
            worker(args, plan)
        elif args.command == "gate":
            gate(args, plan)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[PIPELINE ERROR] {error}", file=sys.stderr)
        sys.exit(1)
