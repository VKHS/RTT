"""Slurm orchestration checks using local mocks; no cluster jobs are submitted."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def hpc():
    spec = importlib.util.spec_from_file_location("rtt_hpc_test", ROOT / "src/rtt/hpc.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path):
    source = tmp_path / "source with spaces"
    (source / "src/rtt").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "experiments/ogbn_arxiv").mkdir(parents=True)
    shutil.copy2(ROOT / "src/rtt/hpc.py", source / "src/rtt/hpc.py")
    shutil.copy2(ROOT / "scripts/arrhenius.sh", source / "scripts/arrhenius.sh")
    (source / "src/rtt/__init__.py").write_text("")
    (source / "src/rtt/cli.py").write_text('''import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--run-dir');p.add_argument('--shard-index',type=int,default=0)
a,rest=p.parse_known_args();root=Path(a.run_dir)
from .hpc import expected_artifacts
for name in expected_artifacts(a.stage,a.shard_index):
    out=root/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'stage':a.stage,'args':rest}))
''')
    (source / "experiments/ogbn_arxiv/config.json").write_text('{"dataset":"ogbn_arxiv"}\n')
    return source


@pytest.fixture
def slurm(hpc, monkeypatch):
    calls = []
    active = []
    original = hpc.subprocess.run

    def mocked(command, *args, **kwargs):
        if command[0] == "sbatch":
            calls.append(command)
            return SimpleNamespace(stdout=f"{1000 + len(calls)};cluster\n", returncode=0)
        if command[0] == "squeue":
            return SimpleNamespace(stdout="\n".join(active), returncode=0)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(hpc.subprocess, "run", mocked)
    return SimpleNamespace(calls=calls, active=active)


def initialize(hpc, repo, tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("RTT_AUDIT_SHARDS", "3")
    monkeypatch.setenv("RTT_MAX_CONCURRENT", "2")
    args = argparse.Namespace(run_dir=str(tmp_path / "run with spaces"), repo=str(repo),
                              config=str(repo / "experiments/ogbn_arxiv/config.json"),
                              dataset="ogbn_arxiv", profile="full", prepare_only=False,
                              data_root=str(tmp_path / "data with spaces"), download=False)
    for key, value in kwargs.items():
        setattr(args, key, value)
    hpc.initialize(args)
    run = Path(args.run_dir)
    return run, hpc.read_json(run / "orchestration/plan.json")


def mark_complete(hpc, run, plan, stage, index=0):
    artifacts = {}
    for name in hpc.expected_artifacts(stage, index):
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stage": stage, "index": index}))
        artifacts[name] = hpc.sha256(path)
    hpc.write_json(hpc.receipt_path(run, stage, index), {
        "status": "COMPLETE", "config_sha256": plan["config_sha256"], "artifacts": artifacts})


def test_preserved_allocation_and_dependency_chain(hpc, repo, tmp_path, monkeypatch, slurm):
    run, plan = initialize(hpc, repo, tmp_path, monkeypatch)
    assert plan["account"] == "naiss2025-5-643-gpu"
    assert plan["storage_project"] == "naiss2025-6-428"
    assert len(slurm.calls) == 5
    for index, command in enumerate(slurm.calls):
        assert "--gpus=1" in command and "--ntasks=1" in command
        assert not any(arg == "-N" or arg.startswith("--nodes") for arg in command)
        dependency = [arg for arg in command if arg.startswith("--dependency")]
        assert dependency == ([] if index == 0 else [f"--dependency=afterok:{1000+index}"])
    assert "--array=0,1,2%2" in slurm.calls[3]
    assert (run / "logs").is_dir()
    assert (Path(plan["code_dir"]) / "src/rtt/cli.py").is_file()
    assert (Path(plan["code_dir"]) / "experiments/ogbn_arxiv/config.json").is_file()


def test_resume_preserves_shard_ids_and_completed_stages(hpc, repo, tmp_path, monkeypatch, slurm):
    run, plan = initialize(hpc, repo, tmp_path, monkeypatch)
    for stage in ("train", "calibrate", "select"):
        mark_complete(hpc, run, plan, stage)
    mark_complete(hpc, run, plan, "audit", 1)
    slurm.calls.clear()
    hpc.submit_pipeline(run, plan)
    assert len(slurm.calls) == 2
    assert "--array=0,2%2" in slurm.calls[0]
    assert "--dependency=afterok:1001" in slurm.calls[1]


def test_active_jobs_block_resume(hpc, repo, tmp_path, monkeypatch, slurm):
    run, plan = initialize(hpc, repo, tmp_path, monkeypatch)
    slurm.active.append("1004")
    previous = len(slurm.calls)
    with pytest.raises(RuntimeError, match="active jobs"):
        hpc.submit_pipeline(run, plan)
    assert len(slurm.calls) == previous


@pytest.mark.parametrize("target", ["source", "extra_source", "config", "plan", "artifact"])
def test_frozen_integrity_before_resubmission(hpc, repo, tmp_path, monkeypatch, slurm, target):
    run, plan = initialize(hpc, repo, tmp_path, monkeypatch)
    mark_complete(hpc, run, plan, "train")
    path = {"source": Path(plan["code_dir"]) / "src/rtt/cli.py",
            "extra_source": Path(plan["code_dir"]) / "src/extra_module.py",
            "config": Path(plan["config"]), "plan": run / "orchestration/plan.json",
            "artifact": run / "checkpoints/model.pt"}[target]
    path.write_text((path.read_text() if path.exists() else "") + "\n# changed\n")
    previous = len(slurm.calls)
    with pytest.raises(RuntimeError, match="changed|Invalid|disappeared"):
        hpc.submit_pipeline(run, plan)
    assert len(slurm.calls) == previous


def test_scientific_failure_is_not_resampled(hpc, repo, tmp_path, monkeypatch, slurm):
    run, plan = initialize(hpc, repo, tmp_path, monkeypatch)
    hpc.write_json(hpc.failure_path(run, "audit", 2), {"retriable": False, "error": "bound violation"})
    previous = len(slurm.calls)
    with pytest.raises(RuntimeError, match="Non-retriable"):
        hpc.submit_pipeline(run, plan)
    assert len(slurm.calls) == previous


def test_worker_runs_frozen_code_and_seals_exact_outputs(hpc, repo, tmp_path, monkeypatch, slurm):
    run, plan = initialize(hpc, repo, tmp_path, monkeypatch, prepare_only=True, download=True)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    hpc.worker(argparse.Namespace(run_dir=str(run), stage="prepare"))
    assert hpc.complete(run, plan, "prepare")
    output = hpc.read_json(run / "data_manifest.json")
    assert "--download" in output["args"]
    assert hpc.read_json(hpc.receipt_path(run, "prepare"))["artifacts"].keys() == {"data_manifest.json"}


def test_audit_worker_requires_all_prerequisites_and_owns_its_shard(hpc, repo, tmp_path, monkeypatch, slurm):
    run, plan = initialize(hpc, repo, tmp_path, monkeypatch)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    args = argparse.Namespace(run_dir=str(run), stage="audit")
    with pytest.raises(RuntimeError, match="Prerequisite"):
        hpc.worker(args)
    for stage in ("train", "calibrate", "select"):
        mark_complete(hpc, run, plan, stage)
    hpc.worker(args)
    assert hpc.complete(run, plan, "audit", 2)
    assert not hpc.receipt_path(run, "audit", 0).exists()
    assert not (run / "records/audit_shard_0000.jsonl").exists()


def test_smoke_forbids_download_and_freezes_resource_overrides(hpc, repo, tmp_path, monkeypatch, slurm):
    monkeypatch.setenv("RTT_TIME_TRAIN", "05:00:00")
    monkeypatch.setenv("RTT_MEM_TRAIN", "80G")
    run, plan = initialize(hpc, repo, tmp_path, monkeypatch, profile="smoke", download=True)
    assert plan["download"] is False
    assert "--download" not in hpc.numerical_command(run, plan, "train")
    monkeypatch.setenv("RTT_TIME_TRAIN", "09:00:00")
    assert hpc.stage_resources(plan, "train")["walltime"] == "05:00:00"
    assert "--mem=80G" in slurm.calls[0]


def test_shell_submission_quotes_paths_and_creates_logs(repo, tmp_path):
    mock = tmp_path / "bin"
    mock.mkdir()
    log = tmp_path / "sbatch_args.jsonl"
    sbatch = mock / "sbatch"
    sbatch.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['SBATCH_TEST_LOG'])
rows=p.read_text().splitlines() if p.exists() else []
with p.open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
print(2000+len(rows))
''')
    sbatch.chmod(0o755)
    environment = os.environ.copy()
    environment.update({"PATH": str(mock) + os.pathsep + environment["PATH"],
                        "RTT_BASE": str(tmp_path / "project with spaces"),
                        "RTT_CODE_DIR": str(repo), "SBATCH_TEST_LOG": str(log),
                        "RTT_AUDIT_SHARDS": "2"})
    environment.pop("SLURM_JOB_ID", None)
    run = tmp_path / "output with spaces"
    result = subprocess.run(["bash", str(repo / "scripts/arrhenius.sh"), "ogbn_arxiv", "smoke", str(run)],
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands) == 5 and (run / "logs").is_dir()
    assert commands[0][-2] == str(run)
    assert any(arg.startswith("--output=" + str(run / "logs")) for arg in commands[0])
