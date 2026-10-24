"""Scheduler-contract tests with fake sbatch/srun; no real jobs are submitted."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rtt_pipeline", ROOT / "rtt_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rtt-launch-test-")
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_suite(self):
        suite = self.base / "suite"
        manifest = suite / "000_MANIFEST"
        source = self.base / "source"
        source.mkdir()
        for name in (pipeline.RUNNER, "rtt_pipeline.py", "run_rtt_iclr2027.sh"):
            (source / name).write_text("# test source\n")
        code = pipeline.snapshot(source, suite)
        for stage in pipeline.CORE:
            with (manifest / f"tasks_{stage}.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["ordinal", "task", "stage"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"ordinal": "0042", "task": "first", "stage": stage})
                if stage == "audit":
                    writer.writerow({"ordinal": "0061", "task": "second", "stage": stage})
        plan = {"code_dir": str(code), "mode": "smoke", "account": "naiss2025-5-643-gpu",
                "cpus": 16, "max_concurrent": 2, "dataset_kind": "synthetic",
                "dataset_root": str(self.base / "data"), "graph_npz": str(self.base / "graph.npz")}
        return suite, plan

    def fake_scheduler(self, calls, active=""):
        def run(command, **kwargs):
            if command[0] == "squeue":
                self.assertIn("--format=%F", command)
                return subprocess.CompletedProcess(command, 0, active, "")
            self.assertEqual(command[0], "sbatch")
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, str(1000 + len(calls)) + "\n", "")
        return run

    def test_shell_syntax(self):
        for name in ("run_rtt_iclr2027.sh", "setup_arrhenius.sh"):
            subprocess.run(["bash", "-n", str(ROOT / name)], check=True)

    def test_dag_waits_for_gates_and_joins_audits_before_reduce(self):
        suite, plan = self.make_suite()
        calls = []
        with mock.patch.object(pipeline.subprocess, "run", self.fake_scheduler(calls)), contextlib.redirect_stdout(io.StringIO()):
            pipeline.submit_pipeline(suite, plan)
        self.assertEqual(len(calls), 2 * len(pipeline.CORE))
        for command in calls:
            self.assertIn("--account=naiss2025-5-643-gpu", command)
            self.assertIn("--gpus=1", command)
            self.assertIn("--ntasks-per-node=1", command)
            self.assertFalse(any(x == "-N" or x.startswith("--nodes") for x in command))
        # Main chain is array -> gate -> next array; evaluations fan out after selection.
        self.assertIn("--dependency=afterok:1001", calls[1])
        self.assertIn("--dependency=afterok:1002", calls[2])
        for index in (12, 14, 16):
            self.assertIn("--dependency=afterok:1012", calls[index])
        self.assertIn("--dependency=afterok:1014:1016:1018", calls[18])
        self.assertIn("--array=0,1%2", calls[12])

    def test_resume_skips_completed_tasks_without_changing_ordinals(self):
        suite, plan = self.make_suite()
        task = pipeline.tasks_for(suite, "audit")[0]
        pipeline.write_json(pipeline.result_dir(suite, "audit", task) / "summary.json", {"status": "COMPLETE"})
        calls = []
        with mock.patch.object(pipeline.subprocess, "run", self.fake_scheduler(calls)), contextlib.redirect_stdout(io.StringIO()):
            pipeline.submit_pipeline(suite, plan)
        self.assertIn("--array=1%2", calls[12])
        self.assertEqual(pipeline.tasks_for(suite, "audit")[1]["ordinal"], "0061")

    def test_failed_scientific_check_blocks_all_new_submissions(self):
        suite, plan = self.make_suite()
        task = pipeline.tasks_for(suite, "audit")[0]
        pipeline.write_json(pipeline.result_dir(suite, "audit", task) / "summary.json",
                            {"status": "COMPLETE", "quick": {"rtt": {"violations": 1}}})
        calls = []
        with mock.patch.object(pipeline.subprocess, "run", self.fake_scheduler(calls)):
            with self.assertRaisesRegex(RuntimeError, "scientific check"):
                pipeline.submit_pipeline(suite, plan)
        self.assertEqual(calls, [])

    def test_active_array_blocks_resume(self):
        suite, plan = self.make_suite()
        pipeline.record_submission(suite, {"job_id": "456", "stage": "audit", "role": "worker"})
        calls = []
        with mock.patch.object(pipeline.subprocess, "run", self.fake_scheduler(calls, active="456\n")):
            with self.assertRaisesRegex(RuntimeError, "active jobs"):
                pipeline.submit_pipeline(suite, plan)
        self.assertEqual(calls, [])

    def test_worker_uses_manifest_index_and_frozen_dataset(self):
        suite, plan = self.make_suite()
        args = type("Args", (), {"suite": str(suite), "stage": "audit"})()
        seen = []

        def fake_run(command, **kwargs):
            seen.append(command)
            result = Path(command[command.index("--run_dir") + 1])
            pipeline.write_json(result / "summary.json", {"status": "COMPLETE"})
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(pipeline.subprocess, "run", fake_run), \
             mock.patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "1"}), \
             contextlib.redirect_stdout(io.StringIO()):
            pipeline.worker(args, plan)
        self.assertEqual(seen[0][seen[0].index("--task") + 1], "second")
        self.assertEqual(seen[0][seen[0].index("--ordinal") + 1], "0061")
        self.assertEqual(seen[0][seen[0].index("--dataset_kind") + 1], "synthetic")
        self.assertIn("--require_cuda", seen[0])

    def test_altered_protocol_blocks_resume(self):
        suite, plan = self.make_suite()
        taskfile = suite / "000_MANIFEST/tasks_audit.tsv"
        pipeline.write_json(suite / "000_MANIFEST/protocol_sha256.json",
                            {"000_MANIFEST/tasks_audit.tsv": pipeline.sha256(taskfile)})
        taskfile.write_text(taskfile.read_text().replace("second", "different_episode"))
        with self.assertRaisesRegex(RuntimeError, "Frozen protocol changed"):
            pipeline.verify_source(suite, plan)

    def test_altered_checkpoint_blocks_resume(self):
        suite, plan = self.make_suite()
        artifacts = suite / "ARTIFACTS"
        artifacts.mkdir()
        checkpoint = artifacts / "incumbent_main.pt"
        checkpoint.write_bytes(b"first-checkpoint")
        pipeline.write_json(artifacts / "incumbent_main.pt.manifest.json",
                            {"artifact": checkpoint.name, "sha256": pipeline.sha256(checkpoint)})
        checkpoint.write_bytes(b"different-checkpoint")
        with self.assertRaisesRegex(RuntimeError, "artifact missing or altered"):
            pipeline.verify_artifacts(suite)

    def test_supplied_config_is_copied_and_used_for_prep_and_init(self):
        suite = self.base / "new-suite"
        source = self.base / "code"
        source.mkdir()
        for name in (pipeline.RUNNER, "rtt_pipeline.py", "run_rtt_iclr2027.sh"):
            (source / name).write_text("# test source\n")
        config = self.base / "overrides.json"
        config.write_text('{"head_episodes": 24}\n')
        args = type("Args", (), {"suite": str(suite), "code_dir": str(source), "mode": "pilot"})()
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            if "--prepare_dataset_only" in command:
                graph = Path(command[command.index("--rtt_graph_npz") + 1])
                graph.write_bytes(b"dataset bytes")
                return
            manifest = suite / "000_MANIFEST"
            pipeline.write_json(manifest / "rtt_config.json", {"config": {"head_episodes": 24}})
            pipeline.write_json(manifest / "task_options.json", {})
            for stage in pipeline.STAGE_DIRS:
                (manifest / f"tasks_{stage}.tsv").write_text("ordinal\ttask\tstage\n")

        env = {"RTT_DATA_ROOT": str(self.base / "data"), "RTT_GRAPH_NPZ": str(self.base / "graph.npz"),
               "RTT_VENV": str(self.base / "venv"), "RTT_BASE": str(self.base),
               "RTT_PYTHON_MODULE": "GPU/Python/example", "RTT_BUILD_MODULE": "GPU/build/example",
               "RTT_CONFIG": str(config)}
        with mock.patch.dict(os.environ, env), mock.patch.object(pipeline, "run_checked", fake_run), \
             mock.patch.object(pipeline, "submit_pipeline"):
            pipeline.initialize(args)
        frozen = suite / "000_MANIFEST/code/submitted_config.json"
        self.assertEqual(frozen.read_bytes(), config.read_bytes())
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command[command.index("--config") + 1], str(frozen))
        plan = pipeline.read_json(suite / "000_MANIFEST/pipeline.json")
        pipeline.verify_source(suite, plan)
        config.write_text('{"head_episodes": 999}\n')
        self.assertEqual(pipeline.read_json(frozen)["head_episodes"], 24)

    def install_mock_commands(self):
        bindir = self.base / "bin"
        bindir.mkdir()
        # sbatch verifies logs exist at submission time, before worker code can run.
        body = """#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
for arg in args:
    if arg.startswith('--output=') or arg.startswith('--error='):
        assert pathlib.Path(arg.split('=',1)[1]).parent.is_dir()
with open(os.environ['MOCK_LOG'], 'a') as f:
    f.write(json.dumps({'name': pathlib.Path(sys.argv[0]).name, 'args': args})+'\\n')
if pathlib.Path(sys.argv[0]).name == 'sbatch': print('12345')
"""
        for name in ("sbatch", "srun", "nvidia-smi"):
            (bindir / name).write_text(body)
            (bindir / name).chmod(0o755)
        return bindir

    def test_login_wrapper_creates_logs_and_preserves_quoted_paths(self):
        bindir = self.install_mock_commands()
        env = dict(os.environ)
        for key in list(env):
            if key.startswith("SLURM_") or key.startswith("RTT_"):
                env.pop(key)
        env.update(PATH=str(bindir) + ":" + os.environ["PATH"], RTT_BASE=str(self.base / "base with spaces"),
                   MOCK_LOG=str(self.base / "calls.jsonl"))
        suite = self.base / "suite with spaces"
        result = subprocess.run(["bash", str(ROOT / "run_rtt_iclr2027.sh"), "pilot", str(suite)],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        call = json.loads((self.base / "calls.jsonl").read_text())
        self.assertEqual(call["name"], "sbatch")
        self.assertEqual(call["args"][-3:], ["_initialize", str(suite), "pilot"])
        self.assertIn("--account=naiss2025-5-643-gpu", call["args"])
        self.assertTrue((suite / "logs").is_dir())

    def test_worker_shell_calls_srun_with_existing_venv(self):
        bindir = self.install_mock_commands()
        venv = self.base / "venv"
        (venv / "bin").mkdir(parents=True)
        python = venv / "bin" / "python"
        python.write_text("#!/bin/bash\nexit 0\n")
        python.chmod(0o755)
        suite = self.base / "job-output"
        env = dict(os.environ)
        env.update(PATH=str(bindir) + ":" + os.environ["PATH"], RTT_BASE=str(self.base),
                   RTT_CODE_DIR=str(ROOT), RTT_VENV=str(venv), MOCK_LOG=str(self.base / "calls.jsonl"),
                   SLURM_JOB_ID="9", SLURM_CPUS_PER_TASK="16")
        # Lmod is a shell function on the cluster; this stand-in only tests dispatch.
        env["BASH_FUNC_module%%"] = "() { return 0;\n}"
        result = subprocess.run(["bash", str(ROOT / "run_rtt_iclr2027.sh"), "_worker", str(suite), "audit"],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in (self.base / "calls.jsonl").read_text().splitlines()]
        command = next(call["args"] for call in calls if call["name"] == "srun")
        self.assertEqual(command[:5], ["--ntasks=1", str(python), "-u", "rtt_pipeline.py", "worker"])
        self.assertEqual(command[-2:], ["--stage", "audit"])


if __name__ == "__main__":
    unittest.main()
