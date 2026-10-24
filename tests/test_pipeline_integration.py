"""Tiny CPU integration of real CLI stages, without a dataset download.

This test deliberately uses few training/calibration examples. It checks wiring,
artifact persistence and fail-closed reporting, not statistical power or arxiv
performance. Numerical/nonzero-serving behavior has separate focused tests.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def test_tiny_cli_stage_pipeline(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "run_rtt_iclr2027_arrhenius.py"
    # Match Python's script-launch path semantics for sibling modules.
    monkeypatch.syspath_prepend(str(source.parent))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    spec = importlib.util.spec_from_file_location("rtt_pipeline_test_module", source)
    rtt = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rtt
    spec.loader.exec_module(rtt)
    torch.set_num_threads(1)
    config = {
        "T": 4, "open_first": 1, "b_window": 2, "hidden": 8,
        "enc_hidden": 12, "head_hidden": 8, "att_dim": 4, "bank_size": 4,
        "probes_per_sign": 2, "spectral_rank": 2, "spectral_rank_tight": 4,
        "nq": 4, "n_r": 4, "inc_epochs": 2, "inc_eval_every": 1,
        "inc_patience": 2, "proposal_episodes": 2, "head_episodes": 2,
        "placement_episodes": 2, "growth_calls": 2, "crossfit_folds": 2,
        "crossfit_episodes": 2, "selection_pool_calls": 1,
        "audit_N_mixture": 2, "audit_N_component": 1,
        "transfer_N": 1, "dev_N": 1, "reserved_nodes": 8, "node_inspect": 4,
        "call_sizes": [4], "row_budgets": [1.0], "frontier_grid": 6,
        "gauge_angles": 16, "solver_max_newton": 12,
        "budget_sweep_multipliers": [25.0], "planted_gammas": [0.0],
    }
    config_path = tmp_path / "tiny_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    suite_path = tmp_path / "suite"
    common = ["--suite", str(suite_path), "--dataset_kind", "synthetic",
              "--synthetic_nodes", "240", "--synthetic_classes", "3",
              "--synthetic_features", "8", "--synthetic_seed", "53",
              "--cpu", "--threads", "1", "--scale", "preflight",
              "--config", str(config_path), "--breadth", "0", "--swap_heads", "0",
              "--audit_shard", "2", "--audit_component_shard", "1"]
    assert rtt.main(common + ["--init_suite"]) == 0
    suite = rtt.Suite(str(suite_path))
    frozen_config, _ = suite.config()
    assert frozen_config.T == 4 and frozen_config.hidden == 8
    manifests = {stage: {row["task"]: row["ordinal"] for row in rtt.read_task_manifest(
        suite.manifest / f"tasks_{stage}.tsv")} for stage in rtt.STAGES}

    def run(stage, task):
        ordinal = manifests[stage][task]
        code = rtt.main(common + ["--stage", stage, "--task", task, "--ordinal", ordinal])
        run_dir = suite.run_dir(stage, f"{ordinal}_{task}")
        failure = run_dir / "failure_summary.json"
        assert code == 0, failure.read_text() if failure.exists() else (stage, task, code)
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["status"] == "COMPLETE"

    for stage, task in [("incumbent", "main"), ("proposal", "native"),
                        ("phase2", "heads_native"), ("phase2", "correctors_native"),
                        ("phase2", "crossfit"), ("placement", "native")]:
        run(stage, task)
    corrector_info_path = suite.run_dir("phase2", "0002_correctors_native") / "summary.json"
    corrector_info = json.loads(corrector_info_path.read_text())["correctors"]
    assert all(row["training_call_generator"] == "phase2_heads" for row in corrector_info.values())
    assert all(row["steps"] == 2 for row in corrector_info.values())
    for member in rtt.default_library():
        run("calibration", f"native__{member.name}")
    for task in manifests["selection"]:
        run("selection", task)
    assert rtt.main(common + ["--stage_gate", "selection"]) == 0
    assert suite.has("selection.json")
    run("audit", "mixture__00000_00002")
    run("reduce", "reduce")

    audit = rtt.stage_records(suite, "audit")
    scored = [record for record in audit if record.get("kind") != "group"]
    assert scored and all(record["status"] == "OK" for record in scored)
    assert not any(record.get("violation_damage") or record.get("violation_row") for record in scored)
    # A small synthetic run must not silently become the reported arxiv study.
    assert all(record["nq"] == 4 for record in scored)
    assert all("certificate_status" in record for record in scored)
    selected = suite.load("selection.json", kind="json")
    assert selected.get("selected") is None, "One call cannot support the registered release criteria."
    summary = json.loads((suite.stage_dir("reduce") / "summary.json").read_text())
    assert summary.get("release", {}).get("decision") != "RELEASE"

    # The standalone inference entry point restores the exact dataset options,
    # including a non-default synthetic graph, without receiving query labels.
    import predict_rtt
    for member in ("registered", rtt.default_library()[0].name):
        output = tmp_path / ("prediction_" + member)
        predict_rtt.main(["--suite", str(suite_path), "--output", str(output),
                          "--member", member, "--cpu", "--threads", "1"])
        report = json.loads((output / "call.json").read_text())
        assert report["labels_supplied_to_policy"] is False
        assert report["separate_full_incumbent_endpoint"] is False
        assert report["query_count"] == 4
        if member == "registered":
            assert report["policy"] == "incumbent_only_unselected"
            assert report["release_mode"] == "NO_STEP"
        else:
            assert report["selection_status"] == "EXPLICIT_EXPLORATORY_MEMBER"
        with np.load(output / "predictions.npz", allow_pickle=False) as predictions:
            assert predictions["probabilities"].shape == (4, 3)
            assert np.isfinite(predictions["log_prob"]).all()
            np.testing.assert_allclose(predictions["probabilities"].sum(1), 1.0, atol=1e-12)
