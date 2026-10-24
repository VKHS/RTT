"""Frozen-run integrity across fitting, calibration, selection and sharded audit.

The small arxiv fixture is synthetic and trains once for this module. Each test
receives an independent copy, so corruptions and committed audits cannot affect
the next test or be mistaken for benchmark evidence.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rtt.configuration import resolve_config, validate_config
from rtt.experiment import Experiment
from rtt.runtime import configure, read_json, save_checkpoint, sha256, write_json


@pytest.fixture(scope="module")
def frozen_source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("frozen_arxiv_source")
    source = Path(__file__).resolve().parents[1] / "experiments" / "ogbn_arxiv" / "config.json"
    config = resolve_config(source, profile="smoke", data_root=directory / "unused_real_data")
    # One preregistered member is sufficient to exercise the complete lifecycle.
    config["protocol"]["member_multipliers"] = [1.0]
    validate_config(config)
    previous_threads = torch.get_num_threads()
    previous_python, previous_numpy = random.getstate(), np.random.get_state()
    previous_torch = torch.random.get_rng_state()
    configure(config["protocol"]["seed"], threads=1)
    try:
        experiment = Experiment(config, directory / "run", device="cpu")
        training = experiment.train()
        calibration = experiment.calibrate()
        selection = experiment.select()
        assert training["status"] == calibration["status"] == "COMPLETE"
        assert config["protocol"]["audit_calls"] == 4
        assert read_json(experiment.directory / "manifest.json")["synthetic"] is True
        # Four calibration calls do not establish a finite conformal radius at
        # this coverage; two selection calls must not produce a release claim.
        assert selection["selected"] is None
        yield SimpleNamespace(directory=experiment.directory, config=config)
    finally:
        torch.set_num_threads(previous_threads)
        random.setstate(previous_python)
        np.random.set_state(previous_numpy)
        torch.random.set_rng_state(previous_torch)


@pytest.fixture
def frozen_run(frozen_source, tmp_path):
    directory = tmp_path / "isolated_run"
    shutil.copytree(frozen_source.directory, directory)
    return Experiment(copy.deepcopy(frozen_source.config), directory, device="cpu")


def test_two_shards_refuse_partial_reduction_and_resampling_then_report_four_calls(frozen_run):
    experiment = frozen_run
    first = experiment.audit(shard_index=0, shards=2)
    assert first["status"] == "COMPLETE"
    assert first["indices"] == [0, 2] and first["count"] == 2
    with pytest.raises(ValueError, match="incomplete.*partial sample"):
        experiment.report()
    for name in ("summary.json", "metrics.csv", "REPORT.md"):
        assert not (experiment.directory / name).exists(), "A partial sample produced a completed report"

    records_path = experiment.directory / "records" / "audit_shard_0000.jsonl"
    metadata_path = records_path.with_suffix(".meta.json")
    committed_records, committed_metadata = records_path.read_bytes(), metadata_path.read_bytes()
    with pytest.raises(ValueError, match="already committed.*do not resample"):
        experiment.audit(shard_index=0, shards=2)
    assert records_path.read_bytes() == committed_records
    assert metadata_path.read_bytes() == committed_metadata

    second = experiment.audit(shard_index=1, shards=2)
    assert second["indices"] == [1, 3] and second["count"] == 2
    # A fresh process-equivalent object must accept the complete persisted run.
    reloaded = Experiment(copy.deepcopy(experiment.config), experiment.directory)
    summary = reloaded.report()
    assert summary["status"] == "COMPLETE"
    assert summary["n"] == summary["n_expected"] == summary["n_unique_calls"] == 4
    assert summary["n_failed"] == 0 and summary["endpoint_checked_calls"] == 4
    assert sum(summary["release_counts"].values()) == 4
    assert summary["synthetic"] is True and summary["statistical_release"] is False

    records = []
    for path in sorted((experiment.directory / "records").glob("audit_shard_*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    expected_ids = {f"ogbn_arxiv:audit:audit:{index:08d}" for index in range(4)}
    assert {row["call_id"] for row in records} == expected_ids and len(records) == 4
    assert all(row["phase"] == "audit" and row["status"] == "OK" for row in records)
    with (experiment.directory / "metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert {row["call_id"] for row in metrics} == expected_ids and len(metrics) == 4
    assert read_json(experiment.directory / "summary.json")["n"] == 4
    assert "Completed calls: 4." in (experiment.directory / "REPORT.md").read_text()


@pytest.mark.parametrize("filename,manifest_key", [
    ("model.pt", "model_checkpoint_sha256"),
    ("adaptation.pt", "adaptation_checkpoint_sha256"),
])
def test_training_manifest_rejects_changed_checkpoint_even_with_fresh_checksum_sidecar(
        frozen_run, filename, manifest_key):
    experiment = frozen_run
    experiment.load()  # Cached in-memory modules must not bypass later integrity checks.
    path = experiment.directory / "checkpoints" / filename
    manifest = read_json(experiment.directory / "manifest.json")
    original_hash = manifest[manifest_key]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    key = next(name for name, value in payload["state_dict"].items()
               if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel())
    replacement = payload["state_dict"][key].clone()
    replacement.reshape(-1)[0] += 0.125
    payload["state_dict"][key] = replacement
    # This writes a valid new checkpoint plus a matching local sidecar. The
    # earlier training manifest must still reject the changed frozen content.
    save_checkpoint(path, payload)
    changed_hash = sha256(path)
    assert changed_hash != original_hash
    assert read_json(path.with_suffix(path.suffix + ".sha256.json"))["sha256"] == changed_hash
    assert read_json(experiment.directory / "manifest.json")[manifest_key] == original_hash
    with pytest.raises(ValueError, match="frozen trained checkpoint changed") as error:
        experiment.load()
    assert filename in str(error.value)


@pytest.mark.parametrize("corruption", ["selection_pointer", "calibration_contents"])
def test_selected_policy_rejects_inconsistent_calibration_hash_before_any_audit(frozen_run, corruption):
    experiment = frozen_run
    selection_path = experiment.directory / "selection.json"
    calibration_path = experiment.directory / "calibration.json"
    selection = read_json(selection_path)
    assert selection["calibration_sha256"] == sha256(calibration_path)
    if corruption == "selection_pointer":
        selection["calibration_sha256"] = "0" * 64
        write_json(selection_path, selection)
    else:
        calibration = read_json(calibration_path)
        member = next(iter(calibration["members"].values()))
        member["budget"] += 0.001
        write_json(calibration_path, calibration)
        assert sha256(calibration_path) != selection["calibration_sha256"]
    # Even incumbent-only selection must remain bound to its frozen inputs.
    with pytest.raises(ValueError, match="Selection and frozen calibration/configuration differ"):
        experiment.audit(shard_index=0, shards=2)
    assert not (experiment.directory / "records").exists()
    assert not (experiment.directory / "summary.json").exists()
