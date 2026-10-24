"""Real stage integration on explicit task-shaped synthetic fixtures."""
import json
from pathlib import Path

import numpy as np
import pytest

from rtt.cli import main
from rtt.datasets import DATASETS


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", DATASETS)
def test_separate_dataset_pipeline_and_prediction(dataset, tmp_path):
    config = ROOT / "experiments" / dataset / "config.json"
    run = tmp_path / dataset
    common = ["--config", str(config), "--run-dir", str(run), "--profile", "smoke", "--threads", "1"]
    code = main(["run", *common])
    failure = run / "failures/run_0000.json"
    assert code == 0, failure.read_text() if failure.exists() else code
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["synthetic"] is True
    assert manifest["configuration"]["dataset"]["name"] == dataset
    training = json.loads((run / "training.json").read_text())
    assert training["adaptation"]["heads"]["steps"] == 2
    calibration = json.loads((run / "calibration.json").read_text())
    assert len(calibration["members"]) == 5
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "COMPLETE"
    assert summary["n"] == 4
    assert summary["synthetic"] is True
    assert summary["statistical_release"] is False
    records = [json.loads(row) for row in (run / "records/audit_shard_0000.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert all(row["endpoint_checked"] and row["labels_supplied"] is False for row in records)
    assert all(row["call_damage_bound"] <= row["H_plus"] for row in records)
    assert all(row["row_damage_max"] <= row["H_row"] for row in records)
    output = tmp_path / (dataset + "_prediction")
    assert main(["predict", *common, "--output", str(output)]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["labels_supplied"] is False
    with np.load(output / "predictions.npz", allow_pickle=False) as prediction:
        probabilities = np.exp(prediction["log_probabilities"])
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(-1), 1.0, atol=1e-12)
        if dataset == "ogbg_molpcba":
            assert probabilities.shape == (4, 128, 2)
        else:
            assert probabilities.ndim == 2
    if dataset == "ogbg_molpcba":
        assert 0 < records[0]["task_units"] < 4 * 128
        population = tmp_path / "molecular_population"
        assert main(["evaluate", *common, "--output", str(population), "--batch-size", "8"]) == 0
        metrics = json.loads((population / "population_metrics.json").read_text())
        assert metrics["served"]["ap_tasks_scored"] > 0
        assert 0 <= metrics["served"]["macro_ap"] <= 1


def test_configuration_smoke_never_requests_pretrained_downloads():
    from rtt.configuration import resolve_config
    config = resolve_config(ROOT / "experiments/imagenet_c/config.json", profile="smoke", download=True)
    assert config["model"]["pretrained"] is False
    assert config["model"]["allow_download"] is False
    assert "weights" not in config["model"]
    assert config["dataset"]["download"] is False
