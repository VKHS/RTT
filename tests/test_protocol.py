"""Protocol tests exercise leakage, finite-sample ranks and failed-audit gates."""
import json
import math

import numpy as np
import pytest

from rtt.datasets import DatasetBundle, DatasetError, GraphSample
from rtt.datasets.ogbg_molpcba import attach_scaffolds
from rtt.protocol import (LEAF_POOLS, conformal_radius, make_protocol, members,
                          sample_ids, summarize_records)


def config(**values):
    return {"protocol": {"seed": 7, "query_size": 4, "zeta": 0.002,
                          "delta": 0.05, "member_multipliers": [1, 5, 10, 25, 50],
                          **values}, "adaptation": {"H_plus": 0.05}}


def tabular(n=96):
    return DatasetBundle("acs_income", "tabular", np.arange(n) % 2,
                         {"train": np.arange(n // 2), "valid": np.arange(n // 2, 3 * n // 4),
                          "test": np.arange(3 * n // 4, n)}, 2,
                         x=np.zeros((n, 3), dtype=np.float32)).validate()


def molecule():
    bundle = tabular()
    bundle.name, bundle.task, bundle.num_classes = "ogbg_molpcba", "graph", 128
    bundle.y = np.full((96, 128), np.nan, dtype=np.float32)
    bundle.y[:, 0] = np.arange(96) % 2
    bundle.x = None
    sample = GraphSample(np.zeros((1, 9), dtype=np.int64), np.empty((2, 0), dtype=np.int64),
                         np.empty((0, 3), dtype=np.int64))
    bundle.graphs = [sample] * 96
    bundle.metadata["scaffold_ids"] = [f"scaffold_{i // 2}" for i in range(96)]
    return bundle.validate()


def record(index, gain=0.001, budget=0.05, **changes):
    result = {"call_id": f"audit:{index}", "status": "OK", "phase": "audit",
              "gain": gain, "down": -gain, "H_plus": budget, "H_row": 1.0,
              "call_damage_bound": max(0.0, -gain), "row_damage_max": max(0.0, -gain),
              "endpoint_checked": True, "mode": "R1", "numerical_status": "research_float_checks"}
    result.update(changes)
    return result


def test_official_allocations_are_partitioned_once_and_serializable():
    bundle = tabular()
    protocol = make_protocol(bundle, config())
    assert protocol == make_protocol(bundle, config())
    json.dumps(protocol, allow_nan=False)
    pools = protocol["pools"]
    assert len(pools["inc_train"]) == 43
    assert len(pools["inc_valid"]) == 5
    assert [len(pools[key]) for key in ("temperature", "calibration", "selection")] == [5, 10, 9]
    assert len(pools["audit"]) == 19 and len(pools["reserved"]) == 5
    assert sorted(pools["inc_train"] + pools["inc_valid"]) == pools["fit"]
    leaves = [item for key in LEAF_POOLS for item in pools[key]]
    assert len(leaves) == len(set(leaves)) == bundle.n
    assert pools["transfer"] == []


def test_arxiv_uses_genuine_years_and_records_unused_cohort():
    bundle = tabular()
    bundle.name, bundle.task = "ogbn_arxiv", "node"
    bundle.edge_index = np.empty((2, 0), dtype=np.int64)
    bundle.splits = {"train": np.arange(36), "valid": np.arange(36, 60), "test": np.arange(60, 96)}
    bundle.year = np.r_[np.repeat(2017, 36), np.repeat(2018, 24), np.repeat(2019, 24), np.repeat(2020, 12)]
    protocol = make_protocol(bundle, config())
    assert protocol["split_policy"] == "arxiv_temporal"
    assert protocol["sizes"]["reserved"] == 8
    assert protocol["sizes"]["audit"] == protocol["sizes"]["development_unused"] == 8
    assert np.all(bundle.year[protocol["pools"]["transfer"]] == 2020)
    for key in ("temperature", "calibration", "selection"):
        assert np.all(bundle.year[protocol["pools"][key]] == 2018)
    bundle.year[36] = 2019
    with pytest.raises(DatasetError, match="validation year 2018"):
        make_protocol(bundle, config())


def test_no_fabricated_years_and_no_silent_query_size_change():
    bundle = tabular()
    bundle.name, bundle.task = "ogbn_arxiv", "node"
    bundle.edge_index = np.empty((2, 0), dtype=np.int64)
    protocol = make_protocol(bundle, {"protocol": {"seed": 1}})
    assert protocol["split_policy"] == "official_disjoint"
    assert protocol["query_size"] == 4096
    assert any("non-temporal" in note for note in protocol["notes"])
    with pytest.raises(DatasetError, match="actual publication years"):
        make_protocol(bundle, config(split_policy="arxiv_temporal"))


def test_image_subdivisions_keep_base_identity_together(tmp_path):
    paths = []
    for index in range(24):
        path = tmp_path / f"image_{index}.jpg"
        path.write_bytes(b"fixture-only; protocol checks identity, not decoding")
        paths.append(str(path))
    ids = [f"base_{index // 2}" for index in range(24)]
    bundle = DatasetBundle("imagenet_c", "image", np.arange(24) // 2 % 2,
                           {"train": np.arange(6), "valid": np.arange(6, 12), "test": np.arange(12, 24)},
                           2, image_paths=paths, metadata={"image_ids": ids}).validate()
    protocol = make_protocol(bundle, config())
    assert protocol["grouping"] == "image_id"
    assert all(protocol["sizes"][key] == 2 for key in ("temperature", "calibration", "selection"))
    assignments = {}
    for role in LEAF_POOLS:
        for index in protocol["pools"][role]:
            assert assignments.setdefault(ids[index], role) == role
    bundle.splits["valid"] = np.array([6, 7, 8, 9])
    with pytest.raises(DatasetError, match="2 independent groups"):
        make_protocol(bundle, config())


def test_unknown_and_insufficient_label_pools_fail_clearly():
    bundle = tabular()
    bundle.y[0] = -1
    with pytest.raises(DatasetError, match="unknown labels"):
        make_protocol(bundle, config())
    bundle = tabular(8)
    with pytest.raises(DatasetError, match="Validation allocation has 2"):
        make_protocol(bundle, config())


def test_scaffold_policy_requires_real_manifest_and_preserves_groups(tmp_path):
    bundle = molecule()
    protocol = make_protocol(bundle, config(split_policy="scaffold_groups"))
    assignments = {}
    for role in LEAF_POOLS:
        for index in protocol["pools"][role]:
            group = bundle.metadata["scaffold_ids"][index]
            assert assignments.setdefault(group, role) == role
    del bundle.metadata["scaffold_ids"]
    with pytest.raises(DatasetError, match="requires real"):
        make_protocol(bundle, config(split_policy="scaffold_groups"))
    path = tmp_path / "scaffolds.csv"
    path.write_text("graph_id,scaffold_id\n" + "".join(f"{i},scaffold_{i // 2}\n" for i in range(96)))
    cfg = config(split_policy="scaffold_groups")
    cfg["dataset"] = {"options": {"scaffold_path": str(path)}}
    protocol = make_protocol(bundle, cfg)
    assert bundle.metadata["scaffold_manifest_sha256"]
    assert protocol["grouping"] == "scaffold_id"
    bundle.metadata["scaffold_ids"][48] = bundle.metadata["scaffold_ids"][0]
    with pytest.raises(DatasetError, match="crosses train and valid"):
        make_protocol(bundle, cfg)


def test_scaffold_manifest_rejects_missing_duplicate_or_invented_index(tmp_path):
    bundle = molecule()
    path = tmp_path / "scaffolds.json"
    path.write_text(json.dumps(["one"] * 95))
    with pytest.raises(DatasetError, match="cover every"):
        attach_scaffolds(bundle, path)
    path = tmp_path / "scaffolds.csv"
    path.write_text("graph_id,scaffold_id\n0,a\n0,b\n")
    with pytest.raises(DatasetError, match="Duplicate"):
        attach_scaffolds(bundle, path)


def test_query_sampling_has_no_replacement_and_never_uses_other_pool():
    protocol = make_protocol(tabular(), config())
    sample = sample_ids(protocol, "audit", 10, 3)
    assert len(sample) == len(set(sample)) == 10
    assert set(sample) <= set(protocol["pools"]["audit"])
    assert np.array_equal(sample, sample_ids(protocol, "audit", 10, 3))
    assert len(sample_ids(protocol, "audit", 999, 3)) == protocol["sizes"]["audit"]
    with pytest.raises(DatasetError, match="empty"):
        sample_ids(protocol, "transfer", 4, 3)


def test_conformal_radius_is_conservative_exact_order_statistic():
    assert math.isinf(conformal_radius([], 0.1))
    assert math.isinf(conformal_radius([1, 2, 3, 4], 0.1))
    assert conformal_radius(np.arange(1, 10), 0.1) == 9
    assert conformal_radius(np.arange(1, 20), 0.2) == 16
    assert math.isinf(conformal_radius([1, 2, float("inf")], 0.25))
    with pytest.raises(ValueError, match="nonnegative"):
        conformal_radius([1, float("nan")], 0.1)
    with pytest.raises(ValueError, match="delta"):
        conformal_radius([1], 0)


def test_members_are_fixed_order_and_budget_multiples():
    menu = members(config())
    assert menu[0] == {"name": "m1", "budget": 0.002}
    assert menu[-1] == {"name": "m5", "budget": 0.1}
    with pytest.raises(ValueError, match="strictly increasing"):
        members(config(member_multipliers=[5, 1]))


def test_complete_summary_cp_bound_and_call_weighting():
    records = [record(0, gain=0.001, items=1), record(1, gain=0.009, items=999)]
    summary = summarize_records(records, config(), expected_count=2)
    assert summary["status"] == "COMPLETE"
    assert summary["mean_gain"] == pytest.approx(0.005)
    assert summary["u"] == pytest.approx(1 - 0.05 ** (1 / 2))
    assert summary["weighting"] == "uniform_by_call"
    assert summary["ci_lo"] < summary["mean_gain"] < summary["ci_hi"]
    assert summary["betting_ucb"] is not None
    assert summary["phase"] == "audit"
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("problem", ["duplicate", "error", "missing_id", "missing_call", "nonfinite", "wrong_sign", "mixed_phase"])
def test_incomplete_audit_blocks_all_bounds(problem):
    records = [record(0), record(1)]
    if problem == "duplicate":
        records[1]["call_id"] = records[0]["call_id"]
    elif problem == "error":
        records[1]["status"] = "ERROR"
    elif problem == "missing_id":
        del records[1]["call_id"]
    elif problem == "missing_call":
        records.pop()
    elif problem == "nonfinite":
        records[1]["gain"] = float("nan")
    elif problem == "wrong_sign":
        records[1]["down"] = 0.001
    elif problem == "mixed_phase":
        records[1]["phase"] = "selection"
    summary = summarize_records(records, config(), expected_count=2)
    assert summary["status"] == "INCOMPLETE"
    assert summary["errors"]
    for key in ("u", "cp_upper", "ci_lo", "ci_hi", "betting_ucb", "betting_lb", "deterministic_regression_upper"):
        assert summary[key] is None
    assert not summary["deterministic_tolerance"]


def test_deterministic_tolerance_needs_budget_and_final_endpoint_evidence():
    records = [record(index, budget=0.002) for index in range(3)]
    summary = summarize_records(records, config(), 3)
    assert summary["deterministic_tolerance"]
    assert summary["deterministic_regression_upper"] == 0
    # A rejected proposal followed by a correctly checked incumbent is valid.
    records[1].update(mode="F", checker_passed_before_fallback=False, gain=0.0, down=0.0)
    assert summarize_records(records, config(), 3)["deterministic_tolerance"]
    records[1]["endpoint_checked"] = False
    summary = summarize_records(records, config(), 3)
    assert summary["status"] == "COMPLETE" and summary["u"] is not None
    assert not summary["deterministic_tolerance"] and summary["betting_ucb"] is None


def test_endpoint_contradiction_fails_closed_and_strict_regression_threshold():
    bad = record(0, gain=-0.003, budget=0.002)
    assert summarize_records([bad], config(), 1)["status"] == "INCOMPLETE"
    summary = summarize_records([record(0, gain=-0.002), record(1, gain=-0.00200000001)], config(), 2)
    assert summary["N_zeta"] == 1


def test_primary_metric_scope_and_missing_binary_tasks_are_explicit():
    rows = [record(0, primary_metric_name="macro_ap", primary_metric_reference=0.2, primary_metric_served=0.3),
            record(1, primary_metric_name="macro_ap", primary_metric_reference=None, primary_metric_served=None)]
    metric = summarize_records(rows, config(), 2)["primary_metric"]
    assert metric["reference"] == 0.2 and metric["served"] == 0.3
    assert metric["n_scored_calls"] == metric["n_missing_calls"] == 1
    assert "not pooled AP" in metric["ap_note"]


def test_empty_summary_cannot_claim_any_bound():
    summary = summarize_records([], config(), expected_count=2)
    assert summary["status"] == "NOT_RUN"
    assert summary["mean_gain"] is None and summary["u"] is None
    assert not summary["deterministic_tolerance"]
