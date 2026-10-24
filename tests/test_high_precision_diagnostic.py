"""Independent Decimal endpoint checks use the checker's actual reference.

The served bound and row bounds are also checked, so correcting the reference
of the mathematical charge cannot hide an understated final release bound.
"""
from __future__ import annotations

import dataclasses
import decimal
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture(scope="module")
def rtt():
    source = Path(__file__).resolve().parents[1] / "run_rtt_iclr2027_arrhenius.py"
    spec = importlib.util.spec_from_file_location("rtt_hp_test_module", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    return module


def test_shared_prefix_and_actual_served_logits_are_enclosed(rtt):
    old_precision = decimal.getcontext().prec
    result = rtt.high_precision_reevaluation(seed=1, n_cert=8, digits=50)
    assert decimal.getcontext().prec == old_precision
    assert result["comparison_tolerance"] == 0.0
    assert result["passed"], result
    assert result["charge_below_ideal_count"] == 0
    assert result["damage_below_served_count"] == 0
    assert result["row_below_served_count"] == 0
    assert result["min_charge_minus_ideal_damage"] >= 0
    assert result["min_damage_bound_minus_served_damage"] >= 0
    assert result["min_row_bound_minus_served_row_damage"] >= 0
    # This fixed CPU seed reproduces the original incorrect-reference failure.
    # It remains visible in the report instead of being suppressed by tolerance.
    assert result["legacy_origin_reference_below_count"] > 0


@pytest.mark.parametrize("bound,count_key", [
    ("charge_total", "charge_below_ideal_count"),
    ("damage_bound", "damage_below_served_count"),
    ("row_bounds", "row_below_served_count"),
])
def test_understated_charge_or_served_bound_is_rejected(rtt, monkeypatch, bound, count_key):
    original = rtt.Checker.certify_once

    def understate(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        zero = torch.zeros_like(result.row_bounds) if bound == "row_bounds" else 0.0
        return dataclasses.replace(result, **{bound: zero})

    monkeypatch.setattr(rtt.Checker, "certify_once", understate)
    result = rtt.high_precision_reevaluation(seed=1, n_cert=1, digits=50)
    assert result["comparison_tolerance"] == 0.0
    assert not result["passed"]
    assert result[count_key] == 1
    assert result["float64_below_exact_count"] == 1
    assert result["min_certified_minus_exact"] < 0
