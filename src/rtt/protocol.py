"""Explicit data roles, call sampling, conformal ranks and complete-call summaries.

These utilities describe the reconstructed experiment protocol. They do not turn
the general exact-reference-tail backend into the manuscript's tube backend, or
turn ordinary floating-point endpoint checks into validated arithmetic.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path

import numpy as np

from .datasets.common import DatasetError


POOL_NAMES = ("fit", "inc_train", "inc_valid", "temperature", "calibration",
              "selection", "audit", "reserved", "transfer", "development_unused")
LEAF_POOLS = tuple(name for name in POOL_NAMES if name != "fit")


def _settings(config):
    if not isinstance(config, Mapping):
        raise ValueError("Experiment configuration must be a mapping")
    value = config.get("protocol", config)
    if not isinstance(value, Mapping):
        raise ValueError("protocol must be a mapping")
    return value


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _stream(seed, name):
    digest = hashlib.sha256(f"RTT-protocol-v1|{seed}|{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _group_array(values, n, label):
    values = np.asarray(values, dtype=object)
    if values.shape != (n,):
        raise DatasetError(f"{label} must have exactly one identity per dataset item")
    groups = []
    for value in values:
        if value is None or isinstance(value, (list, dict, tuple)):
            raise DatasetError(f"{label} contains a missing or nonscalar identity")
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            raise DatasetError(f"{label} contains a nonfinite identity")
        text = str(value)
        if not text.strip():
            raise DatasetError(f"{label} contains an empty identity")
        groups.append(text)
    return np.asarray(groups, dtype=str)


def _partition(indices, weights, rng, *, groups=None, description="pool"):
    """Allocate whole groups; each positive-weight role gets at least one group."""
    indices = np.asarray(indices, dtype=np.int64)
    ordered = np.sort(indices)
    if groups is None:
        # Products has millions of test nodes. Avoid one Python/NumPy object per
        # node when an item-level permutation gives exactly the same allocation.
        blocks, count = None, len(ordered)
    else:
        collected = {}
        for index in ordered:
            collected.setdefault(str(groups[index]), []).append(int(index))
        blocks = [np.asarray(collected[key], dtype=np.int64) for key in sorted(collected)]
        count = len(blocks)
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0) or weights.ndim != 1:
        raise ValueError("Partition weights must be a positive finite vector")
    if count < len(weights):
        raise DatasetError(f"{description} has {count} independent groups, but {len(weights)} "
                           "nonempty roles are required; provide more data or a different explicit split")
    target = count * weights / weights.sum()
    allocation = np.maximum(1, np.floor(target).astype(int))
    # Largest-remainder allocation, subject to the nonempty-role constraint.
    while int(allocation.sum()) > count:
        eligible = np.flatnonzero(allocation > 1)
        choice = eligible[np.argmax((allocation - target)[eligible])]
        allocation[choice] -= 1
    while int(allocation.sum()) < count:
        allocation[np.argmax(target - allocation)] += 1
    order = rng.permutation(count)
    output, start = [], 0
    for number in allocation:
        assigned = order[start:start + int(number)]
        output.append(np.sort(ordered[assigned] if blocks is None else
                              np.concatenate([blocks[index] for index in assigned])))
        start += int(number)
    return output


def _assert_group_isolation(pools, groups, description):
    assigned = {}
    for name, indices in pools.items():
        for index in indices:
            identity = str(groups[index])
            previous = assigned.setdefault(identity, name)
            if previous != name:
                raise DatasetError(f"{description} {identity!r} crosses {previous} and {name}")


def _validate_pools(bundle, pools, groups):
    assigned = np.zeros(bundle.n, dtype=np.uint8)
    for name in LEAF_POOLS:
        ids = np.asarray(pools[name], dtype=np.int64)
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= bundle.n):
            raise DatasetError(f"Protocol role {name} contains duplicate or invalid item IDs")
        assigned[ids] += 1
        labels = bundle.y[ids]
        if bundle.multilabel:
            if np.any(~np.isfinite(labels).any(axis=1)):
                raise DatasetError(f"Protocol role {name} contains an entirely unobserved molecular label")
        elif np.any(labels < 0):
            raise DatasetError(f"Protocol role {name} contains unknown labels")
    if assigned.max(initial=0) > 1:
        raise DatasetError("Protocol leaf roles overlap")
    fitting = np.sort(np.concatenate((pools["inc_train"], pools["inc_valid"])))
    if not np.array_equal(fitting, np.sort(pools["fit"])):
        raise DatasetError("inc_train and inc_valid must exactly partition the fit parent pool")
    if groups is not None:
        _assert_group_isolation({name: pools[name] for name in LEAF_POOLS}, groups, "Group identity")


def make_protocol(bundle, config):
    """Build JSON-serializable data-role IDs from genuine dataset allocations.

    ``fit`` intentionally contains ``inc_train`` and ``inc_valid``. All other
    roles are disjoint leaves. No query size or timestamp is invented to make a
    small fixture look like a full experiment.
    """
    bundle.validate()
    settings = _settings(config)
    seed = _integer(settings.get("seed", 1), "protocol.seed")
    query_size = _integer(settings.get("query_size", 4096), "protocol.query_size", minimum=1)
    requested = str(settings.get("split_policy", "official_disjoint"))
    if requested not in {"official_disjoint", "arxiv_temporal", "scaffold_groups"}:
        raise DatasetError(f"Unknown split_policy: {requested!r}")
    name = bundle.name.lower().replace("-", "_")
    policy, notes, groups = requested, [], None
    if requested == "official_disjoint" and name == "ogbn_arxiv" and bundle.year is not None:
        policy = "arxiv_temporal"
        notes.append("Actual ogbn-arxiv publication years activate the registered chronological allocation.")
    if bundle.task == "image":
        if "image_ids" not in bundle.metadata:
            if bundle.image_paths is None:
                raise DatasetError("Image protocol requires metadata.image_ids or image paths")
            identities = [Path(path).name for path in bundle.image_paths]
            notes.append("Image identities default to source filenames; provide image_ids when filenames are not globally unique.")
        else:
            identities = bundle.metadata["image_ids"]
        groups = _group_array(identities, bundle.n, "metadata.image_ids")
        notes.append("All image subdivisions keep every corruption of a base image in one role; fractions apply to identity groups.")
    if policy == "scaffold_groups":
        if name != "ogbg_molpcba":
            raise DatasetError("scaffold_groups is a molecular protocol for ogbg_molpcba")
        if "scaffold_ids" not in bundle.metadata:
            options = config.get("dataset", {}).get("options", {})
            path = options.get("scaffold_path")
            if not path:
                raise DatasetError("scaffold_groups requires real metadata.scaffold_ids or dataset.options.scaffold_path")
            from .datasets.ogbg_molpcba import attach_scaffolds
            attach_scaffolds(bundle, path)
        groups = _group_array(bundle.metadata["scaffold_ids"], bundle.n, "scaffold_ids")
        notes.append("Actual supplied scaffold identities are held together in every subdivision; this explicit reconstruction does not recover the paper's unreleased reserved-scaffold manifest.")
    if groups is not None:
        _assert_group_isolation(bundle.splits, groups, "Underlying identity")
    pools = {key: np.asarray([], dtype=np.int64) for key in POOL_NAMES}
    pools["fit"] = np.sort(bundle.splits["train"])
    pools["inc_train"], pools["inc_valid"] = _partition(
        pools["fit"], (0.9, 0.1), _stream(seed, "incumbent"), groups=groups, description="Training allocation")
    pools["temperature"], pools["calibration"], pools["selection"] = _partition(
        bundle.splits["valid"], (0.2, 0.4, 0.4), _stream(seed, "validation"),
        groups=groups, description="Validation allocation")
    if policy == "arxiv_temporal":
        if name != "ogbn_arxiv" or bundle.year is None:
            raise DatasetError("arxiv_temporal requires ogbn_arxiv and actual publication years")
        year = np.asarray(bundle.year)
        if not np.isfinite(year).all() or np.any(year != year.astype(np.int64)):
            raise DatasetError("arxiv_temporal publication years must be finite integers")
        if np.any(year[bundle.splits["train"]] > 2017) or np.any(year[bundle.splits["valid"]] != 2018):
            raise DatasetError("arxiv_temporal requires official train years <=2017 and validation year 2018")
        test = bundle.splits["test"]
        if np.any(~np.isin(year[test], [2019, 2020])):
            raise DatasetError("arxiv_temporal test allocation must contain only actual 2019/2020 nodes")
        cohort = np.sort(test[year[test] == 2019])
        if len(cohort) < 3:
            raise DatasetError("The 2019 cohort needs at least three labeled nodes for reserved, development_unused and audit")
        requested_reserved = _integer(settings.get("reserved_nodes", 8192), "protocol.reserved_nodes", minimum=1)
        number = min(requested_reserved, len(cohort) // 3)
        order = _stream(seed, "arxiv_2019").permutation(cohort)
        pools["reserved"] = np.sort(order[:number])
        rest = order[number:]
        halfway = len(rest) // 2
        pools["development_unused"] = np.sort(rest[:halfway])
        pools["audit"] = np.sort(rest[halfway:])
        pools["transfer"] = np.sort(test[year[test] == 2020])
        if number < requested_reserved:
            notes.append(f"2019 reserved size reduced from {requested_reserved} to {number} because the cohort contains {len(cohort)} nodes.")
        notes.append("2018 supplies temperature/calibration/selection; 2019 supplies reserved, unused development and final audit; 2020 is transfer only.")
        if not len(pools["transfer"]):
            notes.append("No 2020 nodes were supplied; the transfer experiment is unavailable, not replaced by another cohort.")
    else:
        pools["audit"], pools["reserved"] = _partition(
            bundle.splits["test"], (0.8, 0.2), _stream(seed, "test"),
            groups=groups, description="Test allocation")
        notes.append("Official/source-defined train, validation and test allocations are preserved; within them the reconstruction uses 90/10, 20/40/40 and 80/20 subdivisions.")
        if name == "ogbn_arxiv":
            notes.append("No publication years were supplied: this is an explicit non-temporal fixture allocation, not the real arxiv chronological protocol.")
        if name == "ogbg_molpcba":
            notes.append("The audit/reserved subdivision is custom. Neither pooled episodic AP nor its reserved holdout is the full official-test evaluation or the manuscript's recovered reserved-scaffold-halves protocol.")
    _validate_pools(bundle, pools, groups)
    for role in ("temperature", "calibration", "selection", "audit"):
        if not len(pools[role]):
            raise DatasetError(f"Required protocol role {role} is empty")
    for role in ("calibration", "selection", "audit", "reserved", "transfer"):
        if 0 < len(pools[role]) < query_size:
            notes.append(f"{role} has {len(pools[role])} items < query_size={query_size}; each query uses that entire pool without within-query replacement.")
    return {"pools": {key: value.tolist() for key, value in pools.items()},
            "notes": notes, "seed": seed, "split_policy": policy,
            "requested_split_policy": requested, "query_size": query_size,
            "grouping": "image_id" if bundle.task == "image" else "scaffold_id" if policy == "scaffold_groups" else "item",
            "sizes": {key: len(value) for key, value in pools.items()}}


def sample_ids(protocol, pool, size, seed):
    """Uniform subset without within-call replacement; independent calls may overlap."""
    size = _integer(size, "query size", minimum=1)
    seed = _integer(seed, "query seed")
    if pool not in protocol.get("pools", {}):
        raise DatasetError(f"Unknown protocol pool: {pool}")
    raw = np.asarray(protocol["pools"][pool])
    if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw != raw.astype(np.int64)):
        raise DatasetError(f"Protocol pool {pool} must be a one-dimensional integer array")
    ids = raw.astype(np.int64)
    if not len(ids):
        raise DatasetError(f"Protocol pool {pool} is empty; this experiment cannot be run")
    if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise DatasetError(f"Protocol pool {pool} has negative or duplicate IDs")
    return np.sort(np.random.default_rng(seed).choice(ids, size=min(size, len(ids)), replace=False))


def conformal_radius(scores, delta):
    """Split-conformal rank ceil((n+1)(1-delta)); unattainable rank means infinity."""
    if not isinstance(delta, (int, float, np.integer, np.floating)) or isinstance(delta, (bool, np.bool_)) or not 0 < float(delta) < 1:
        raise ValueError("Conformal delta must be in (0,1)")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or np.isnan(scores).any() or np.isneginf(scores).any() or np.any(scores < 0):
        raise ValueError("Conformal scores must be a one-dimensional nonnegative vector; +infinity is a valid fail-closed score")
    rank = int(math.ceil((len(scores) + 1) * (1 - float(delta))))
    if rank > len(scores):
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def members(config):
    """Preregistered menu in its declared order; no outcome-dependent additions."""
    settings = _settings(config)
    zeta = float(settings.get("zeta", 0.002))
    values = settings.get("member_multipliers", [1.0, 5.0, 10.0, 25.0, 50.0])
    if not math.isfinite(zeta) or zeta < 0:
        raise ValueError("protocol.zeta must be finite and nonnegative")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("member_multipliers must be a nonempty ordered list")
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("member_multipliers must be numbers, not booleans")
    multipliers = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0 for value in multipliers) or sorted(set(multipliers)) != multipliers:
        raise ValueError("member_multipliers must be strictly increasing, positive and finite")
    return [{"name": f"m{index + 1}", "budget": value * zeta} for index, value in enumerate(multipliers)]


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _endpoint(record, down):
    """Check the evidence for the output actually returned, including fallback."""
    checked = record.get("endpoint_checked") is True or record.get("checker_passed_after_fallback") is True
    if not checked:
        return False, None
    values = {key: _number(record.get(key)) for key in
              ("H_plus", "H_row", "call_damage_bound", "row_damage_max")}
    if any(value is None for value in values.values()):
        return False, "endpoint_checked record lacks finite H_plus/H_row/call_damage_bound/row_damage_max"
    if values["H_plus"] < 0 or values["H_row"] < 0:
        return False, "endpoint budget is negative"
    if values["call_damage_bound"] > values["H_plus"] or values["row_damage_max"] > values["H_row"]:
        return False, "returned endpoint exceeds its declared call or row budget"
    # This tolerance checks consistency between two independently evaluated
    # floating-point loss expressions. It never enlarges H_plus or zeta.
    tolerance = 1e-10 * max(1.0, abs(down), abs(values["call_damage_bound"]))
    if down > values["call_damage_bound"] + tolerance:
        return False, "observed loss damage exceeds the declared all-label endpoint damage"
    return True, None


def _primary_metric(records):
    named = [record for record in records if record.get("primary_metric_name") is not None]
    names = sorted({str(record["primary_metric_name"]) for record in named})
    if not names:
        return None
    if len(names) != 1:
        return {"name": None, "status": "INCONSISTENT_METRIC_NAMES", "names": names}
    pairs = [(_number(record.get("primary_metric_reference")),
              _number(record.get("primary_metric_served"))) for record in named]
    valid = [(reference, served) for reference, served in pairs if reference is not None and served is not None]
    reference = float(np.mean([pair[0] for pair in valid])) if valid else None
    served = float(np.mean([pair[1] for pair in valid])) if valid else None
    return {"name": names[0], "reference": reference, "served": served,
            "change": served - reference if valid else None,
            "n_scored_calls": len(valid), "n_missing_calls": len(records) - len(valid),
            "aggregation": "uniform_mean_of_defined_call_metrics",
            "scope": "episodic descriptive metric; not a full official-dataset evaluation",
            "ap_note": ("A mean of per-call AP values is not pooled AP. Compute pooled AP separately from the saved labels and predictions, and retain its exact population and repeated-item policy."
                        if names[0] in {"ap", "macro_ap", "mean_average_precision", "AP"} else None)}


def summarize_records(records, config, expected_count=None):
    """Summarize complete unique calls with equal weight per call.

    Errors, duplicate/missing IDs, nonfinite loss values, inconsistent loss signs,
    mixed experimental phases and an expected-count mismatch suppress all bounds.
    Gain intervals are descriptive paired-t intervals. CP bounds concern the
    frozen call distribution, with no automatic multiple-policy correction.
    """
    settings = _settings(config)
    delta = _number(settings.get("delta", 0.05))
    zeta = _number(settings.get("zeta", 0.002))
    if delta is None or not 0 < delta < 1 or zeta is None or zeta < 0:
        raise ValueError("Summary requires 0<delta<1 and finite nonnegative zeta")
    if expected_count is not None:
        expected_count = _integer(expected_count, "expected_count")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be an ordered sequence of per-call mappings")
    errors, notes, ids, mappings = [], [], [], []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"Record {index} is not a mapping")
            continue
        mappings.append(record)
        identity = record.get("call_id")
        if isinstance(identity, (bool, np.bool_)) or not isinstance(identity, (str, int, np.integer)) or not str(identity).strip():
            errors.append(f"Record {index} lacks a valid call_id")
        else:
            ids.append(str(identity))
    counts = Counter(ids)
    duplicates = sorted(identity for identity, count in counts.items() if count > 1)
    if duplicates:
        errors.append(f"Duplicate call_id values: {duplicates[:10]}" + (" ..." if len(duplicates) > 10 else ""))
    if expected_count is not None and (len(records) != expected_count or len(counts) != expected_count):
        errors.append(f"Expected {expected_count} unique calls; received {len(records)} records and {len(counts)} unique call IDs")
    phases = sorted({str(record.get("phase", record.get("stage"))) for record in mappings
                     if record.get("phase", record.get("stage")) is not None})
    if len(phases) > 1:
        errors.append(f"Cannot pool experimental phases: {phases}")
    policies = sorted({str(record["policy"]) for record in mappings if record.get("policy") is not None})
    if len(policies) > 1:
        errors.append(f"Cannot pool distinct frozen policies: {policies}")
    valid, endpoint_flags = [], []
    for index, record in enumerate(mappings):
        identity = record.get("call_id")
        if identity is None or str(identity) not in counts or counts[str(identity)] != 1:
            continue
        if record.get("status") != "OK":
            errors.append(f"Call {identity}: status is {record.get('status', 'MISSING')}")
            continue
        gain, down = _number(record.get("gain")), _number(record.get("down"))
        if gain is None or down is None:
            errors.append(f"Call {identity}: finite gain and down are required")
            continue
        if not math.isclose(gain, -down, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(f"Call {identity}: gain and down disagree in sign or magnitude")
            continue
        endpoint_ok, endpoint_error = _endpoint(record, down)
        if endpoint_error:
            errors.append(f"Call {identity}: {endpoint_error}")
            continue
        valid.append(record)
        endpoint_flags.append(endpoint_ok)
    complete = bool(valid) and len(valid) == len(records) and not errors
    status = "COMPLETE" if complete else "NOT_RUN" if not records else "INCOMPLETE"
    gains = np.asarray([float(record["gain"]) for record in valid], dtype=np.float64)
    downs = -gains
    n = len(gains)
    events = int(np.sum(downs > zeta))
    mean = float(gains.mean()) if n else None
    lo = hi = cp = betting_upper = betting_lower = None
    deterministic = complete and all(endpoint_flags) and all(
        float(record["H_plus"]) <= zeta and float(record["down"]) <= zeta for record in valid)
    if complete:
        from scipy.stats import beta, t
        cp = 1.0 if events == n else float(beta.ppf(1 - delta, events + 1, n - events))
        if n >= 2:
            half = float(t.ppf(1 - delta / 2, n - 1) * gains.std(ddof=1) / math.sqrt(n))
            lo, hi = mean - half, mean + half
        else:
            notes.append("One call has no estimated sampling variance; a gain interval is unavailable.")
        if not all(endpoint_flags):
            notes.append("Some calls lack final endpoint evidence; empirical CP remains descriptive of the complete sample, but no budget-derived tolerance or betting claim is emitted.")
        elif settings.get("betting", True):
            budgets = [float(record["H_plus"]) for record in valid]
            declared = [0.0] + [row["budget"] for row in members(config)]
            configured = _number(config.get("adaptation", {}).get("H_plus"))
            if configured is not None and configured >= 0:
                declared.append(configured)
            # A constant selected budget is permitted only if it was registered.
            # Otherwise use the predeclared maximum, never a data-fitted range.
            same = all(value == budgets[0] for value in budgets)
            bound = budgets[0] if same and budgets[0] in declared else max(declared)
            if any(value > bound for value in budgets) or np.any(downs > bound):
                notes.append("Recorded call budgets/damages exceed the preregistered one-sided bound; betting inference is unavailable.")
            else:
                from .core.statistics import betting_ucb
                betting_upper = float(betting_ucb(downs, bound, delta, mode="horizon_free"))
                betting_lower = -betting_upper
        else:
            notes.append("Betting inference is disabled by configuration.")
    if not complete:
        notes.append("Incomplete or ambiguous call records cannot be dropped to obtain an inference sample; all confidence and tolerance bounds are blocked.")
    if expected_count is None:
        notes.append("No external expected_count was supplied; completeness is checked only against the records present.")
    numerical = sorted({str(record["numerical_status"]) for record in valid if record.get("numerical_status") is not None})
    return {"status": status, "n": n, "n_attempted": len(records), "n_failed": len(records) - n,
            "n_unique_calls": len(counts), "n_expected": expected_count,
            "errors": errors, "notes": notes, "phase": phases[0] if len(phases) == 1 else None,
            "policy": policies[0] if len(policies) == 1 else None,
            "weighting": "uniform_by_call", "delta": delta, "zeta": zeta,
            "mean_gain": mean, "ci_lo": lo, "ci_hi": hi,
            "gain_interval_scope": "descriptive paired-t interval; not a no-regression certificate",
            "N_zeta": events, "regression_rate": events / n if n else None,
            "u": cp, "cp_upper": cp, "betting_ucb": betting_upper, "betting_lb": betting_lower,
            "deterministic_tolerance": bool(deterministic),
            "deterministic_regression_upper": 0.0 if deterministic else None,
            "endpoint_checked_calls": int(sum(endpoint_flags)),
            "bound_scope": "one frozen policy and one registered call distribution; no automatic multiplicity adjustment",
            "numerical_status": numerical,
            "assurance_scope": "research floating-point output checks; no validated-arithmetic or tube-runtime claim",
            "modes": dict(Counter(str(record.get("mode", "UNKNOWN")) for record in valid)),
            "primary_metric": _primary_metric(valid)}
