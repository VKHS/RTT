"""Validated dataset-specific experiment configuration."""
from __future__ import annotations

import copy
import math
from pathlib import Path

from .runtime import read_json


DEFAULT_TRAINING = {
    "epochs": 200, "patience": 30, "lr": 0.002, "weight_decay": 0.0005,
    "batch_size": 64, "steps_per_epoch": 1, "proposal_steps": 256,
    "head_steps": 512, "head_lr": 0.001, "fit_temperature": True,
    "train_incumbent": True,
}
DEFAULT_ADAPTATION = {
    "backend": "exact_reference_tail", "open_first": 0, "bank_size": 16,
    "roles": 8, "head_hidden": 64, "probes": 16, "row_trim": 0.8,
    "omega": 0.9, "rho_x": 0.000001, "H_plus": 0.05, "H_row": 1.0,
    "kappa_train": 2.0, "covariance_floor": 0.000001,
    "solver_steps": 24, "line_search_steps": 16, "coefficient_cap": 1.0,
    "budget_margin": 0.0000001,
    "proposal_kind": "native", "tent_lr": 0.001, "tent_steps": 1,
}
DEFAULT_PROTOCOL = {
    "seed": 1, "query_size": 64, "calibration_calls": 512,
    "selection_calls": 512, "audit_calls": 2048, "delta_cov": 0.1,
    "delta": 0.05, "p_star": 0.02, "zeta": 0.002,
    "member_multipliers": [1.0, 5.0, 10.0, 25.0, 50.0],
    "minimum_gain": 0.0, "split_policy": "official_disjoint",
}


def resolve_config(path, *, profile="full", data_root=None, download=False):
    source = read_json(path)
    allowed = {"name", "dataset", "model", "training", "adaptation", "protocol", "notes"}
    extra = set(source) - allowed
    if extra:
        raise ValueError(f"Unknown experiment configuration keys: {sorted(extra)}")
    for required in ("name", "dataset", "model"):
        if required not in source:
            raise ValueError(f"Missing configuration section: {required}")
    if not isinstance(source["dataset"], dict) or "name" not in source["dataset"]:
        raise ValueError("dataset.name is required")
    if not isinstance(source["model"], dict) or "name" not in source["model"]:
        raise ValueError("model.name is required")
    if profile not in {"full", "pilot", "smoke"}:
        raise ValueError("profile must be full, pilot or smoke")
    result = copy.deepcopy(source)
    for section, defaults in (("training", DEFAULT_TRAINING), ("adaptation", DEFAULT_ADAPTATION),
                              ("protocol", DEFAULT_PROTOCOL)):
        values = result.get(section, {})
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {section} keys: {sorted(unknown)}")
        result[section] = {**defaults, **values}
    result["profile"] = profile
    root = data_root if data_root is not None else result["dataset"].get("root", "./data/" + result["dataset"]["name"])
    result["dataset"]["root"] = str(Path(root).expanduser().resolve())
    result["dataset"].setdefault("options", {})
    for key in ("npz", "manifest", "fit_root", "class_map", "data_pt", "scaffold_path"):
        value = result["dataset"]["options"].get(key)
        if value is not None:
            option_path = Path(value).expanduser()
            if not option_path.is_absolute():
                option_path = Path(result["dataset"]["root"]) / option_path
            result["dataset"]["options"][key] = str(option_path.resolve())
    for key in ("checkpoint", "weights_path"):
        value = result["model"].get(key)
        if value is not None:
            checkpoint = Path(value).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = Path(result["dataset"]["root"]) / checkpoint
            result["model"][key] = str(checkpoint.resolve())
    result["dataset"]["download"] = bool(download)
    result["model"]["allow_download"] = bool(download)
    if profile == "pilot":
        for key, limit in (("epochs", 20), ("patience", 5), ("proposal_steps", 12), ("head_steps", 24)):
            result["training"][key] = min(result["training"][key], limit)
        result["training"]["steps_per_epoch"] = min(result["training"]["steps_per_epoch"], 2)
        for key, limit in (("calibration_calls", 32), ("selection_calls", 16), ("audit_calls", 32)):
            result["protocol"][key] = min(result["protocol"][key], limit)
    if profile == "smoke":
        result["model"].update({"hidden": 8, "enc_hidden": 12, "pretrained": False, "allow_download": False})
        for key in ("checkpoint", "weights_path", "weights"):
            result["model"].pop(key, None)
        result["dataset"]["download"] = False
        if result["model"]["name"] not in {"resnet50", "gcn", "graphsage"}:
            result["model"]["T"] = 3
        result["training"].update({"epochs": 2, "patience": 2, "batch_size": 4,
                                  "proposal_steps": 2, "head_steps": 2, "steps_per_epoch": 1,
                                  "train_incumbent": result["model"]["name"] != "resnet50"})
        result["protocol"].update({"query_size": 4, "calibration_calls": 4,
                                  "selection_calls": 2, "audit_calls": 4,
                                  "split_policy": "official_disjoint"})
        result["adaptation"].update({"open_first": 0, "bank_size": 4, "head_hidden": 8,
                                     "probes": 4, "solver_steps": 8, "line_search_steps": 6})
    validate_config(result)
    return result


def validate_config(config):
    for section, keys in (("training", ("epochs", "patience", "batch_size", "steps_per_epoch", "proposal_steps", "head_steps")),
                          ("protocol", ("query_size", "calibration_calls", "selection_calls", "audit_calls")),
                          ("adaptation", ("bank_size", "roles", "head_hidden", "probes", "solver_steps", "line_search_steps"))):
        for key in keys:
            value = config[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for key in ("delta", "delta_cov", "p_star"):
        if not 0 < config["protocol"][key] < 1:
            raise ValueError(f"protocol.{key} must be in (0,1)")
    for key in ("H_plus", "rho_x", "budget_margin"):
        if not math.isfinite(config["adaptation"][key]) or config["adaptation"][key] < 0:
            raise ValueError(f"adaptation.{key} must be nonnegative and finite")
    for key in ("H_row", "covariance_floor", "coefficient_cap", "row_trim", "tent_lr"):
        if not math.isfinite(config["adaptation"][key]) or config["adaptation"][key] <= 0:
            raise ValueError(f"adaptation.{key} must be positive and finite")
    if not 0 < config["adaptation"]["omega"] < 1:
        raise ValueError("H_row must be positive and omega must be in (0,1)")
    if not isinstance(config["adaptation"]["tent_steps"], int) or config["adaptation"]["tent_steps"] < 1:
        raise ValueError("tent_steps must be a positive integer")
    if not math.isfinite(config["adaptation"]["kappa_train"]) or config["adaptation"]["kappa_train"] < 0:
        raise ValueError("kappa_train must be finite and nonnegative")
    if config["adaptation"]["open_first"] < 0:
        raise ValueError("open_first must be nonnegative")
    if config["adaptation"]["backend"] != "exact_reference_tail":
        raise ValueError("Dataset experiment entry points use exact_reference_tail; use the native arxiv entry point for tube admission")
    if config["adaptation"]["proposal_kind"] not in {"native", "tent"}:
        raise ValueError("proposal_kind must be native or tent")
    if config["adaptation"]["proposal_kind"] == "tent" and config["model"]["name"] != "resnet50":
        raise ValueError("The last-four-block Tent proposal requires ResNet-50")
    multipliers = config["protocol"]["member_multipliers"]
    if not multipliers or any(not math.isfinite(value) or value <= 0 for value in multipliers):
        raise ValueError("member_multipliers must contain positive finite numbers")
    if sorted(set(multipliers)) != multipliers:
        raise ValueError("member_multipliers must be strictly increasing")

