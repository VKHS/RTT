"""Persisted fitting, calibration, selection and independent dataset audits."""
from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .adaptation import (AdaptationModules, ReferenceTailEngine, coefficient_geometry,
                         joint_score, role_ids, true_tail_gradient)
from .batching import BatchFactory, synthetic_bundle
from .datasets import load_dataset
from .frontier import continuous_floor_frontier
from .losses import floor_damage, log_probabilities, supervised_loss, task_metrics, unit_weights
from .models import build_model
from .protocol import conformal_radius, make_protocol, members, summarize_records
from .runtime import (NUMERICAL_STATUS, Timer, environment, json_safe, load_checkpoint,
                      object_hash, progress, read_json, save_checkpoint, seed_for, sha256, write_json)
from .training import fit_adaptation, train_incumbent


def scientific_config(config):
    result = copy.deepcopy(config)
    result["dataset"].pop("download", None)
    result["model"].pop("allow_download", None)
    return result


def primary_metric(bundle):
    if bundle.multilabel:
        return "macro_ap"
    metric = bundle.metadata.get("metric", "accuracy")
    return "auc" if metric in {"roc_auc", "auc"} or bundle.name == "twitch" else "f1" if bundle.name == "elliptic" else "accuracy"


class Experiment:
    def __init__(self, config, directory, device="cpu"):
        self.config, self.directory = config, Path(directory).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.bundle = self.factory = self.model = self.modules = self.protocol = None
        self.config_hash = object_hash(scientific_config(config))

    def _load_data(self):
        if self.bundle is not None:
            return
        dataset = self.config["dataset"]
        name = dataset["name"]
        if self.config["profile"] == "smoke":
            task = "graph" if name == "ogbg_molpcba" else "image" if name == "imagenet_c" else "tabular" if name == "acs_income" else "node"
            classes = 1000 if task == "image" else 2 if name in {"twitch", "facebook100", "elliptic", "acs_income"} else 3
            self.bundle = synthetic_bundle(name, task, self.directory / "smoke_data", seed=self.config["protocol"]["seed"], classes=classes)
        else:
            self.bundle = load_dataset(name, dataset["root"], dataset["options"], dataset["download"])
        self.factory = BatchFactory(self.bundle, self.device, image_size=32 if self.config["profile"] == "smoke" else 224)

    def prepare(self):
        self._load_data()
        path = self.directory / "data_manifest.json"
        old = read_json(path) if path.exists() else None
        if old and old["configuration_sha256"] != self.config_hash:
            raise ValueError("This run's dataset/configuration is frozen; choose a new run directory")
        inventory_hash = None
        if self.bundle.image_paths is not None:
            inventory = []
            for name in self.bundle.image_paths:
                file = Path(name)
                stat = file.stat()
                inventory.append((str(file), stat.st_size, stat.st_mtime_ns))
            inventory_hash = object_hash({"files": inventory, "labels": self.bundle.y,
                                          "splits": self.bundle.splits, "image_ids": self.bundle.metadata.get("image_ids"),
                                          "dataset": self.bundle.name})
        # Image bytes are hashed at first preparation. Later stages reuse this
        # digest only when the complete path/size/mtime inventory is unchanged.
        if old and inventory_hash is not None and inventory_hash == old.get("image_inventory_sha256"):
            digest = old["content_sha256"]
        else:
            digest = self.bundle.fingerprint()
        if old and digest != old["content_sha256"]:
            raise ValueError("The prepared dataset content changed after this run was frozen")
        report = {"dataset": self.bundle.name, "task": self.bundle.task, "items": self.bundle.n,
                  "input_dim": self.bundle.input_dim, "output_dim": self.bundle.output_dim,
                  "multilabel": self.bundle.multilabel, "split_sizes": {k: len(v) for k, v in self.bundle.splits.items()},
                  "content_sha256": digest, "configuration_sha256": self.config_hash,
                  "image_inventory_sha256": inventory_hash,
                  "image_digest_cache_rule": "reuse only when complete path/size/mtime inventory is unchanged",
                  "synthetic": bool(self.bundle.metadata.get("synthetic", False)),
                  "metadata": self.bundle.metadata}
        if not old:
            write_json(path, report)
        protocol_path = self.directory / "protocol.json"
        if protocol_path.exists():
            self.protocol = read_json(protocol_path)
        else:
            self.protocol = make_protocol(self.bundle, self.config)
            write_json(protocol_path, self.protocol)
        return report

    def train(self):
        if (self.directory / "training.json").exists():
            raise ValueError("Training is already complete; use later stage commands or choose a new run")
        data = self.prepare()
        self.model = build_model(self.config["model"]["name"], self.bundle.input_dim,
                                 self.bundle.output_dim, self.config["model"]).to(self.device)
        if self.config["adaptation"]["open_first"] >= self.model.T:
            raise ValueError("open_first is outside the incumbent horizon")
        incumbent = train_incumbent(self.model, self.factory, self.protocol, self.config)
        save_checkpoint(self.directory / "checkpoints/model.pt", {
            "state_dict": self.model.state_dict(), "configuration_sha256": self.config_hash,
            "input_dim": self.bundle.input_dim, "output_dim": self.bundle.output_dim})
        self.modules, adaptation = fit_adaptation(self.model, self.factory, self.protocol, self.config)
        save_checkpoint(self.directory / "checkpoints/adaptation.pt", {
            "state_dict": self.modules.state_dict(), "configuration_sha256": self.config_hash})
        manifest = {"configuration": scientific_config(self.config), "configuration_sha256": self.config_hash,
                    "dataset_sha256": data["content_sha256"], "protocol_sha256": sha256(self.directory / "protocol.json"),
                    "model_checkpoint_sha256": sha256(self.directory / "checkpoints/model.pt"),
                    "adaptation_checkpoint_sha256": sha256(self.directory / "checkpoints/adaptation.pt"),
                    "environment": environment(), "backend": "exact_reference_tail",
                    "synthetic": data["synthetic"], "numerical_status": NUMERICAL_STATUS,
                    "training_scheme": "raw-proposal warm start plus supervised admission heads; no implicit KKT differentiation"}
        write_json(self.directory / "manifest.json", manifest)
        result = {"incumbent": incumbent, "adaptation": adaptation, "status": "COMPLETE"}
        write_json(self.directory / "training.json", result)
        return result

    def load(self):
        self.prepare()
        manifest = read_json(self.directory / "manifest.json")
        if manifest["configuration_sha256"] != self.config_hash:
            raise ValueError("Configuration differs from the frozen trained experiment")
        if sha256(self.directory / "protocol.json") != manifest["protocol_sha256"]:
            raise ValueError("The frozen label-role allocation changed")
        for filename, key in (("model.pt", "model_checkpoint_sha256"), ("adaptation.pt", "adaptation_checkpoint_sha256")):
            if sha256(self.directory / "checkpoints" / filename) != manifest[key]:
                raise ValueError(f"The frozen trained checkpoint changed: {filename}")
        if self.model is None:
            settings = dict(self.config["model"])
            settings.update(pretrained=False, allow_download=False)
            settings.pop("checkpoint", None)
            settings.pop("weights_path", None)
            settings.pop("weights", None)
            self.model = build_model(settings["name"], self.bundle.input_dim, self.bundle.output_dim, settings).to(self.device)
            checkpoint = load_checkpoint(self.directory / "checkpoints/model.pt")
            if checkpoint["configuration_sha256"] != self.config_hash:
                raise ValueError("Model checkpoint belongs to a different experiment")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            # Do not reproject an already exported checkpoint while loading.
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        if self.modules is None:
            self.modules = AdaptationModules(self.model, self.config["adaptation"], seed=self.config["protocol"]["seed"]).to(self.device)
            checkpoint = load_checkpoint(self.directory / "checkpoints/adaptation.pt")
            if checkpoint["configuration_sha256"] != self.config_hash:
                raise ValueError("Adaptation checkpoint belongs to a different experiment")
            self.modules.load_state_dict(checkpoint["state_dict"])
            self.modules.eval()
            for parameter in self.modules.parameters():
                parameter.requires_grad_(False)
        return ReferenceTailEngine(self.model, self.modules, self.config["adaptation"])

    def calibrate(self):
        engine = self.load()
        destination = self.directory / "calibration.json"
        if destination.exists():
            raise ValueError("Calibration is already frozen; use a new run for a different calibration")
        progress_path = self.directory / "calibration_progress.json"
        frozen = {"configuration_sha256": self.config_hash,
                  "model_checkpoint_sha256": sha256(self.directory / "checkpoints/model.pt"),
                  "training_checkpoint_sha256": sha256(self.directory / "checkpoints/adaptation.pt"),
                  "protocol_sha256": sha256(self.directory / "protocol.json")}
        calibration = read_json(progress_path) if progress_path.exists() else {**frozen, "members": {}}
        if any(calibration.get(key) != value for key, value in frozen.items()):
            raise ValueError("Partial calibration belongs to different frozen inputs")
        depths, settings = self.modules.open_depths, self.config["protocol"]
        delta_depth = settings["delta_cov"] / len(depths)
        for member in members(self.config):
            row = calibration["members"].setdefault(member["name"], {**member, "radii": {}, "scores": {}})
            for depth in depths:
                if str(depth) in row["radii"]:
                    continue
                scores = row["scores"].setdefault(str(depth), [])
                radii = {int(k): float(value) for k, value in row["radii"].items()}
                for index in range(len(scores), settings["calibration_calls"]):
                    batch = self.factory.sample(self.protocol["pools"]["calibration"], settings["query_size"],
                                                seed_for(settings["seed"], "calibration", member["name"], depth, index))
                    state, context = engine.serve(batch.unlabeled(), radii, budget=member["budget"], stop_before=depth)
                    module = self.modules.depths[str(depth)]
                    with torch.no_grad():
                        exported = context.get("rtt_exported_states", {}).get(depth)
                        candidates = module.candidates(state, context, self.config["adaptation"]["row_trim"], exported)
                        field, variance = module.estimate(state, candidates, batch.units)
                        roles = role_ids(state, context, self.config["adaptation"]["roles"])
                        estimated, covariance, _ = coefficient_geometry(candidates, field, variance, roles, self.config["adaptation"]["roles"])
                    gradient = true_tail_gradient(self.model, batch, state, depth + 1, context)
                    with torch.no_grad():
                        actual, _, _ = coefficient_geometry(candidates, gradient, variance, roles, self.config["adaptation"]["roles"])
                        score = joint_score(actual - estimated, covariance)
                    scores.append(score)
                    progress("calibration", len(scores), settings["calibration_calls"], member=member["name"], depth=depth)
                    if len(scores) % 8 == 0:
                        write_json(progress_path, calibration)
                row["radii"][str(depth)] = conformal_radius([float(value) for value in scores], delta_depth)
                write_json(progress_path, calibration)
        calibration.update({"status": "COMPLETE", "delta_per_depth": delta_depth,
                            "calibration_scope": "fixed-member sequential calibration; selected-member nominal coverage is not claimed"})
        write_json(destination, calibration)
        return calibration

    def _member(self, name=None):
        calibration = read_json(self.directory / "calibration.json")
        if (calibration["configuration_sha256"] != self.config_hash
                or calibration["training_checkpoint_sha256"] != sha256(self.directory / "checkpoints/adaptation.pt")
                or calibration["model_checkpoint_sha256"] != sha256(self.directory / "checkpoints/model.pt")
                or calibration["protocol_sha256"] != sha256(self.directory / "protocol.json")):
            raise ValueError("Calibration and checkpoint/configuration differ")
        if name is None:
            selection = read_json(self.directory / "selection.json")
            if (selection["configuration_sha256"] != self.config_hash
                    or selection["calibration_sha256"] != sha256(self.directory / "calibration.json")):
                raise ValueError("Selection and frozen calibration/configuration differ")
            name = selection["selected"]
        if name is None:
            return {"name": "incumbent_only", "budget": 0.0, "radii": {}}
        if name not in calibration["members"]:
            raise ValueError(f"Unknown calibrated member: {name}")
        row = calibration["members"][name]
        return {"name": name, "budget": row["budget"], "radii": {int(k): float(v) for k, v in row["radii"].items()}}

    def _corrector(self, batch, result, budget):
        with Timer(self.device) as timer, torch.no_grad():
            # Charge a complete reference pass to the corrector, independently
            # of the paired RTT evaluation's cached reference output.
            context = dict(batch.context)
            state = self.model.encode(batch.x, context)
            for depth in range(self.model.T):
                state = self.model.transition(state, depth, context)
            reference = batch.select(self.model.logits(state, context))
            embedding = batch.select(self.model.embedding(state, context))
            q_logits = reference + self.modules.corrector(embedding)
            p = log_probabilities(reference, batch.multilabel).exp().cpu().numpy()
            q = log_probabilities(q_logits, batch.multilabel).exp().cpu().numpy()
            shape = p.shape
            p = np.maximum(p.reshape(-1, shape[-1]), 1e-300)
            p /= p.sum(1, keepdims=True)
            q = q.reshape(p.shape)
            q /= q.sum(1, keepdims=True)
            weights = unit_weights(batch.mask).cpu().numpy().reshape(-1)
            safe_budget = max(0.0, budget - self.config["adaptation"]["budget_margin"])
            _, projected, _, metadata = continuous_floor_frontier(q, p, weights, safe_budget,
                                                                  max(0.0, self.config["adaptation"]["H_row"] - self.config["adaptation"]["budget_margin"]))
            if not metadata["converged"]:
                raise FloatingPointError("Continuous corrector allocation failed its numerical checks")
            log_probability = torch.as_tensor(projected, dtype=torch.float64, device=self.device).log().reshape(shape)
            logits = log_probability[..., 1] - log_probability[..., 0] if batch.multilabel else log_probability
            damage, row, _, _ = floor_damage(reference, logits, batch.mask, batch.multilabel)
            fallback = not bool(torch.isfinite(logits).all()) or float(damage) > budget or bool((row[batch.mask] > self.config["adaptation"]["H_row"]).any())
            if fallback:
                logits = reference.clone()
        return logits, {"latency_seconds": timer.seconds, "fallback": fallback, "frontier": metadata}

    def evaluate_call(self, phase, index, member, *, pool=None, corrector=False):
        settings = self.config["protocol"]
        pool = pool or ("selection" if phase == "selection" else "audit")
        call_id = f"{self.bundle.name}:{phase}:{pool}:{index:08d}"
        seed = seed_for(settings["seed"], phase, pool, index)
        batch = self.factory.sample(self.protocol["pools"][pool], settings["query_size"], seed)
        engine = ReferenceTailEngine(self.model, self.modules, self.config["adaptation"])
        result = engine.serve(batch.unlabeled(), member["radii"], budget=member["budget"])
        with torch.no_grad():
            reference_loss = float(supervised_loss(result.reference_logits, batch.labels, batch.multilabel, batch.mask))
            loss = float(supervised_loss(result.logits, batch.labels, batch.multilabel, batch.mask))
        reference_metrics = task_metrics(batch.labels.cpu().numpy(), result.reference_logits.cpu().numpy(), multilabel=batch.multilabel)
        metrics = task_metrics(batch.labels.cpu().numpy(), result.logits.cpu().numpy(), multilabel=batch.multilabel)
        metric = primary_metric(self.bundle)
        record = {"call_id": call_id, "phase": phase, "status": "OK", "member": member["name"],
                  "query_items": len(batch.item_ids), "task_units": batch.units,
                  "gain": reference_loss - loss, "down": loss - reference_loss,
                  "reference_loss": reference_loss, "loss": loss,
                  "primary_metric_name": metric, "primary_metric_reference": reference_metrics.get(metric),
                  "primary_metric_served": metrics.get(metric), "metrics": metrics,
                  "reference_metrics": reference_metrics, **result.report}
        if corrector:
            corrected, correction = self._corrector(batch, result, member["budget"])
            corrected_loss = float(supervised_loss(corrected, batch.labels, batch.multilabel, batch.mask))
            record["corrector"] = {"gain": reference_loss - corrected_loss, "loss": corrected_loss,
                                    "metrics": task_metrics(batch.labels.cpu().numpy(), corrected.cpu().numpy(), multilabel=batch.multilabel),
                                    **correction}
        return record

    def select(self):
        self.load()
        path = self.directory / "selection.json"
        if path.exists():
            raise ValueError("Selection is frozen; a different selection requires a new run")
        menu = members(self.config)
        selection_config = copy.deepcopy(self.config)
        selection_config["protocol"]["delta"] /= len(menu)
        sequence, selected = [], None
        for specification in menu:
            member = self._member(specification["name"])
            records = []
            for index in range(self.config["protocol"]["selection_calls"]):
                records.append(self.evaluate_call("selection", index, member))
                progress("selection", index + 1, self.config["protocol"]["selection_calls"], member=member["name"])
            summary = summarize_records(records, selection_config, expected_count=len(records))
            passed = (summary["status"] == "COMPLETE" and summary["u"] is not None
                      and summary["u"] <= self.config["protocol"]["p_star"]
                      and summary["betting_lb"] is not None
                      and summary["betting_lb"] > self.config["protocol"]["minimum_gain"])
            sequence.append({**specification, "passed": passed, "summary": summary})
            write_json(self.directory / "selection_records" / (specification["name"] + ".json"), records)
            if not passed:
                break
            selected = specification["name"]
        result = {"selected": selected, "sequence": sequence, "configuration_sha256": self.config_hash,
                  "calibration_sha256": sha256(self.directory / "calibration.json"),
                  "status": "SELECTED" if selected is not None else "NO_MEMBER_SELECTED",
                  "rule": "least-to-most fixed sequence; Bonferroni member bounds; stop at first failure"}
        write_json(path, result)
        return result

    def audit(self, shard_index=0, shards=1):
        self.load()
        if shards < 1 or not 0 <= shard_index < shards:
            raise ValueError("Require 0 <= shard-index < shards")
        member = self._member()
        path = self.directory / "records" / f"audit_shard_{shard_index:04d}.jsonl"
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            raise ValueError("This audit shard is already committed; do not resample a completed audit")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        count = self.config["protocol"]["audit_calls"]
        indices = list(range(shard_index, count, shards))
        with temporary.open("w") as stream:
            for completed, index in enumerate(indices, 1):
                record = self.evaluate_call("audit", index, member, corrector=True)
                stream.write(json.dumps(json_safe(record), sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                progress("audit", completed, len(indices), shard=shard_index, call=index, mode=record["mode"])
        temporary.replace(path)
        metadata = {"status": "COMPLETE", "shard_index": shard_index, "shards": shards,
                    "indices": indices, "count": len(indices), "records_sha256": sha256(path),
                    "configuration_sha256": self.config_hash, "selection_sha256": sha256(self.directory / "selection.json")}
        write_json(meta_path, metadata)
        return metadata

    def report(self):
        self.load()
        self._member()
        records, seen_indices, shard_counts = [], [], set()
        for meta_path in sorted((self.directory / "records").glob("audit_shard_*.meta.json")):
            metadata = read_json(meta_path)
            path = meta_path.with_suffix("").with_suffix(".jsonl")
            if metadata["configuration_sha256"] != self.config_hash or metadata["selection_sha256"] != sha256(self.directory / "selection.json"):
                raise ValueError("Audit shard was produced by a different frozen policy")
            if sha256(path) != metadata["records_sha256"]:
                raise ValueError("An audit record file changed after completion")
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != metadata["count"]:
                raise ValueError("Audit shard count differs from its completion ledger")
            records.extend(rows)
            seen_indices.extend(metadata["indices"])
            shard_counts.add(metadata["shards"])
        expected = self.config["protocol"]["audit_calls"]
        if len(shard_counts) != 1 or sorted(seen_indices) != list(range(expected)):
            raise ValueError("Audit is incomplete or has overlapping shards; reduction refuses a partial sample")
        records.sort(key=lambda row: row["call_id"])
        summary = summarize_records(records, self.config, expected_count=expected)
        summary.update({"dataset": self.bundle.name, "task": self.bundle.task,
                        "synthetic": self.config["profile"] == "smoke", "backend": "exact_reference_tail",
                        "numerical_status": NUMERICAL_STATUS, "selection": read_json(self.directory / "selection.json"),
                        "mean_latency_seconds": float(np.mean([row["latency_seconds"] for row in records])),
                        "release_counts": {mode: sum(row["mode"] == mode for row in records) for mode in ("R1", "NO_STEP", "F")},
                        "corrector_mean_gain": float(np.mean([row["corrector"]["gain"] for row in records])),
                        "corrector_capacity": self.modules.capacity,
                        "measurement_scope": "independent sampled episodes; task metrics are means of per-episode metrics, not official-test population metrics"})
        summary["statistical_release"] = bool(summary["status"] == "COMPLETE" and summary["selection"]["selected"] is not None
                                              and summary["u"] is not None and summary["u"] <= self.config["protocol"]["p_star"]
                                              and summary["betting_lb"] is not None and summary["betting_lb"] > self.config["protocol"]["minimum_gain"])
        write_json(self.directory / "summary.json", summary)
        with (self.directory / "metrics.csv").open("w", newline="") as stream:
            columns = ["call_id", "gain", "down", "reference_loss", "loss", "mode", "accepted_steps",
                       "call_damage_bound", "row_damage_max", "latency_seconds", "primary_metric_name",
                       "primary_metric_reference", "primary_metric_served"]
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        lines = [f"# RTT — {self.bundle.name}", "", f"Dataset task: {self.bundle.task}.",
                 f"Synthetic execution check: {summary['synthetic']}.",
                 f"Completed calls: {summary['n']}.",
                 f"Mean cross-entropy gain: {summary['mean_gain']} nat per task unit.",
                 f"Regression events: {summary['N_zeta']}; upper bound: {summary['u']}.",
                 f"Betting lower bound on mean gain: {summary['betting_lb']}.",
                 f"Statistical release: {summary['statistical_release']}.", "",
                 "The exact-reference-tail backend includes a separate full reference pass and all trial-tail work in its timing.",
                 "Reported task metrics average episode metrics; use the separate evaluate command for official population metrics.",
                 "Numerical checks are research evidence, not a formal floating-point enclosure proof.", ""]
        (self.directory / "REPORT.md").write_text("\n".join(lines))
        return summary

    def predict(self, output, *, pool="audit", index=0, member_name=None):
        engine = self.load()
        member = self._member(member_name)
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Prediction output must be a new or empty directory")
        batch = self.factory.sample(self.protocol["pools"][pool], self.config["protocol"]["query_size"],
                                    seed_for(self.config["protocol"]["seed"], "predict", pool, index), labels=False)
        result = engine.serve(batch, member["radii"], budget=member["budget"])
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "predictions.npz", item_ids=batch.item_ids,
                            logits=result.logits.cpu().numpy(), log_probabilities=log_probabilities(result.logits, batch.multilabel).cpu().numpy())
        report = {**result.report, "member": member["name"], "exploratory_member": member_name is not None,
                  "pool": pool, "query_items": len(batch.item_ids)}
        write_json(output / "prediction.json", report)
        return report

    def evaluate_population(self, output, *, split="test", batch_size=None):
        """One separately committed pass over an official split, not audit resampling."""
        engine = self.load()
        member = self._member()
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Population evaluation must use a new output directory")
        if split not in self.bundle.splits:
            raise ValueError(f"Unknown official split: {split}")
        indices = self.bundle.splits[split]
        size = batch_size or self.config["protocol"]["query_size"]
        labels, reference, served = [], [], []
        for start in range(0, len(indices), size):
            batch = self.factory.make(indices[start:start + size])
            result = engine.serve(batch.unlabeled(), member["radii"], budget=member["budget"])
            labels.append(batch.labels.cpu().numpy())
            reference.append(result.reference_logits.cpu().numpy())
            served.append(result.logits.cpu().numpy())
        labels, reference, served = map(np.concatenate, (labels, reference, served))
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "population_predictions.npz", item_ids=indices, labels=labels,
                            reference_logits=reference, served_logits=served)
        result = {"dataset": self.bundle.name, "split": split, "items": len(indices), "batch_size": size,
                  "reference": task_metrics(labels, reference, multilabel=self.bundle.multilabel),
                  "served": task_metrics(labels, served, multilabel=self.bundle.multilabel),
                  "scope": "exact empirical metric on the specified official population; overlaps any custom pools drawn from it",
                  "numerical_status": NUMERICAL_STATUS}
        write_json(output / "population_metrics.json", result)
        return result
