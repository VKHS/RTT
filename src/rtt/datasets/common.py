"""Typed, framework-independent data contracts shared by dataset adapters.

Unknown node labels are -1. Missing molecular assay labels are NaN. Neither is
silently made into a negative label. Split indices always refer to scored items
(nodes, molecules, images or rows), not individual binary assay entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class DatasetError(ValueError):
    """Missing data, incompatible schema, or invalid experimental split."""


@dataclass
class GraphSample:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray | None = None

    def validate(self) -> "GraphSample":
        self.x = np.asarray(self.x)
        if self.x.ndim != 2 or len(self.x) == 0 or not np.isfinite(self.x).all():
            raise DatasetError("Graph node features must be a finite [nodes, features] matrix")
        self.edge_index = checked_edges(self.edge_index, len(self.x))
        if self.edge_attr is not None:
            self.edge_attr = np.asarray(self.edge_attr)
            if self.edge_attr.ndim != 2 or len(self.edge_attr) != self.edge_index.shape[1]:
                raise DatasetError("Graph edge attributes must align with edge_index")
            if not np.isfinite(self.edge_attr).all():
                raise DatasetError("Graph edge attributes must be finite")
        return self


@dataclass
class DatasetBundle:
    name: str
    task: str
    y: np.ndarray
    splits: dict[str, np.ndarray]
    num_classes: int
    x: np.ndarray | None = None
    edge_index: np.ndarray | None = None
    graphs: Sequence[GraphSample] | None = None
    image_paths: list[str] | None = None
    year: np.ndarray | None = None
    time: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def input_dim(self) -> int:
        if self.x is not None:
            return int(self.x.shape[1])
        if self.graphs is not None and len(self.graphs):
            return int(self.graphs[0].x.shape[1])
        return 3 if self.task == "image" else 0

    @property
    def output_dim(self) -> int:
        return self.num_classes

    @property
    def multilabel(self) -> bool:
        return self.y.ndim == 2 and self.y.shape[1] > 1

    def validate(self, *, require_nonempty: bool = True) -> "DatasetBundle":
        if self.task not in {"node", "graph", "image", "tabular"}:
            raise DatasetError(f"Unsupported task: {self.task}")
        self.y = np.asarray(self.y)
        if self.y.ndim not in (1, 2) or not len(self.y):
            raise DatasetError("Labels must contain at least one item")
        if self.y.ndim == 2 and self.y.shape[1] == 1:
            self.y = self.y[:, 0]
        if self.num_classes < 2:
            raise DatasetError("num_classes must be >=2 (or the number of molecular tasks)")
        if self.multilabel:
            if self.y.shape[1] != self.num_classes:
                raise DatasetError("Multilabel task count differs from num_classes")
            known = self.y[np.isfinite(self.y)]
            if np.any((known != 0) & (known != 1)):
                raise DatasetError("Known molecular assay labels must be 0 or 1")
        else:
            if not np.isfinite(self.y).all() or np.any(self.y != self.y.astype(np.int64)):
                raise DatasetError("Single-label targets must be integers; use -1 for unknown")
            self.y = self.y.astype(np.int64, copy=False)
            if np.any((self.y < -1) | (self.y >= self.num_classes)):
                raise DatasetError("Class labels fall outside [-1,num_classes)")
        required = {"train", "valid", "test"}
        if not required.issubset(self.splits):
            raise DatasetError("Explicit train, valid and test splits are required")
        seen = np.zeros(self.n, dtype=np.uint8)
        for key in ("train", "valid", "test"):
            indices = np.asarray(self.splits[key])
            if indices.dtype == bool:
                if indices.shape != (self.n,):
                    raise DatasetError(f"{key} mask has wrong shape")
                indices = np.flatnonzero(indices)
            if indices.ndim != 1 or np.any(indices != indices.astype(np.int64)):
                raise DatasetError(f"{key} indices must be a one-dimensional integer array")
            indices = indices.astype(np.int64, copy=False)
            if np.any(indices < 0) or np.any(indices >= self.n) or len(np.unique(indices)) != len(indices):
                raise DatasetError(f"Invalid or duplicate {key} indices")
            if require_nonempty and not len(indices):
                raise DatasetError(f"Empty {key} split")
            if self.multilabel:
                if np.any(~np.isfinite(self.y[indices]).any(axis=1)):
                    raise DatasetError(f"{key} includes molecules without any observed assay")
            elif np.any(self.y[indices] < 0):
                raise DatasetError(f"{key} contains unknown labels")
            seen[indices] += 1
            self.splits[key] = indices
        if seen.max(initial=0) > 1:
            raise DatasetError("Train, validation and test sets overlap")
        if self.task in {"node", "tabular"}:
            self.x = np.asarray(self.x)
            if self.x.ndim != 2 or len(self.x) != self.n or not np.isfinite(self.x).all():
                raise DatasetError("Features must be finite and aligned with labels")
            self.x = self.x.astype(np.float32, copy=False)
        if self.task == "node":
            self.edge_index = checked_edges(self.edge_index, self.n)
        elif self.task == "graph":
            if self.graphs is None or len(self.graphs) != self.n:
                raise DatasetError("Molecular graph count differs from labels")
            for index in {0, self.n - 1}:
                self.graphs[index].validate()
        elif self.task == "image":
            if self.image_paths is None or len(self.image_paths) != self.n:
                raise DatasetError("Image path count differs from labels")
            missing = next((path for path in self.image_paths if not Path(path).is_file()), None)
            if missing is not None:
                raise DatasetError(f"Image file is missing: {missing}")
            identities = self.metadata.get("image_ids", [Path(path).name for path in self.image_paths])
            if len(identities) != self.n:
                raise DatasetError("Image identity count differs from images")
            assigned = {}
            for split in ("train", "valid", "test"):
                for index in self.splits[split]:
                    identity = str(identities[index])
                    current = (split, int(self.y[index]))
                    if assigned.setdefault(identity, current) != current:
                        raise DatasetError(f"Underlying image {identity} crosses splits or changes label")
        for field_name in ("year", "time"):
            value = getattr(self, field_name)
            if value is not None and np.asarray(value).shape != (self.n,):
                raise DatasetError(f"{field_name} must have one entry per item")
        return self

    def fingerprint(self) -> str:
        """Full numeric content hash, including split IDs and graph topology."""
        digest = hashlib.sha256()
        digest.update(self.name.encode())
        digest.update(self.task.encode())

        def add(array):
            if array is None:
                digest.update(b"None")
                return
            array = np.ascontiguousarray(array)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            view = memoryview(array).cast("B")
            for start in range(0, len(view), 1 << 20):
                digest.update(view[start:start + (1 << 20)])

        for array in (self.x, self.y, self.edge_index, self.year, self.time):
            add(array)
        for key in sorted(self.splits):
            digest.update(key.encode())
            add(self.splits[key])
        if self.graphs is not None:
            for graph in self.graphs:
                add(graph.x)
                add(graph.edge_index)
                add(graph.edge_attr)
        if self.image_paths is not None:
            # Image content, rather than mutable mtimes, defines this fingerprint.
            for image_path in self.image_paths:
                path = Path(image_path)
                digest.update(path.name.encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        digest.update(block)
        return digest.hexdigest()


def checked_edges(value, n: int) -> np.ndarray:
    edges = np.asarray(value)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise DatasetError("edge_index must have shape [2,edges]")
    if np.any(edges != edges.astype(np.int64)):
        raise DatasetError("Graph endpoints must be integers")
    edges = edges.astype(np.int64, copy=False)
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise DatasetError("Graph endpoint outside the node array")
    return edges


def source_split(indices, *, seed: int = 1, valid_fraction: float = 0.2):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2 or not 0 < valid_fraction < 1:
        raise DatasetError("Source domain needs at least two labeled examples and 0<valid_fraction<1")
    order = np.random.default_rng(seed).permutation(indices)
    n_valid = min(len(order) - 1, max(1, int(round(len(order) * valid_fraction))))
    return np.sort(order[n_valid:]), np.sort(order[:n_valid])


def standardize_train(bundle: DatasetBundle, enabled: bool = True) -> DatasetBundle:
    if enabled and bundle.x is not None:
        x = np.asarray(bundle.x, dtype=np.float32)
        training = x[bundle.splits["train"]].astype(np.float64)
        mean = training.mean(0)
        std = training.std(0)
        std[std < 1e-8] = 1.0
        bundle.x = ((x - mean) / std).astype(np.float32)
        bundle.metadata["normalization"] = "mean/std fitted on train split only"
        bundle.metadata["normalization_mean"] = mean.tolist()
        bundle.metadata["normalization_std"] = std.tolist()
    return bundle


def portable_bundle(path, *, expected_name: str | None = None) -> DatasetBundle:
    """Load a portable NPZ without pickle or fabricated splits.

    Required: name, task, num_classes, y, train_idx, valid_idx, test_idx.
    Node/tabular: x; node: edge_index. Graph: concatenated x/edge_index plus
    graph_ptr and edge_ptr, optional edge_attr. Image: unicode image_paths.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"name", "task", "num_classes", "y", "train_idx", "valid_idx", "test_idx"}
        missing = sorted(required - set(data.files))
        if missing:
            raise DatasetError(f"Portable dataset is missing keys: {missing}")
        name = str(data["name"].item())
        if expected_name is not None and name.replace("-", "_").lower() != expected_name.replace("-", "_").lower():
            raise DatasetError(f"Portable dataset declares {name!r}, expected {expected_name!r}")
        task = str(data["task"].item())
        values = {key: data[key].copy() for key in data.files}
    graphs = None
    if task == "graph":
        for key in ("x", "edge_index", "graph_ptr", "edge_ptr"):
            if key not in values:
                raise DatasetError(f"Molecular NPZ requires {key}")
        gp, ep = values["graph_ptr"], values["edge_ptr"]
        if gp.ndim != 1 or ep.shape != gp.shape or len(gp) != len(values["y"]) + 1:
            raise DatasetError("Graph/edge pointer arrays have incompatible shapes")
        if not np.issubdtype(gp.dtype, np.integer) or not np.issubdtype(ep.dtype, np.integer):
            raise DatasetError("Graph/edge pointer arrays must contain integers")
        if gp[0] != 0 or ep[0] != 0 or gp[-1] != len(values["x"]) or ep[-1] != values["edge_index"].shape[1]:
            raise DatasetError("Graph/edge pointers must cover exactly the concatenated arrays")
        if np.any(np.diff(gp) <= 0) or np.any(np.diff(ep) < 0):
            raise DatasetError("Molecular pointer arrays must be ordered")
        graphs = []
        for i in range(len(gp) - 1):
            attrs = values.get("edge_attr")
            # Portable molecular edge indices are global concatenated node IDs.
            graphs.append(GraphSample(values["x"][gp[i]:gp[i+1]],
                                      values["edge_index"][:, ep[i]:ep[i+1]] - gp[i],
                                      None if attrs is None else attrs[ep[i]:ep[i+1]]).validate())
    metadata = json.loads(str(values["metadata_json"].item())) if "metadata_json" in values else {}
    metadata.update({"source": str(path), "format": "portable_npz"})
    if "image_ids" in values:
        metadata["image_ids"] = values["image_ids"].tolist()
    image_paths = None
    if "image_paths" in values:
        image_paths = [str((path.parent / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(p)
                       for p in values["image_paths"].tolist()]
    result = DatasetBundle(name, task, values["y"],
                           {key: values[key + "_idx"] for key in ("train", "valid", "test")},
                           int(values["num_classes"].item()),
                           x=values.get("x") if task != "graph" else None,
                           edge_index=values.get("edge_index") if task == "node" else None,
                           graphs=graphs, image_paths=image_paths, year=values.get("year"),
                           time=values.get("time"), metadata=metadata)
    return result.validate()


def options_dict(options: Mapping | None) -> dict:
    return dict(options or {})
