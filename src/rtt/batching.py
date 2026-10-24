"""Task-aware batches; labels can be removed before admission."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .datasets.common import DatasetBundle, GraphSample
from .runtime import seed_for


@dataclass
class Batch:
    x: torch.Tensor
    context: dict
    item_ids: np.ndarray
    mask: torch.Tensor
    multilabel: bool
    query: torch.Tensor | None = None
    labels: torch.Tensor | None = None

    @property
    def device(self):
        return self.x.device

    @property
    def units(self):
        return int(self.mask.sum())

    def unlabeled(self):
        return replace(self, labels=None, context=dict(self.context))

    def select(self, logits):
        return logits if self.query is None else logits[self.query]


class BatchFactory:
    def __init__(self, bundle, device="cpu", image_size=224):
        self.bundle, self.device = bundle, torch.device(device)
        self.image_size = image_size
        self.node_x = self.edges = None
        if bundle.task == "node":
            self.node_x = torch.as_tensor(bundle.x, dtype=torch.float32, device=device)
            self.edges = torch.as_tensor(bundle.edge_index, dtype=torch.long, device=device)

    def _image(self, path):
        from PIL import Image
        with Image.open(path) as image:
            image = image.convert("RGB")
            shorter = max(self.image_size, round(self.image_size * 256 / 224))
            width, height = image.size
            scale = shorter / min(width, height)
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
            width, height = image.size
            left, top = (width - self.image_size) // 2, (height - self.image_size) // 2
            array = np.asarray(image.crop((left, top, left + self.image_size, top + self.image_size))).copy()
        value = torch.from_numpy(array).permute(2, 0, 1).float() / 255
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return (value - mean) / std

    def make(self, item_ids, *, labels=True):
        ids = np.asarray(item_ids, dtype=np.int64)
        if not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("Each call needs distinct, nonempty item IDs")
        b, device = self.bundle, self.device
        observed = np.isfinite(b.y[ids]) if b.multilabel else np.ones(len(ids), dtype=bool)
        target = torch.as_tensor(b.y[ids].copy(), device=device) if labels else None
        context, query = {}, None
        if b.task == "node":
            x = self.node_x
            context["edge_index"] = self.edges
            query = torch.as_tensor(ids.copy(), device=device)
        elif b.task == "tabular":
            x = torch.as_tensor(b.x[ids], dtype=torch.float32, device=device)
        elif b.task == "image":
            x = torch.stack([self._image(b.image_paths[index]) for index in ids]).to(device)
        else:
            features, edges, attrs, membership, offset = [], [], [], [], 0
            for graph_index, index in enumerate(ids):
                graph = b.graphs[int(index)].validate()
                features.append(torch.as_tensor(graph.x))
                edges.append(torch.as_tensor(graph.edge_index) + offset)
                membership.append(torch.full((len(graph.x),), graph_index, dtype=torch.long))
                if graph.edge_attr is not None:
                    attrs.append(torch.as_tensor(graph.edge_attr))
                offset += len(graph.x)
            x = torch.cat(features).to(device)
            context = {"edge_index": torch.cat(edges, dim=1).to(device),
                       "batch": torch.cat(membership).to(device), "num_graphs": len(ids)}
            if attrs:
                if len(attrs) != len(ids):
                    raise ValueError("Every graph in a batch must have compatible edge attributes")
                context["edge_attr"] = torch.cat(attrs).to(device)
        return Batch(x, context, ids, torch.as_tensor(observed, device=device), b.multilabel, query, target)

    def sample(self, pool, size, seed, *, labels=True):
        ids = np.random.default_rng(seed).choice(pool, size=min(size, len(pool)), replace=False)
        return self.make(ids, labels=labels)


def synthetic_bundle(name, task, directory, *, seed=1, classes=3, image_size=32):
    """Explicit fixtures used by smoke runs; never reported as benchmark results."""
    rng = np.random.default_rng(seed_for("fixture", seed, name))
    n = 96 if task != "image" else 24
    splits = {"train": np.arange(0, n // 2), "valid": np.arange(n // 2, 3 * n // 4), "test": np.arange(3 * n // 4, n)}
    y = rng.integers(classes, size=n, dtype=np.int64)
    metadata = {"synthetic": True, "source": "generated smoke fixture", "benchmark_results": False}
    if task in {"node", "tabular"}:
        x = rng.normal(size=(n, 8)).astype(np.float32)
        edges = np.stack((np.arange(n), np.roll(np.arange(n), -1)))
        edges = np.concatenate((edges, edges[::-1]), axis=1)
        return DatasetBundle(name, task, y, splits, classes, x=x,
                             edge_index=edges if task == "node" else None, metadata=metadata).validate()
    if task == "graph":
        graphs = []
        for _ in range(n):
            nodes = 5
            x = np.zeros((nodes, 9), dtype=np.int64)
            x[:, 0] = rng.integers(1, 9, nodes)
            edges = np.stack((np.arange(nodes - 1), np.arange(1, nodes)))
            edges = np.concatenate((edges, edges[::-1]), axis=1)
            graphs.append(GraphSample(x, edges, np.zeros((edges.shape[1], 3), dtype=np.int64)))
        y = rng.integers(0, 2, size=(n, 128)).astype(np.float32)
        y[rng.random(y.shape) < 0.3] = np.nan
        return DatasetBundle(name, task, y, splits, 128, graphs=graphs, metadata=metadata).validate()
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(n):
        path = directory / f"image_{index:04d}.png"
        if not path.exists():
            pixels = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
        else:
            # Advance the generator identically when replaying a stored fixture.
            rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
        paths.append(str(path.resolve()))
    return DatasetBundle(name, task, y, splits, classes, image_paths=paths, metadata=metadata).validate()
