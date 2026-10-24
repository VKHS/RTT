"""Molecular graphs, categorical atom/bond features and masked 128-task labels."""
from pathlib import Path
from collections.abc import Sequence
import csv
import hashlib
import json
import numpy as np
from .common import DatasetBundle, DatasetError, GraphSample


class OGBMoleculeSequence(Sequence):
    """Avoid duplicating all OGB molecular arrays in a second Python graph list."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        graph, _ = self.dataset[int(index)]
        return GraphSample(np.asarray(graph["node_feat"], dtype=np.int64),
                           np.asarray(graph["edge_index"], dtype=np.int64),
                           np.asarray(graph["edge_feat"], dtype=np.int64)).validate()


def attach_scaffolds(bundle, path):
    """Attach externally derived scaffold IDs in exact OGB graph-index order.

    CSV: graph_id,scaffold_id, exactly once for every graph. JSON: a list in
    dataset order, or an object mapping decimal graph IDs to scaffold IDs.
    Actual scaffolds must be derived from the matching official molecules; this
    loader never treats a split label or a random group as a chemical scaffold.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise DatasetError(f"Scaffold identity manifest is missing: {path}")
    identities = [None] * bundle.n
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"graph_id", "scaffold_id"}.issubset(reader.fieldnames or []):
                raise DatasetError("Scaffold CSV requires graph_id and scaffold_id columns")
            entries = [(row["graph_id"], row["scaffold_id"]) for row in reader]
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if isinstance(value, list):
            entries = list(enumerate(value))
        elif isinstance(value, dict):
            entries = list(value.items())
        else:
            raise DatasetError("Scaffold JSON must be a list or graph_id-to-scaffold_id object")
    else:
        raise DatasetError("Scaffold identity manifest must be .csv or .json")
    for raw_index, identity in entries:
        try:
            index = int(raw_index)
        except (TypeError, ValueError, OverflowError) as error:
            raise DatasetError(f"Invalid scaffold graph_id: {raw_index!r}") from error
        if str(index) != str(raw_index).strip() or index < 0 or index >= bundle.n:
            raise DatasetError(f"Scaffold graph_id outside exact dataset indices: {raw_index!r}")
        if identities[index] is not None:
            raise DatasetError(f"Duplicate scaffold graph_id: {index}")
        if identity is None or isinstance(identity, (list, dict, bool)) or not str(identity).strip():
            raise DatasetError("Scaffold IDs must be nonempty scalar identities; use an explicit shared ID for an empty acyclic scaffold")
        if isinstance(identity, float) and not np.isfinite(identity):
            raise DatasetError("Scaffold IDs must be finite")
        identities[index] = str(identity)
    if any(identity is None for identity in identities):
        raise DatasetError("Scaffold identity manifest must cover every dataset graph exactly once")
    bundle.metadata.update(scaffold_ids=identities, scaffold_manifest=str(path),
                           scaffold_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           scaffold_provenance="user-supplied chemical scaffold identities; derivation must match the official molecule order")
    return bundle


def load(root, options=None, download=False):
    try:
        from ogb.graphproppred import GraphPropPredDataset
    except ImportError as error:
        raise DatasetError("Install the data extra: pip install '.[data]' (ogb)") from error
    root = Path(root)
    if not download and not (root / "ogbg_molpcba").exists():
        raise DatasetError(f"ogbg-molpcba is absent from {root}; enable download explicitly")
    dataset = GraphPropPredDataset(name="ogbg-molpcba", root=str(root))
    split = dataset.get_idx_split()
    y = np.asarray(dataset.labels, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != 128:
        raise DatasetError("ogbg-molpcba must expose 128 binary assay tasks")
    # Molecules with no observed assay are context-only, never scored as negatives.
    observed = np.isfinite(y).any(axis=1)
    splits = {key: np.asarray(split[key], dtype=np.int64) for key in ("train", "valid", "test")}
    splits = {key: idx[observed[idx]] for key, idx in splits.items()}
    bundle = DatasetBundle("ogbg_molpcba", "graph", y, splits, 128, graphs=OGBMoleculeSequence(dataset),
                         metadata={"source": "https://ogb.stanford.edu/docs/graphprop/", "split": "official_scaffold",
                                   "metric": "mean_average_precision", "missing_label": "NaN",
                                   "categorical_atom_features": True, "categorical_bond_features": True,
                                   "atom_cardinalities": [119,5,12,12,10,6,6,2,2],
                                   "bond_cardinalities": [5,6,2]}).validate()
    if (options or {}).get("scaffold_path"):
        attach_scaffolds(bundle, options["scaffold_path"])
    return bundle
