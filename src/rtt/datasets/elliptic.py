"""Original Elliptic transaction CSVs with explicit temporal boundaries."""
from pathlib import Path
import csv
import numpy as np
from .common import DatasetBundle, DatasetError, options_dict, standardize_train


def load(root, options=None, download=False):
    root = Path(root); options = options_dict(options)
    features_file = root / "elliptic_txs_features.csv"
    labels_file = root / "elliptic_txs_classes.csv"
    edges_file = root / "elliptic_txs_edgelist.csv"
    for file in (features_file, labels_file, edges_file):
        if not file.is_file():
            raise DatasetError(f"Missing Elliptic CSV: {file}; obtain the original dataset from its official distribution")
    features = np.loadtxt(features_file, delimiter=",", dtype=np.float64, ndmin=2)
    if features.shape[1] < 3:
        raise DatasetError("Elliptic features require transaction ID, time step, and features")
    ids = features[:, 0].astype(np.int64)
    if len(np.unique(ids)) != len(ids) or np.any(features[:, 0] != ids):
        raise DatasetError("Invalid/duplicate Elliptic transaction IDs")
    mapping = {str(int(txid)): i for i, txid in enumerate(ids)}
    time = features[:, 1].astype(np.int64)
    if np.any(time != features[:, 1]) or np.any(time < 1):
        raise DatasetError("Elliptic time steps must be positive integers")
    y = np.full(len(ids), -1, dtype=np.int64)
    with labels_file.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["txId"] not in mapping:
                raise DatasetError("Elliptic label transaction absent from feature table")
            value = row["class"].strip()
            if value not in {"1", "2", "unknown"}:
                raise DatasetError(f"Unknown Elliptic class code {value}")
            y[mapping[row["txId"]]] = {"1": 1, "2": 0, "unknown": -1}[value]
    with edges_file.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    try:
        edges = np.asarray([[mapping[row["txId1"]], mapping[row["txId2"]]] for row in rows],
                           dtype=np.int64).reshape(-1, 2).T
    except KeyError as error:
        raise DatasetError("Elliptic edge schema or endpoint does not match feature IDs") from error
    if edges.size and np.any(time[edges[0]] != time[edges[1]]):
        raise DatasetError("Original Elliptic should have no edges between time steps; a modified graph needs a separate causal protocol")
    train_end = int(options.get("train_end", 34))
    valid_end = int(options.get("valid_end", 42))
    if not 0 < train_end < valid_end:
        raise DatasetError("Require 0 < train_end < valid_end")
    known = y >= 0
    splits = {"train": np.flatnonzero(known & (time <= train_end)),
              "valid": np.flatnonzero(known & (time > train_end) & (time <= valid_end)),
              "test": np.flatnonzero(known & (time > valid_end))}
    bundle = DatasetBundle("elliptic", "node", y, splits, 2, x=features[:, 2:].astype(np.float32),
                           edge_index=edges, time=time,
                           metadata={"source": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
                                     "train_end": train_end, "valid_end": valid_end,
                                     "split": "time_steps", "positive_class": "illicit", "metric": "f1",
                                     "unknown_label": -1, "time_feature_excluded": True,
                                     "boundary_provenance": "explicit reconstruction configuration"})
    return standardize_train(bundle.validate(), options.get("standardize", True))
