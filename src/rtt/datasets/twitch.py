"""Twitch-explicit language graphs in the original MUSAE CSV/JSON format."""
import csv
import json
from pathlib import Path
import numpy as np
from .common import DatasetBundle, DatasetError, options_dict, source_split


def _domain(root: Path, name: str):
    folder = root / name
    prefix = "musae_" + name
    targets = folder / (prefix + "_target.csv")
    features = folder / (prefix + "_features.json")
    edges = folder / (prefix + "_edges.csv")
    for file in (targets, features, edges):
        if not file.is_file():
            raise DatasetError(f"Missing Twitch file: {file}; extract the official language archive")
    with targets.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "mature" not in rows[0] or "new_id" not in rows[0]:
        raise DatasetError("Twitch target CSV requires new_id and mature columns")
    ids = [int(row["new_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise DatasetError("Duplicate Twitch node IDs")
    order = np.argsort(ids)
    ids = np.asarray(ids)[order]
    mapping = {int(value): i for i, value in enumerate(ids)}
    truth = {"true": 1, "false": 0, "1": 1, "0": 0}
    try:
        y = np.asarray([truth[rows[i]["mature"].strip().lower()] for i in order], dtype=np.int64)
    except KeyError as error:
        raise DatasetError("Twitch mature values must be Boolean") from error
    raw = json.loads(features.read_text())
    if any(int(key) not in mapping for key in raw):
        raise DatasetError("Twitch feature key absent from target node IDs")
    bags = [raw.get(str(int(node)), []) for node in ids]
    with edges.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        edge_rows = list(reader)
    if header is None:
        raise DatasetError("Empty Twitch edge CSV")
    try:
        edge_index = np.asarray([[mapping[int(row[0])], mapping[int(row[1])]] for row in edge_rows],
                                dtype=np.int64).reshape(-1, 2).T
    except KeyError as error:
        raise DatasetError("Twitch edge refers to absent node") from error
    return y, bags, edge_index


def load(root, options=None, download=False):
    options = options_dict(options)
    source = options.get("source_domain")
    targets = options.get("target_domains") or ([options["target_domain"]] if options.get("target_domain") else [])
    if not source or not targets or isinstance(targets, str):
        raise DatasetError("Specify source_domain and target_domains for Twitch cross-graph evaluation")
    names = [source] + list(targets)
    if len(names) != len(set(names)):
        raise DatasetError("Twitch source and target domains must be distinct")
    domains = [_domain(Path(root), name) for name in names]
    # Feature IDs are shared vocabulary IDs in the original files, not locally
    # reindexed categories. Preserve their correspondence across languages.
    max_feature = max((int(value) for _, bags, _ in domains for bag in bags for value in bag), default=-1)
    width = int(options.get("num_features", max_feature + 1))
    if width <= max_feature or width < 1:
        raise DatasetError("num_features must cover every original Twitch feature ID")
    xs, ys, edges, labels, offset = [], [], [], [], 0
    for name, (y, bags, edge_index) in zip(names, domains):
        x = np.zeros((len(y), width), dtype=np.float32)
        for row, bag in enumerate(bags):
            features = np.asarray(bag, dtype=np.int64)
            if features.size and features.min() < 0:
                raise DatasetError("Twitch feature IDs must be nonnegative")
            x[row, features] = 1.0
        xs.append(x); ys.append(y); edges.append(edge_index + offset)
        labels.extend([name] * len(y)); offset += len(y)
    train, valid = source_split(np.arange(len(domains[0][0])), seed=int(options.get("seed", 1)),
                                valid_fraction=float(options.get("valid_fraction", 0.2)))
    return DatasetBundle("twitch", "node", np.concatenate(ys),
                         {"train": train, "valid": valid, "test": np.arange(len(domains[0][0]), offset)},
                         2, x=np.concatenate(xs), edge_index=np.concatenate(edges, axis=1),
                         metadata={"source": "https://snap.stanford.edu/data/twitch-social-networks.html",
                                   "source_domain": source, "target_domains": list(targets), "domains": labels,
                                   "split": "source_random_train_valid__target_test",
                                   "split_seed": int(options.get("seed", 1)), "metric": "roc_auc"}).validate()
