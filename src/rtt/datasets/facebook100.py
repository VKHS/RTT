"""Facebook100 MATLAB graphs with gender excluded from the feature matrix."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .common import DatasetBundle, DatasetError, options_dict, source_split


def load(root, options=None, download=False):
    options = options_dict(options)
    source = options.get("source_domain")
    targets = options.get("target_domains") or ([options["target_domain"]] if options.get("target_domain") else [])
    if not source or not targets or isinstance(targets, str):
        raise DatasetError("Specify source_domain and target_domains for Facebook100 cross-graph evaluation")
    names = [source] + list(targets)
    if len(names) != len(set(names)):
        raise DatasetError("Facebook100 source and target domains must be distinct")
    label_column = int(options.get("label_column", 1))
    raw_features, ys, edges, domains, offset = [], [], [], [], 0
    for name in names:
        file = Path(root) / (name + ".mat")
        if not file.is_file():
            raise DatasetError(f"Missing Facebook100 file: {file}")
        data = loadmat(file)
        if "A" not in data or "local_info" not in data:
            raise DatasetError("Facebook100 MATLAB file requires A and local_info")
        info = np.asarray(data["local_info"], dtype=np.int64)
        if not 0 <= label_column < info.shape[1]:
            raise DatasetError("Invalid Facebook100 label_column")
        if data["A"].shape != (len(info), len(info)):
            raise DatasetError("Facebook100 adjacency and metadata size differ")
        labels = info[:, label_column] - 1
        if np.any((labels < -1) | (labels > 1)):
            raise DatasetError("This profile expects gender codes 0=unknown,1,2")
        raw_features.append(np.delete(info, label_column, axis=1))
        ys.append(labels)
        edges.append(np.asarray(data["A"].nonzero(), dtype=np.int64) + offset)
        domains.extend([name] * len(info)); offset += len(info)
    source_n = len(ys[0])
    train, valid = source_split(np.flatnonzero(ys[0] >= 0), seed=int(options.get("seed", 1)),
                                valid_fraction=float(options.get("valid_fraction", 0.2)))
    all_features = np.concatenate(raw_features)
    encoded, vocabulary = [], []
    for column in range(all_features.shape[1]):
        # Fixed shared source-fitted encoding, with an explicit unseen category.
        categories = np.unique(raw_features[0][train, column])
        vocabulary.append(categories.tolist())
        block = np.zeros((offset, len(categories) + 1), dtype=np.float32)
        mapping = {value: j for j, value in enumerate(categories)}
        for row, value in enumerate(all_features[:, column]):
            block[row, mapping.get(value, len(categories))] = 1.0
        encoded.append(block)
    y = np.concatenate(ys)
    test = np.flatnonzero((np.arange(offset) >= source_n) & (y >= 0))
    return DatasetBundle("facebook100", "node", y, {"train": train, "valid": valid, "test": test},
                         2, x=np.concatenate(encoded, axis=1), edge_index=np.concatenate(edges, axis=1),
                         metadata={"source": "https://github.com/CUAI/Non-Homophily-Large-Scale",
                                   "source_domain": source, "target_domains": list(targets), "domains": domains,
                                   "label_column": label_column, "categorical_vocabulary": vocabulary,
                                   "split": "source_random_train_valid__target_test", "metric": "accuracy",
                                   "unknown_label": -1, "gender_feature_excluded": True}).validate()
