"""GOOD-Arxiv's released masks; never approximate GOOD by relabeling OGB splits."""
from pathlib import Path
from collections.abc import Mapping
import numpy as np
from .common import DatasetBundle, DatasetError, options_dict, standardize_train


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _field(data, key, default=None):
    return data.get(key, default) if isinstance(data, Mapping) else getattr(data, key, default)


def bundle_from_good(data, domain, shift, options=None):
    options = options_dict(options)
    # GOOD's loader returns a one-graph dataset. Trusted exported tuples may
    # contain (Data, slices), while public loader returns (dataset, meta_info).
    if isinstance(data, tuple):
        data = data[0]
    if _field(data, "x") is None:
        try:
            data = data[0]
        except (KeyError, TypeError, IndexError) as error:
            raise DatasetError("GOOD export must contain a graph with x/y/edge_index and released masks") from error
    y = _array(_field(data, "y")).reshape(-1)
    split_keys = {"train": "train_mask", "valid": options.get("valid_mask", "val_mask"), "test": "test_mask"}
    splits = {}
    for split, key in split_keys.items():
        value = _field(data, key)
        if value is None:
            raise DatasetError(f"GOOD export lacks required official mask {key}")
        mask = _array(value)
        if mask.ndim != 1 or mask.shape != y.shape or mask.dtype != bool:
            raise DatasetError(f"GOOD {key} must be a Boolean vector; masks are not regenerated")
        splits[split] = np.flatnonzero(mask & (y >= 0))
    for key in ("id_val_mask", "id_test_mask"):
        mask = _field(data, key)
        if mask is not None:
            splits[key.removesuffix("_mask")] = np.flatnonzero(_array(mask))
    year = _field(data, "node_year", _field(data, "year"))
    known = y[y >= 0]
    num_classes = int(options.get("num_classes", max(2, int(known.max(initial=0)) + 1)))
    result = DatasetBundle(f"good_arxiv_{domain}", "node", y, splits, num_classes,
                           x=_array(_field(data, "x")).astype(np.float32),
                           edge_index=_array(_field(data, "edge_index")).astype(np.int64),
                           year=None if year is None else _array(year).reshape(-1).astype(np.int64),
                           metadata={"source": "https://github.com/divelab/GOOD", "domain": domain,
                                     "shift": shift, "split": "official_GOOD_masks",
                                     "validation_mask": split_keys["valid"], "metric": "accuracy"})
    return standardize_train(result.validate(), options.get("standardize", True))


def load(root, options=None, download=False, *, domain="degree"):
    options = options_dict(options)
    expected_shift = "covariate" if domain == "degree" else "concept"
    shift = options.get("shift", expected_shift)
    if shift != expected_shift:
        raise DatasetError(f"good_arxiv_{domain} is registered for shift={expected_shift!r}; use a separate experiment for {shift!r}")
    path = options.get("data_pt")
    if path:
        if not options.get("trust_data_pickle", False):
            raise DatasetError("GOOD .pt exports may contain Python objects; explicitly set trust_data_pickle=true for a trusted official export")
        import torch
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
    else:
        root = Path(root)
        if not download and not any(root.glob(f"**/{domain}/processed/*.pt")):
            raise DatasetError("No cached GOOD-Arxiv processed tensors. Enable download or supply data_pt; portable NPZ is also supported")
        try:
            from GOOD.data.good_datasets.good_arxiv import GOODArxiv
        except ImportError as error:
            raise DatasetError("Install the official GOOD package and PyG, or export its exact graph/masks to portable NPZ") from error
        data, _metadata = GOODArxiv.load(dataset_root=str(root), domain=domain, shift=shift, generate=False)
    return bundle_from_good(data, domain, shift, options)
