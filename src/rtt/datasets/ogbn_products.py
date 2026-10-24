"""Official ogbn-products graph and sales-rank split; no fabricated years."""
from pathlib import Path
import numpy as np
from .common import DatasetBundle, DatasetError, options_dict, standardize_train


def load(root, options=None, download=False):
    options = options_dict(options)
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as error:
        raise DatasetError("Install the data extra: pip install '.[data]' (ogb)") from error
    root = Path(root)
    if not download and not (root / "ogbn_products").exists():
        raise DatasetError(f"ogbn-products is absent from {root}; enable download explicitly")
    dataset = NodePropPredDataset(name="ogbn-products", root=str(root))
    graph, y = dataset[0]
    split = dataset.get_idx_split()
    bundle = DatasetBundle("ogbn_products", "node", np.asarray(y).reshape(-1),
                           {key: np.asarray(split[key], dtype=np.int64) for key in ("train", "valid", "test")},
                           int(dataset.num_classes), x=np.asarray(graph["node_feat"], dtype=np.float32),
                           edge_index=np.asarray(graph["edge_index"], dtype=np.int64),
                           metadata={"source": "https://ogb.stanford.edu/docs/nodeprop/", "split": "official_sales_rank",
                                     "directed": False, "metric": "accuracy"})
    return standardize_train(bundle.validate(), options.get("standardize", True))
