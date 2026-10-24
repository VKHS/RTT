"""Official ogbn-arxiv graph, labels, years and chronological splits."""
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
    if not download and not (root / "ogbn_arxiv").exists():
        raise DatasetError(f"ogbn-arxiv is absent from {root}; enable download explicitly")
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=str(root))
    graph, y = dataset[0]
    split = dataset.get_idx_split()
    year = np.asarray(graph["node_year"]).reshape(-1).astype(np.int64)
    bundle = DatasetBundle("ogbn_arxiv", "node", np.asarray(y).reshape(-1),
                           {key: np.asarray(split[key], dtype=np.int64) for key in ("train", "valid", "test")},
                           int(dataset.num_classes), x=np.asarray(graph["node_feat"], dtype=np.float32),
                           edge_index=np.asarray(graph["edge_index"], dtype=np.int64), year=year,
                           metadata={"source": "https://ogb.stanford.edu/docs/nodeprop/", "split": "official_time",
                                     "directed": True, "metric": "accuracy"})
    return standardize_train(bundle.validate(), options.get("standardize", True))
