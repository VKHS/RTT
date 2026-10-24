"""Separate loaders for every dataset family used by the RTT manuscript."""
from importlib import import_module
from pathlib import Path
from .common import DatasetBundle, DatasetError, GraphSample, portable_bundle


DATASETS = (
    "ogbn_arxiv", "good_arxiv_degree", "good_arxiv_time", "twitch", "facebook100",
    "elliptic", "ogbn_products", "imagenet_c", "acs_income", "ogbg_molpcba",
)
ALIASES = {"arxiv": "ogbn_arxiv", "products": "ogbn_products", "molpcba": "ogbg_molpcba",
           "twitch_explicit": "twitch", "facebook_100": "facebook100", "fb100": "facebook100",
           "folktables": "acs_income", "acsincome": "acs_income", "imagenetc": "imagenet_c"}


def canonical_name(name: str) -> str:
    value = name.lower().replace("-", "_")
    value = ALIASES.get(value, value)
    if value not in DATASETS:
        raise DatasetError(f"Unknown dataset {name!r}; choose one of {', '.join(DATASETS)}")
    return value


def load_dataset(name: str, root, options=None, download: bool = False) -> DatasetBundle:
    name = canonical_name(name)
    options = dict(options or {})
    path = options.get("npz") or (root if Path(root).suffix.lower() == ".npz" else None)
    if path is not None:
        bundle = portable_bundle(path)
        if canonical_name(bundle.name) != name:
            raise DatasetError(f"Portable dataset {bundle.name!r} cannot be used as {name!r}")
        expected = "graph" if name == "ogbg_molpcba" else "image" if name == "imagenet_c" else "tabular" if name == "acs_income" else "node"
        if bundle.task != expected:
            raise DatasetError(f"Dataset {name} requires task={expected!r}")
        if name == "ogbg_molpcba":
            if not bundle.multilabel or bundle.num_classes != 128:
                raise DatasetError("ogbg-molpcba requires 128 binary assay columns, preserving missing labels as NaN")
            for index in {0, bundle.n - 1}:
                sample = bundle.graphs[index]
                if sample.x.shape[1] != 9 or sample.edge_attr is None or sample.edge_attr.shape[1] != 3:
                    raise DatasetError("ogbg-molpcba requires nine categorical atom and three categorical bond features")
            bundle.metadata.update(categorical_atom_features=True, categorical_bond_features=True)
            if options.get("scaffold_path"):
                from .ogbg_molpcba import attach_scaffolds
                attach_scaffolds(bundle, options["scaffold_path"])
        bundle.name = name
        return bundle
    return import_module("." + name, __name__).load(root, options, bool(download))


__all__ = ["DATASETS", "DatasetBundle", "DatasetError", "GraphSample", "canonical_name", "load_dataset", "portable_bundle"]
