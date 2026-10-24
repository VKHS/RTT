"""ImageNet-C JPEG paths with stable class mappings and image-identity isolation."""
import csv
import json
from pathlib import Path
import numpy as np
from .common import DatasetBundle, DatasetError, options_dict, source_split


CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(root: Path, class_to_idx):
    if not root.is_dir():
        raise DatasetError(f"Image directory is missing: {root}")
    result = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in class_to_idx:
            raise DatasetError(f"Unmapped ImageNet class directory: {folder.name}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                result.append((str(path.resolve()), int(class_to_idx[folder.name]), folder.name + "/" + path.name))
    if not result:
        raise DatasetError(f"No images found under {root}")
    return result


def _manifest(path, options):
    path = Path(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"path", "label", "split"}.issubset(rows[0]):
        raise DatasetError("Image CSV requires path,label,split columns; optional image_id and domain")
    paths, labels, domains, image_ids, identities = [], [], [], [], {}
    splits = {key: [] for key in ("train", "valid", "test")}
    for index, row in enumerate(rows):
        split = row["split"]
        if split not in splits:
            raise DatasetError(f"Unknown image split: {split}")
        file = Path(row["path"])
        if not file.is_absolute():
            file = path.parent / file
        if not file.is_file():
            raise DatasetError(f"Image manifest file missing: {file}")
        label = int(row["label"])
        # All corrupted copies of the same underlying image must have one split.
        identity = row.get("image_id") or file.name
        prior = identities.setdefault(identity, (split, label))
        if prior != (split, label):
            raise DatasetError(f"Underlying image {identity} crosses splits or changes label")
        paths.append(str(file.resolve())); labels.append(label); domains.append(row.get("domain", "unspecified"))
        image_ids.append(identity)
        splits[split].append(index)
    if len(set(paths)) != len(paths):
        raise DatasetError("Duplicate image file in manifest")
    classes = int(options.get("num_classes", 1000))
    return DatasetBundle("imagenet_c", "image", np.asarray(labels),
                         {key: np.asarray(value, dtype=np.int64) for key, value in splits.items()},
                         classes, image_paths=paths,
                         metadata={"source": "https://github.com/hendrycks/robustness", "manifest": str(path),
                                   "domains": domains, "image_ids": image_ids,
                                   "split": "explicit_image_identity_manifest", "metric": "accuracy"}).validate()


def load(root, options=None, download=False):
    options = options_dict(options)
    if options.get("manifest"):
        return _manifest(options["manifest"], options)
    root = Path(root)
    fit_root = options.get("fit_root")
    if not fit_root:
        raise DatasetError("ImageNet-C requires a clean disjoint fit_root, or an explicit CSV manifest; corrupted audit images are not fitting data")
    fit_root = Path(fit_root)
    mapping_path = options.get("class_map")
    if mapping_path:
        class_to_idx = json.loads(Path(mapping_path).read_text())
        if not isinstance(class_to_idx, dict):
            raise DatasetError("class_map must be a JSON {wnid: index} object matching the checkpoint")
    else:
        classes = sorted(p.name for p in fit_root.iterdir() if p.is_dir()) if fit_root.exists() else []
        expected = int(options.get("num_classes", 1000))
        if len(classes) != expected:
            raise DatasetError("A full class directory vocabulary or explicit class_map is required; subset folders must not silently reindex labels")
        class_to_idx = {name: index for index, name in enumerate(classes)}
    if len(set(class_to_idx.values())) != len(class_to_idx) or set(class_to_idx.values()) != set(range(len(class_to_idx))):
        raise DatasetError("ImageNet class map must contain unique contiguous indices")
    clean = _images(fit_root, class_to_idx)
    clean_ids = {row[2] for row in clean}
    severity = int(options.get("severity", 3))
    if not 1 <= severity <= 5:
        raise DatasetError("ImageNet-C severity must be in 1..5")
    corruptions = options.get("corruptions", list(CORRUPTIONS))
    if not corruptions or len(corruptions) != len(set(corruptions)) or any(c not in CORRUPTIONS for c in corruptions):
        raise DatasetError("Use a nonempty unique subset of the 15 registered ImageNet-C corruptions")
    test, domains, excluded = [], ["clean"] * len(clean), 0
    for corruption in corruptions:
        for row in _images(root / corruption / str(severity), class_to_idx):
            if row[2] in clean_ids:
                excluded += 1
                continue
            test.append(row); domains.append(corruption)
    if not test:
        raise DatasetError("All corrupted images overlap the clean fitting images. Supply disjoint clean data or a manifest splitting underlying image identities")
    train, valid = source_split(np.arange(len(clean)), seed=int(options.get("seed", 1)),
                                valid_fraction=float(options.get("valid_fraction", 0.2)))
    images = clean + test
    return DatasetBundle("imagenet_c", "image", np.asarray([row[1] for row in images], dtype=np.int64),
                         {"train": train, "valid": valid, "test": np.arange(len(clean), len(images))},
                         len(class_to_idx), image_paths=[row[0] for row in images],
                         metadata={"source": "https://github.com/hendrycks/robustness", "severity": severity,
                                   "corruptions": list(corruptions), "class_to_idx": class_to_idx,
                                   "domains": domains, "image_ids": [row[2] for row in images],
                                   "excluded_overlapping_corruptions": excluded,
                                   "split": "clean_source_train_valid__disjoint_corruption_test",
                                   "metric": "accuracy"}).validate()
