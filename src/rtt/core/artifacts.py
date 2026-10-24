"""Experiment directories and hashed artifact persistence."""
from __future__ import annotations

from ._imports import (
    Any,
    Path,
    Tuple,
    argparse,
    dataclasses,
    os,
    torch,
)
from . import api as _api

# ============================================================================
# Section 16: suite layout, artifacts, task lists and stage runners
# ============================================================================

STAGES = ["incumbent", "proposal", "phase2", "placement", "calibration", "selection",
          "development", "audit", "breadth", "node_transfer", "diagnostics", "reduce"]
STAGE_DIRS = {
    "incumbent": "010_INCUMBENT", "proposal": "020_PROPOSAL", "phase2": "030_PHASE2",
    "placement": "040_PLACEMENT", "calibration": "050_CALIBRATION", "selection": "060_SELECTION",
    "development": "070_DEVELOPMENT", "audit": "080_AUDIT", "breadth": "085_BREADTH",
    "node_transfer": "090_NODE_TRANSFER", "diagnostics": "095_DIAGNOSTICS", "reduce": "900_REDUCE",
}
EXTERNAL_PROPOSALS = ["v2", "lora16", "gprgnn"]
NOT_IMPLEMENTED_PROPOSALS = ["Co-GNN (adapter)", "AMP (adapter)", "GTrans (test-time adaptation)",
                             "Matcha (test-time adaptation)", "TSA", "STEM"]


class Suite:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.art = self.root / "ARTIFACTS"
        self.manifest = self.root / "000_MANIFEST"

    def stage_dir(self, stage: str) -> Path:
        return self.root / STAGE_DIRS[stage]

    def run_dir(self, stage: str, task_id: str) -> Path:
        return self.stage_dir(stage) / "runs" / task_id / "RESULT"

    def artifact(self, name: str) -> Path:
        return self.art / name

    def publish(self, name: str, obj: Any, kind: str = "torch") -> dict:
        path = self.artifact(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        if kind == "torch":
            torch.save(obj, tmp)
        else:
            tmp.write_text(_api.to_json(obj) + "\n")
        os.replace(tmp, path)
        digest = _api.sha256_file(path)
        man = {"artifact": name, "sha256": digest, "bytes": path.stat().st_size, "created_utc": _api.now_utc(),
               "version": _api.VERSION}
        _api.write_json(path.with_name(path.name + ".manifest.json"), man)
        return man

    def load(self, name: str, kind: str = "torch", verify: bool = True):
        path = self.artifact(name)
        if not path.is_file():
            raise _api.RTTFailure("ARTIFACT_MISSING", f"artifact missing: {path}")
        mf = path.with_name(path.name + ".manifest.json")
        if verify and mf.is_file():
            m = _api.read_json(mf)
            if m.get("sha256") != _api.sha256_file(path):
                raise _api.RTTFailure("ARTIFACT_MISSING", f"artifact hash mismatch: {path}")
        if kind == "torch":
            return torch.load(path, map_location="cpu", weights_only=False)
        return _api.read_json(path)

    def has(self, name: str) -> bool:
        return self.artifact(name).is_file()

    def config(self) -> Tuple[_api.RTTConfig, dict]:
        obj = _api.read_json(self.manifest / "rtt_config.json")
        cfg = _api.RTTConfig()
        for k, v in obj["config"].items():
            if hasattr(cfg, k):
                cur = getattr(cfg, k)
                if isinstance(cur, tuple) and isinstance(v, list):
                    v = tuple(v)
                setattr(cfg, k, v)
        return cfg, obj


def write_suite_config(suite: Suite, cfg: _api.RTTConfig, args: argparse.Namespace) -> None:
    dataset_keys = ("dataset_kind", "dataset_name", "dataset_root", "graph_pt", "rtt_graph_npz",
                    "external_feature_pt", "feature_normalization", "use_node_year",
                    "synthetic_nodes", "synthetic_classes", "synthetic_features", "synthetic_seed")
    obj = {"config": cfg.to_dict(), "version": _api.VERSION, "version_tag": _api.VERSION_TAG, "created_utc": _api.now_utc(),
           "dataset_kind": args.dataset_kind, "scale": cfg.scale_name,
           "dataset_arguments": {key: getattr(args, key, None) for key in dataset_keys},
           "library": [dataclasses.asdict(m) for m in _api.default_library()],
           "implementation_choices": _api.IMPLEMENTATION_CHOICES}
    _api.write_json(suite.manifest / "rtt_config.json", obj)


