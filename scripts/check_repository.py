#!/usr/bin/env python3
"""Check dataset interfaces, source syntax, and distributable contents offline."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "ogbn_arxiv", "good_arxiv_degree", "good_arxiv_time", "twitch", "facebook100",
    "elliptic", "ogbn_products", "imagenet_c", "acs_income", "ogbg_molpcba",
)


def source_files():
    paths = list(ROOT.glob("*.py")) + list(ROOT.glob("*.sh"))
    for name in ("src", "scripts", "tests", "experiments"):
        for path in (ROOT / name).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh"} and "__pycache__" not in path.parts:
                paths.append(path)
    return sorted(set(paths))


def check_sources():
    required = ("README.md", "pyproject.toml", "CITATION.cff", "CONTRIBUTING.md", "SOURCE_PROVENANCE.json",
                "docs/ARRHENIUS.md", "docs/DATA_SOURCES.md", "docs/DATASET_PROTOCOLS.md",
                "src/rtt/cli.py", "scripts/arrhenius.sh", "scripts/setup_arrhenius.sh",
                "configs/arxiv.json", "requirements-arrhenius.txt", "requirements-cpu.txt",
                "experiments/ogbn_arxiv/run_native.py", "experiments/ogbn_arxiv/run_native.sh",
                ".github/workflows/tests.yml")
    for name in required:
        if not (ROOT / name).is_file():
            raise RuntimeError(f"Missing repository file: {name}")
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if metadata["project"]["scripts"].get("rtt") != "rtt.cli:main":
        raise RuntimeError("The installed rtt command must resolve to rtt.cli:main")
    sys.path.insert(0, str(ROOT / "src"))
    from rtt.configuration import resolve_config
    from rtt.cli import main as cli_main
    if not callable(cli_main):
        raise RuntimeError("The installed command entry point is not callable")
    configurations = 0
    for name in DATASETS:
        directory = ROOT / "experiments" / name
        for filename in ("config.json", "run.py", "run.sh", "README.md"):
            if not (directory / filename).is_file():
                raise RuntimeError(f"Missing dataset entry point: {name}/{filename}")
        configuration = resolve_config(directory / "config.json", profile="full")
        if configuration["dataset"]["name"].lower().replace("-", "_") != name:
            raise RuntimeError(f"Dataset identity does not match its directory: {name}")
        for path in (directory / "config.json", *sorted((directory / "variants").glob("*.json"))):
            for profile in ("full", "pilot", "smoke"):
                variant = resolve_config(path, profile=profile)
                if variant["dataset"]["name"].lower().replace("-", "_") != name:
                    raise RuntimeError(f"Variant dataset identity does not match: {path}")
            configurations += 1
    python_count = shell_count = 0
    for path in source_files():
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
            python_count += 1
        else:
            subprocess.run(["bash", "-n", str(path)], check=True)
            shell_count += 1
    return {"dataset_interfaces": len(DATASETS), "full_configurations": configurations,
            "profile_resolutions": 3 * configurations, "python_syntax": python_count,
            "shell_syntax": shell_count, "benchmark_downloads": 0, "slurm_submissions": 0}


def check_archives(directory):
    directory = Path(directory)
    wheels = sorted(directory.glob("*.whl"))
    sources = sorted(directory.glob("*.tar.gz"))
    if not wheels or not sources:
        raise RuntimeError("Expected a wheel and a source distribution from python -m build")
    required_wheel = {p.relative_to(ROOT / "src").as_posix()
                      for p in (ROOT / "src/rtt").rglob("*.py") if "__pycache__" not in p.parts}
    required_source = {"README.md", "pyproject.toml", "CITATION.cff", "SOURCE_PROVENANCE.json", "docs/ARRHENIUS.md",
                       "configs/arxiv.json", "requirements-arrhenius.txt", "requirements-cpu.txt",
                       "experiments/ogbn_arxiv/run_native.py", "experiments/ogbn_arxiv/run_native.sh"}
    required_source.update(f"experiments/{name}/{filename}"
                           for name in DATASETS for filename in ("config.json", "run.py", "run.sh", "README.md"))
    required_source.update(path.relative_to(ROOT).as_posix()
                           for path in (ROOT / "experiments").glob("*/variants/*.json"))
    checked = []
    for archive in wheels + sources:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as source:
                members = set(source.namelist())
                bad_member = source.testzip()
                if bad_member:
                    raise RuntimeError(f"Corrupted wheel member: {bad_member}")
            required = required_wheel
        else:
            with tarfile.open(archive, "r:gz") as source:
                members = {"/".join(Path(item.name).parts[1:]) for item in source.getmembers() if item.isfile()}
            required = required_source
        missing = required - members
        if missing:
            raise RuntimeError(f"Archive {archive.name} lacks required files: {sorted(missing)}")
        prohibited = [name for name in members if "__pycache__" in Path(name).parts
                      or ".git" in Path(name).parts or name.endswith((".pt", ".pth", ".ckpt", ".pyc"))
                      or any(part in {".venv", "outputs", "runs"} for part in Path(name).parts)]
        if prohibited:
            raise RuntimeError(f"Archive includes generated/private runtime material: {prohibited}")
        checked.append({"file": archive.name, "members": len(members)})
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", help="Also verify wheel and source distributions in this directory")
    args = parser.parse_args(argv)
    result = check_sources()
    if args.archives:
        result["archives"] = check_archives(args.archives)
    result["status"] = "PASS"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
