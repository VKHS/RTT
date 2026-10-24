#!/usr/bin/env python3
"""Dataset-local entry point; shared implementation lives in rtt.cli."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rtt.cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_config=Path(__file__).with_name("config.json")))
