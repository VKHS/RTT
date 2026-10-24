#!/usr/bin/env python3
"""Compatibility entry point for the modular native-arxiv RTT implementation.

The computational implementation lives in ``src/rtt/core``. Dataset-specific
entry points and experiment configurations are provided in ``experiments/``.
"""
from pathlib import Path
import sys

_SOURCE = str(Path(__file__).resolve().parent / "src")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from rtt.core import api as _api

_api.install_compatibility_facade(__name__)

if __name__ == "__main__":
    raise SystemExit(_api.main())
