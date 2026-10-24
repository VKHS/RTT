#!/usr/bin/env python3
"""Expose the native arxiv CLI while preserving its arguments and exit status."""
from pathlib import Path
import os
import sys


if __name__ == "__main__":
    runner = Path(__file__).resolve().parents[2] / "run_rtt_iclr2027_arrhenius.py"
    os.execv(sys.executable, [sys.executable, str(runner), *sys.argv[1:]])
