"""Compatibility imports for continuous probability-floor optimization."""
from pathlib import Path
import sys

_SOURCE = str(Path(__file__).resolve().parent / "src")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from rtt.frontier import continuous_floor_frontier, project_probability_floor

__all__ = ["continuous_floor_frontier", "project_probability_floor"]
