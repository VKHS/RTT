"""Shared third-party imports with a statistics-only fallback when torch is absent."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import dataclasses
import datetime as _dt
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import pickle
import platform
import random
import shutil
import socket
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is required for every GPU stage; reduce/statistics run without it.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on hosts without torch
    torch = None  # type: ignore

    class _NNStub:
        """Lets the module import without torch (version/identity/statistics modes)."""
        Module = object

        def __getattr__(self, name):
            raise RuntimeError("torch is required for this mode")
    nn = _NNStub()  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False
