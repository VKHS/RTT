"""Classified errors and process exit codes."""
from __future__ import annotations

from ._imports import (
    Optional,
)

# ----------------------------------------------------------------------------
# Exit codes shared with the Slurm orchestrator
# ----------------------------------------------------------------------------
EXIT_OK = 0
EXIT_SCIENTIFIC = 1
EXIT_USAGE = 2
EXIT_PREREQUISITE = 33
EXIT_ORCHESTRATION = 70
EXIT_ARTIFACT_MISSING = 74
EXIT_INFRASTRUCTURE = 75

INFRA_PATTERNS = (
    "cudaErrorDevicesUnavailable", "CUDA-capable device", "uncorrectable ECC error",
    "CUDA driver initialization failed", "CUDA initialization error",
    "NVIDIA driver is shutting down", "GPU has fallen off the bus", "device is lost",
    "system not yet initialized", "failed call to cuInit", "driver/library version mismatch",
    "CUDA error: an illegal memory access", "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDA error: unspecified launch failure",
)


class RTTFailure(Exception):
    """A classified failure.  failure_class in {SCIENTIFIC, PREREQUISITE,
    INFRASTRUCTURE, ARTIFACT_MISSING, ORCHESTRATION}."""

    def __init__(self, failure_class: str, message: str, retriable: bool = False,
                 stage: str = "", details: Optional[dict] = None):
        super().__init__(message)
        self.failure_class = failure_class
        self.retriable = retriable
        self.stage = stage
        self.details = details or {}


def exit_code_for(failure_class: str) -> int:
    return {
        "SCIENTIFIC": EXIT_SCIENTIFIC,
        "PREREQUISITE": EXIT_PREREQUISITE,
        "INFRASTRUCTURE": EXIT_INFRASTRUCTURE,
        "ARTIFACT_MISSING": EXIT_ARTIFACT_MISSING,
        "ORCHESTRATION": EXIT_ORCHESTRATION,
    }.get(str(failure_class).upper(), EXIT_SCIENTIFIC)


