"""Operation-count estimates for the native incumbent."""
from __future__ import annotations


def incumbent_flops_per_pass(n: int, nnz: int, d: int, F_in: int, enc_h: int, C: int, T: int) -> float:
    enc = 2.0 * n * (F_in * enc_h + enc_h * d)
    step = 2.0 * nnz * d + 2.0 * n * d * d + 6.0 * n * d
    head = 2.0 * n * d * C
    return enc + T * step + head

