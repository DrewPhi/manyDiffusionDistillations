"""Cross-seed confidence-interval primitives. Pure numpy; no torch, no I/O.

Reusable component: consumed by the cross-seed aggregator, the reproduce
verifier, and figure generation. Never invents a CI for n<2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float
    n: int


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 0) -> Interval:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    mean = float(arr.mean())
    if n == 1:
        return Interval(mean, mean, mean, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1.0 - alpha))
    return Interval(mean, lo, hi, n)


def mean_se(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, se


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                          n_boot: int = 10000, ci: float = 0.95,
                          seed: int = 0) -> Interval:
    aa = np.asarray(list(a), dtype=float)
    bb = np.asarray(list(b), dtype=float)
    if aa.shape != bb.shape:
        raise ValueError(f"paired inputs must match: {aa.shape} vs {bb.shape}")
    return bootstrap_ci((aa - bb).tolist(), n_boot=n_boot, ci=ci, seed=seed)


def format_interval(iv: Interval, sci_below: float = 1e-3) -> str:
    if iv.n <= 1:
        m = f"{iv.mean:.1e}" if abs(iv.mean) < sci_below else f"{iv.mean:.2f}"
        return f"{m} (n=1)"
    if abs(iv.mean) < sci_below:
        return f"{iv.mean:.1e} [{iv.lo:.1e}, {iv.hi:.1e}]"
    return f"{iv.mean:.2f} [{iv.lo:.2f}, {iv.hi:.2f}]"
