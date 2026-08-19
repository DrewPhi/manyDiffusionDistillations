"""Cross-seed confidence-interval primitives. Pure numpy; no torch, no I/O.

Reusable component: consumed by the cross-seed aggregator, the reproduce
verifier, and figure generation. Never invents a CI for n<2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

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
        return f"{m} (n={iv.n})"
    if abs(iv.mean) < sci_below:
        return f"{iv.mean:.1e} [{iv.lo:.1e}, {iv.hi:.1e}]"
    return f"{iv.mean:.2f} [{iv.lo:.2f}, {iv.hi:.2f}]"


@dataclass(frozen=True)
class PairInterval:
    """A mean over pairs, with both an optimistic and an honest interval.

    ``pair_ci`` resamples the pairs themselves. ``model_ci`` resamples the models
    the pairs are built from. They differ because the pairs are not independent:
    a zoo of m models yields m(m-1)/2 pairs, and each model appears in m-1 of
    them, so a pair bootstrap treats one model's idiosyncrasy as m-1 independent
    observations. Report ``model_ci``; ``pair_ci`` is kept beside it so the gap
    is visible rather than a matter of trust.
    """
    mean: float
    pair_ci: Interval
    model_ci: Interval
    n_pairs: int
    n_models: int


def pairwise_cluster_bootstrap(
    values: Mapping[Tuple[str, str], float], n_boot: int = 10000,
    ci: float = 0.95, seed: int = 0,
) -> PairInterval:
    """CI for a mean over pairs, resampling the underlying models.

    ``values`` maps an unordered model pair to its score; key order is not
    significant. Each bootstrap round draws the model list with replacement and
    averages over the pairs that resample induces, skipping the diagonal (a
    model drawn twice contributes no self-pair, because no self-pair was
    measured). Rounds that induce no pairs at all -- possible when the same
    model is drawn every time -- are discarded rather than counted as zero.

    This is the interval to quote for a claim about a zoo. The pair bootstrap
    returned alongside it is systematically narrower and is included only so the
    difference can be seen.
    """
    lookup: dict[Tuple[str, str], float] = {}
    models: list[str] = []
    for (a, b), v in values.items():
        if not math.isfinite(float(v)):
            continue
        lookup[(a, b)] = float(v)
        lookup[(b, a)] = float(v)
        for m in (a, b):
            if m not in models:
                models.append(m)
    models.sort()

    flat = [lookup[(a, b)] for (a, b) in values if (a, b) in lookup]
    pair_ci = bootstrap_ci(flat, n_boot=n_boot, ci=ci, seed=seed)

    m = len(models)
    if m < 2 or not flat:
        return PairInterval(pair_ci.mean, pair_ci, pair_ci, len(flat), m)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        draw = [models[i] for i in rng.integers(0, m, size=m)]
        acc, cnt = 0.0, 0
        for i in range(m):
            for j in range(i + 1, m):
                v = lookup.get((draw[i], draw[j]))
                if v is not None:          # None only for a self-pair
                    acc += v
                    cnt += 1
        if cnt:
            boot.append(acc / cnt)
    if not boot:
        return PairInterval(pair_ci.mean, pair_ci, pair_ci, len(flat), m)

    alpha = (1.0 - ci) / 2.0
    arr = np.asarray(boot, dtype=float)
    model_ci = Interval(pair_ci.mean, float(np.quantile(arr, alpha)),
                        float(np.quantile(arr, 1.0 - alpha)), m)
    return PairInterval(pair_ci.mean, pair_ci, model_ci, len(flat), m)


def interval_payload(iv: "Interval | PairInterval") -> dict:
    """JSON-serializable form, so every beat writes the same interval schema."""
    if isinstance(iv, PairInterval):
        return {
            "mean": iv.mean,
            "ci95": [iv.model_ci.lo, iv.model_ci.hi],
            "ci95_pair_bootstrap": [iv.pair_ci.lo, iv.pair_ci.hi],
            "n_pairs": iv.n_pairs,
            "n_models": iv.n_models,
            "resampled": "models",
        }
    return {"mean": iv.mean, "ci95": [iv.lo, iv.hi], "n": iv.n,
            "resampled": "observations"}
