"""Aggregate a metric across per-seed artifact JSONs into an Interval.

Reusable component. Each input path is one seed's diagnostic JSON with the
schema results[layer_scheme][cell][metric]. No torch, no cluster access.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from manylatents.stats.intervals import Interval, bootstrap_ci


def dig(obj: dict, path: str) -> Any:
    cur: Any = obj
    for k in path.strip("/").split("/"):
        cur = cur[k]
    return cur


def aggregate_paths(paths: Sequence[Path | str], metric_path: str,
                    n_boot: int = 10000, seed: int = 0) -> Interval:
    values = []
    for p in paths:
        obj = json.loads(Path(p).read_text())
        values.append(float(dig(obj, metric_path)))
    return bootstrap_ci(values, n_boot=n_boot, seed=seed)


def aggregate_cell(paths: Sequence[Path | str], layer_scheme: str, cell: str,
                   metric: str, n_boot: int = 10000, seed: int = 0) -> Interval:
    return aggregate_paths(paths, f"results/{layer_scheme}/{cell}/{metric}",
                           n_boot=n_boot, seed=seed)
