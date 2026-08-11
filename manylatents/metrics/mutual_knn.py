"""True mutual-kNN neighbourhood overlap (the PRH alignment baseline).

CrossModalJaccard computes |A∩B|/|A∪B|. The Platonic Representation
Hypothesis literature uses |A∩B|/k. The two rank model pairs differently,
so reproducing the PRH baseline requires this second form.
"""

from typing import Dict, Optional, Union

import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Squeeze a singleton middle axis, matching cross_modal_jaccard's convention."""
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr.squeeze(1)
    return arr


def mutual_knn_pairwise(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Per-sample mutual-kNN overlap |A∩B|/k between two index-aligned spaces.

    Args:
        embeddings_a: (N, D1) or (N, 1, D1).
        embeddings_b: (N, D2) or (N, 1, D2). Row i must be the same probe item
            as row i of ``embeddings_a``.
        k: Neighbourhood size, excluding self. Must be < N.

    Returns:
        (N,) array of overlaps in [0, 1].
    """
    a = _ensure_2d(np.asarray(embeddings_a))
    b = _ensure_2d(np.asarray(embeddings_b))
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Sample count mismatch: {a.shape[0]} vs {b.shape[0]}")
    n = a.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be < n_samples={n}")

    _, idx_a = compute_knn(np.ascontiguousarray(a, dtype=np.float32), k=k, include_self=False)
    _, idx_b = compute_knn(np.ascontiguousarray(b, dtype=np.float32), k=k, include_self=False)

    overlaps = np.empty(n, dtype=float)
    for i in range(n):
        overlaps[i] = len(set(idx_a[i]) & set(idx_b[i])) / float(k)
    return overlaps


@register_metric(
    aliases=["mutual_knn", "prh_alignment"],
    default_params={"k": 10},
    description="Mutual-kNN neighbourhood overlap |A∩B|/k (PRH baseline)",
)
def MutualKNN(
    embeddings: Union[np.ndarray, Dict[str, np.ndarray]],
    dataset=None,
    module=None,
    k: int = 10,
    return_pairwise: bool = False,
    cache: Optional[dict] = None,
) -> Union[float, Dict[str, np.ndarray]]:
    """Mean mutual-kNN overlap across all model pairs.

    Args:
        embeddings: A single array (self-comparison, trivially 1.0) or a dict
            mapping model name to an index-aligned (N, D) activation array.
        dataset: Unused; present for the metric protocol.
        module: Unused; present for the metric protocol.
        k: Neighbourhood size.
        return_pairwise: If True, return {pair_name: (N,) overlaps} instead of
            the scalar mean.

    Returns:
        Scalar mean overlap in [0, 1], or the pairwise dict.

    Higher is better: 1.0 means identical neighbourhood structure.
    """
    if isinstance(embeddings, np.ndarray):
        return 1.0

    names = list(embeddings.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 models for mutual-kNN comparison")

    n_samples = _ensure_2d(np.asarray(embeddings[names[0]])).shape[0]
    for name, emb in embeddings.items():
        if _ensure_2d(np.asarray(emb)).shape[0] != n_samples:
            raise ValueError(f"Sample count mismatch: {name}")

    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise[f"{names[i]}_{names[j]}"] = mutual_knn_pairwise(
                embeddings[names[i]], embeddings[names[j]], k=k
            )

    if return_pairwise:
        return pairwise
    return float(np.stack(list(pairwise.values()), axis=0).mean())
