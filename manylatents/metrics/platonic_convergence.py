"""Cross-model convergence: a model x model alignment matrix plus a scalar score.

`compute_multi_model_spread` collapses everything to one number per timestep,
so it cannot say which model pairs converge. This returns the full matrix and
derives the scalar from it.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from manylatents.metrics.diffop_alignment import (
    build_operator,
    diffop_frobenius_distance,
    diffop_subspace_alignment,
)
from manylatents.metrics.mutual_knn import mutual_knn_pairwise
from manylatents.metrics.registry import register_metric

MEASURES: Tuple[str, ...] = ("mutual_knn", "diffop_frobenius", "diffop_angles")

# Value on the diagonal (a model compared with itself) per measure.
_SELF_VALUE = {"mutual_knn": 1.0, "diffop_frobenius": 0.0, "diffop_angles": 1.0}

HIGHER_IS_BETTER = {"mutual_knn": True, "diffop_frobenius": False, "diffop_angles": True}


def alignment_matrix(
    model_acts: Dict[str, np.ndarray],
    measure: str = "mutual_knn",
    k: int = 10,
    n_components: int = 10,
    knn: int = 35,
) -> Tuple[List[str], np.ndarray]:
    """Pairwise alignment between every pair of index-aligned model snapshots.

    Args:
        model_acts: Model name -> (N, D) activations. Row i must be the same
            probe item in every model.
        measure: One of MEASURES.
        k: Neighbourhood size for "mutual_knn".
        n_components: Eigen-subspace size for "diffop_angles".
        knn: Adaptive-bandwidth neighbour count for operator construction.

    Returns:
        (model_names, matrix) with matrix symmetric of shape (M, M).
    """
    if measure not in MEASURES:
        raise ValueError(f"Unknown measure: {measure}. Expected one of {MEASURES}")

    names = list(model_acts.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 models for convergence measurement")

    n_samples = np.asarray(model_acts[names[0]]).shape[0]
    for name in names:
        if np.asarray(model_acts[name]).shape[0] != n_samples:
            raise ValueError(f"Sample count mismatch: {name}")

    ops = None
    if measure in ("diffop_frobenius", "diffop_angles"):
        ops = {name: build_operator(model_acts[name], knn=knn) for name in names}

    m = len(names)
    mat = np.full((m, m), _SELF_VALUE[measure], dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            if measure == "mutual_knn":
                val = float(mutual_knn_pairwise(model_acts[names[i]], model_acts[names[j]], k=k).mean())
            elif measure == "diffop_frobenius":
                val = diffop_frobenius_distance(ops[names[i]], ops[names[j]])
            else:
                val = diffop_subspace_alignment(
                    ops[names[i]], ops[names[j]], n_components=n_components
                )
            mat[i, j] = val
            mat[j, i] = val
    return names, mat


@register_metric(
    aliases=["platonic_convergence"],
    default_params={"measure": "mutual_knn", "k": 10},
    description="Cross-model convergence matrix + scalar score (PRH-style)",
)
def PlatonicConvergence(
    embeddings: Dict[str, np.ndarray],
    dataset=None,
    module=None,
    cache: Optional[dict] = None,
    measure: str = "mutual_knn",
    k: int = 10,
    n_components: int = 10,
    knn: int = 35,
) -> Dict[str, Any]:
    """Convergence across a model zoo.

    Args:
        embeddings: Model name -> index-aligned (N, D) activations.
        dataset: Unused; present for the metric protocol.
        module: Unused; present for the metric protocol.
        cache: Unused; present for the metric protocol.
        measure: One of MEASURES.
        k: Neighbourhood size for "mutual_knn".
        n_components: Eigen-subspace size for "diffop_angles".
        knn: Adaptive-bandwidth neighbour count for operator construction.

    Returns:
        {"models": [...], "matrix": nested list, "score": float,
         "measure": str, "higher_is_better": bool}

        ``score`` is the mean of the strict upper triangle — the average
        cross-model alignment, with self-comparisons excluded.
    """
    names, mat = alignment_matrix(
        embeddings, measure=measure, k=k, n_components=n_components, knn=knn
    )
    iu = np.triu_indices(len(names), k=1)
    return {
        "models": names,
        "matrix": mat.tolist(),
        "score": float(mat[iu].mean()),
        "measure": measure,
        "higher_is_better": HIGHER_IS_BETTER[measure],
    }
