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
    symmetrize_operator,
)
from manylatents.metrics.mutual_knn import mutual_knn_pairwise
from manylatents.metrics.registry import register_metric

MEASURES: Tuple[str, ...] = ("mutual_knn", "diffop_frobenius", "diffop_angles")

#: Measures computable from a diffusion operator alone. "mutual_knn" is absent
#: because it needs the point cloud, which an operator does not carry.
OPERATOR_MEASURES: Tuple[str, ...] = ("diffop_frobenius", "diffop_angles")

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


def prepare_operator_zoo(model_ops: Dict[str, np.ndarray]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Validate and symmetrize a name -> (N, N) operator mapping.

    Shared by ``alignment_matrix_from_operators`` and the operator-input null so
    both see byte-identical inputs.

    Raises:
        ValueError: Fewer than 2 models, a non-square operator, or operators
            built over different numbers of probe items (which would mean the
            rows are not the same probe, so no comparison is meaningful).
    """
    names = list(model_ops.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 models for convergence measurement")

    ops = {name: symmetrize_operator(model_ops[name]) for name in names}
    n_samples = ops[names[0]].shape[0]
    for name in names:
        if ops[name].shape[0] != n_samples:
            raise ValueError(
                f"Operator shape mismatch: {name} is {ops[name].shape}, "
                f"expected ({n_samples}, {n_samples})"
            )
    return names, ops


def alignment_matrix_from_operators(
    model_ops: Dict[str, np.ndarray],
    measure: str = "diffop_frobenius",
    n_components: int = 10,
) -> Tuple[List[str], np.ndarray]:
    """``alignment_matrix`` for zoos whose diffusion operators are already built.

    Same measures, same diagonal convention, same pairwise functions as the
    activation path — it only skips ``build_operator``. That makes it usable on
    operators cached to disk, where the activations are long gone.

    Operators are symmetrized on entry (see ``symmetrize_operator``); for an
    operator that ``build_operator`` produced this is exactly a no-op, so this
    function reproduces ``alignment_matrix`` bit for bit.

    Args:
        model_ops: Model name -> (N, N) diffusion operator. Row i must be the
            same probe item in every model.
        measure: One of OPERATOR_MEASURES.
        n_components: Eigen-subspace size for "diffop_angles".

    Returns:
        (model_names, matrix) with matrix symmetric of shape (M, M).

    Raises:
        ValueError: If ``measure`` is "mutual_knn" (needs raw activations) or
            otherwise unknown.
    """
    if measure == "mutual_knn":
        raise ValueError(
            "measure='mutual_knn' needs raw activations: it counts shared "
            "neighbours in the point cloud, which a diffusion operator does not "
            f"carry. From cached operators use one of {OPERATOR_MEASURES}."
        )
    if measure not in OPERATOR_MEASURES:
        raise ValueError(f"Unknown measure: {measure}. Expected one of {OPERATOR_MEASURES}")

    names, ops = prepare_operator_zoo(model_ops)

    m = len(names)
    mat = np.full((m, m), _SELF_VALUE[measure], dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            if measure == "diffop_frobenius":
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
