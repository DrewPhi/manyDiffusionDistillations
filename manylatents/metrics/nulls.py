"""Null controls for cross-model convergence claims.

Reviewers will ask whether an alignment number is just the central limit
theorem: operator entries are averages over kernel evaluations, so any two
estimators of the same population quantity approach each other as N grows.
These two nulls answer that.

- ``permutation_null`` shuffles one model's row indices. Marginal geometry is
  preserved exactly; only item correspondence is destroyed. Alignment above
  this null is correspondence-driven.
- ``split_half_spectral_null`` compares two disjoint halves of ONE model. Every
  bit of the resulting distance is finite-sample noise, so it is the CLT floor
  a spectral convergence claim must clear.
"""

from typing import List, Sequence, Tuple, Union

import numpy as np

from manylatents.metrics.diffop_alignment import build_operator


def spectral_distance(p: np.ndarray, q: np.ndarray, n_components: int = 10) -> float:
    """Relative distance between the top-|n| eigenvalue spectra of two operators.

    Comparable across operators built on *different* point sets, which entrywise
    Frobenius distance is not — that is what makes the split-half null possible.
    """
    n_components = min(n_components, p.shape[0], q.shape[0])
    ev_p = np.sort(np.abs(np.linalg.eigvalsh(0.5 * (p + p.T))))[::-1][:n_components]
    ev_q = np.sort(np.abs(np.linalg.eigvalsh(0.5 * (q + q.T))))[::-1][:n_components]
    num = float(np.linalg.norm(ev_p - ev_q))
    den = 0.5 * (float(np.linalg.norm(ev_p)) + float(np.linalg.norm(ev_q)))
    return num / den if den > 0 else 0.0


def permutation_null(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    measure: str = "mutual_knn",
    n_perm: int = 200,
    seed: int = 0,
    k: int = 10,
    n_components: int = 10,
    knn: int = 35,
) -> np.ndarray:
    """Null distribution of the pairwise statistic under shuffled correspondence.

    Args:
        acts_a: (N, D1) activations, index-aligned with acts_b.
        acts_b: (N, D2) activations.
        measure: One of "mutual_knn", "diffop_frobenius", "diffop_angles".
        n_perm: Number of permutations.
        seed: RNG seed; the null is fully determined by it.
        k: Neighbourhood size for "mutual_knn".
        n_components: Eigen-subspace size for "diffop_angles".
        knn: Adaptive-bandwidth neighbour count for operator construction.

    Returns:
        (n_perm,) array of statistics computed on row-permuted acts_b.
    """
    from manylatents.metrics.platonic_convergence import alignment_matrix

    rng = np.random.default_rng(seed)
    a = np.asarray(acts_a)
    b = np.asarray(acts_b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Sample count mismatch: {a.shape[0]} vs {b.shape[0]}")

    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(b.shape[0])
        _, mat = alignment_matrix(
            {"a": a, "b_shuffled": b[perm]},
            measure=measure, k=k, n_components=n_components, knn=knn,
        )
        out[i] = float(mat[0, 1])
    return out


def empirical_p(observed: float, null_samples: Sequence[float], higher_is_better: bool) -> float:
    """One-sided empirical p-value with the standard +1 correction.

    Ties count as "at least as extreme", so the p-value is never zero and never
    understates the null.
    """
    null = np.asarray(list(null_samples), dtype=float)
    if higher_is_better:
        at_least_as_extreme = int(np.sum(null >= observed))
    else:
        at_least_as_extreme = int(np.sum(null <= observed))
    return float((at_least_as_extreme + 1) / (null.size + 1))


def split_half_spectral_null(
    acts: np.ndarray,
    n_splits: int = 20,
    n_components: int = 10,
    knn: int = 35,
    seed: int = 0,
    _return_index_pairs: bool = False,
) -> Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """CLT floor: spectral distance between two disjoint halves of ONE model.

    Both halves are drawn from the same model, so the only mechanism producing a
    nonzero distance is finite-sample noise. A cross-model spectral distance
    must fall below this floor to count as convergence.

    Args:
        acts: (N, D) activations from a single model.
        n_splits: Number of independent random half-splits.
        n_components: Number of leading eigenvalues compared.
        knn: Adaptive-bandwidth neighbour count. Must be < N/2 — enforced by
            raising ``ValueError``, not silently clamped: a wider bandwidth
            over-smooths each half and biases the CLT floor downward, which is
            the one direction this control must never move in silently.
        seed: RNG seed.
        _return_index_pairs: Test hook — return the (idx_a, idx_b) pairs instead
            of the distances, so disjointness can be asserted.

    Returns:
        (n_splits,) array of spectral distances, or the index pairs.

    Raises:
        ValueError: If ``knn >= half`` (``half = N // 2``).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(acts)
    n = x.shape[0]
    half = n // 2
    if knn >= half:
        raise ValueError(
            f"knn={knn} must be < half={half} (n={n}); a wider bandwidth "
            "biases the CLT floor downward"
        )

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    dists = np.empty(n_splits, dtype=float)
    for i in range(n_splits):
        perm = rng.permutation(n)
        idx_a, idx_b = perm[:half], perm[half : 2 * half]
        pairs.append((idx_a, idx_b))
        if _return_index_pairs:
            continue
        op_a = build_operator(x[idx_a], knn=knn)
        op_b = build_operator(x[idx_b], knn=knn)
        dists[i] = spectral_distance(op_a, op_b, n_components=n_components)

    if _return_index_pairs:
        return pairs
    return dists
