"""Null controls for cross-model convergence claims.

Reviewers will ask whether an alignment number is just the central limit
theorem: operator entries are averages over kernel evaluations, so any two
estimators of the same population quantity approach each other as N grows.
These two nulls answer that.

- ``permutation_null`` shuffles one model's row indices. Marginal geometry is
  preserved exactly; only item correspondence is destroyed. Alignment above
  this null is correspondence-driven.
- ``zoo_permutation_null`` does the same for a whole model zoo, and returns the
  null of the *aggregate* statistic (mean of the strict upper triangle) so that
  a multi-model score is compared against a multi-model null.
- ``split_half_spectral_null`` compares two disjoint halves of ONE model. Every
  bit of the resulting distance is finite-sample noise, so it is the CLT floor
  a spectral convergence claim must clear.

Permutation equivariance
------------------------
Both permutation nulls avoid rebuilding operators. For a permutation matrix Pi,
the diffusion operator of row-permuted activations is exactly ``Pi op Pi.T``
(every step of the construction — pairwise distances, the adaptive per-row
bandwidth, the degree normalisation — is equivariant), so:

- the permuted operator is ``op[np.ix_(perm, perm)]``;
- its top eigenvectors span ``eigvecs[perm]``, and principal angles depend only
  on that column space;
- kNN neighbour lists relabel as ``inv[idx[perm]]``.

That turns an O(N^3) eigendecomposition per permutation into an O(N^2) gather
or an ``n_components x n_components`` SVD. ``test_fast_null_matches_naive_path``
pins the fast path to the naive ``alignment_matrix``-rebuilding path.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from manylatents.metrics.diffop_alignment import (
    _ensure_2d,
    build_operator,
    principal_angle_cosines,
    top_eigvecs,
)


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


# ---------------------------------------------------------------------------
# Permutation-equivariant fast path
# ---------------------------------------------------------------------------

#: Measures whose per-permutation statistic is computed from cached, permuted
#: quantities rather than by rebuilding the operator / kNN graph from scratch.
#: A measure listed here MUST be covered by ``test_fast_null_matches_naive_path``.
FAST_MEASURES = frozenset({"mutual_knn", "diffop_frobenius", "diffop_angles"})


class _PermutationCache:
    """Per-model quantities that a row permutation only relabels, never changes.

    Built once per null; each permutation then reuses them. Which quantities are
    cached depends on the measure — see the module docstring for why each one is
    permutation-equivariant.
    """

    def __init__(
        self,
        model_acts: Dict[str, np.ndarray],
        measure: str,
        k: int = 10,
        n_components: int = 10,
        knn: int = 35,
    ) -> None:
        from manylatents.metrics.platonic_convergence import MEASURES

        if measure not in MEASURES:
            raise ValueError(f"Unknown measure: {measure}. Expected one of {MEASURES}")

        self.measure = measure
        self.k = k
        self.n_components = n_components
        self.names = list(model_acts.keys())
        acts = {name: _ensure_2d(np.asarray(model_acts[name])) for name in self.names}

        n_samples = acts[self.names[0]].shape[0]
        for name in self.names:
            if acts[name].shape[0] != n_samples:
                raise ValueError(f"Sample count mismatch: {name}")
        self.n_samples = n_samples

        self.knn_idx: Dict[str, np.ndarray] = {}
        self.ops: Dict[str, np.ndarray] = {}
        self.fro: Dict[str, float] = {}
        self.eigvecs: Dict[str, np.ndarray] = {}

        if measure == "mutual_knn":
            from manylatents.utils.metrics import compute_knn

            if k >= n_samples:
                raise ValueError(f"k={k} must be < n_samples={n_samples}")
            for name in self.names:
                _, idx = compute_knn(
                    np.ascontiguousarray(acts[name], dtype=np.float32),
                    k=k, include_self=False,
                )
                self.knn_idx[name] = np.asarray(idx)
        else:
            for name in self.names:
                op = build_operator(acts[name], knn=knn)
                self.ops[name] = op
                if measure == "diffop_frobenius":
                    self.fro[name] = float(np.linalg.norm(op, "fro"))
                else:
                    self.eigvecs[name] = top_eigvecs(op, n_components)

    # -- per-permutation views ------------------------------------------------

    def view(self, name: str, perm: Optional[np.ndarray]):
        """The cached quantity for ``name`` as seen after row permutation ``perm``.

        ``perm is None`` means the identity (the anchor model).
        """
        if self.measure == "mutual_knn":
            idx = self.knn_idx[name]
            if perm is None:
                return [set(row) for row in idx]
            inv = np.empty(self.n_samples, dtype=np.int64)
            inv[perm] = np.arange(self.n_samples)
            relabelled = inv[idx[perm]]
            return [set(row) for row in relabelled]
        if self.measure == "diffop_frobenius":
            op = self.ops[name]
            return op if perm is None else op[np.ix_(perm, perm)]
        vecs = self.eigvecs[name]
        return vecs if perm is None else vecs[perm]

    def pair_stat(self, view_i, view_j, name_i: str, name_j: str) -> float:
        """The pairwise statistic between two permuted views."""
        if self.measure == "mutual_knn":
            k = float(self.k)
            total = 0.0
            for sa, sb in zip(view_i, view_j):
                total += len(sa & sb)
            return total / (k * self.n_samples)
        if self.measure == "diffop_frobenius":
            den = 0.5 * (self.fro[name_i] + self.fro[name_j])
            if den <= 0:
                return 0.0
            return float(np.linalg.norm(view_i - view_j, "fro")) / den
        cosines = principal_angle_cosines(view_i, view_j)
        return float(cosines.mean())


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
    """Null distribution of the PAIRWISE statistic under shuffled correspondence.

    Only ``acts_b`` is permuted; ``acts_a`` is the anchor. Use
    ``zoo_permutation_null`` when the observed statistic aggregates more than
    one pair — comparing a multi-pair mean to this one-pair null is not a test.

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
    a = np.asarray(acts_a)
    b = np.asarray(acts_b)
    if _ensure_2d(a).shape[0] != _ensure_2d(b).shape[0]:
        raise ValueError(
            f"Sample count mismatch: {_ensure_2d(a).shape[0]} vs {_ensure_2d(b).shape[0]}"
        )

    if measure not in FAST_MEASURES:
        return _naive_permutation_null(
            a, b, measure=measure, n_perm=n_perm, seed=seed,
            k=k, n_components=n_components, knn=knn,
        )

    rng = np.random.default_rng(seed)
    cache = _PermutationCache(
        {"a": a, "b": b}, measure=measure, k=k, n_components=n_components, knn=knn
    )
    view_a = cache.view("a", None)

    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(cache.n_samples)
        out[i] = cache.pair_stat(view_a, cache.view("b", perm), "a", "b")
    return out


def zoo_permutation_null(
    model_acts: Dict[str, np.ndarray],
    measure: str = "mutual_knn",
    n_perm: int = 200,
    seed: int = 0,
    k: int = 10,
    n_components: int = 10,
    knn: int = 35,
) -> np.ndarray:
    """Null distribution of the AGGREGATE zoo statistic under shuffled correspondence.

    The observed zoo score is the mean of the strict upper triangle of the
    alignment matrix over all M(M-1)/2 pairs. This null is the same statistic:
    each iteration independently row-permutes every model except the first
    (the anchor), rebuilds the full alignment matrix, and takes the mean of its
    strict upper triangle. Under it no model shares item correspondence with any
    other — the intended null hypothesis — while every model's marginal geometry
    is preserved exactly.

    Args:
        model_acts: Model name -> (N, D) index-aligned activations. At least 2.
        measure: One of "mutual_knn", "diffop_frobenius", "diffop_angles".
        n_perm: Number of permutations.
        seed: RNG seed; the null is fully determined by it.
        k: Neighbourhood size for "mutual_knn".
        n_components: Eigen-subspace size for "diffop_angles".
        knn: Adaptive-bandwidth neighbour count for operator construction.

    Returns:
        (n_perm,) array of mean-of-upper-triangle statistics.
    """
    names = list(model_acts.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 models for convergence measurement")

    if measure not in FAST_MEASURES:
        return _naive_zoo_permutation_null(
            model_acts, measure=measure, n_perm=n_perm, seed=seed,
            k=k, n_components=n_components, knn=knn,
        )

    rng = np.random.default_rng(seed)
    cache = _PermutationCache(
        model_acts, measure=measure, k=k, n_components=n_components, knn=knn
    )
    m = len(names)

    out = np.empty(n_perm, dtype=float)
    for it in range(n_perm):
        # Anchor (names[0]) keeps its row order; every other model is permuted
        # independently, so no two models share item correspondence.
        views = [cache.view(names[0], None)]
        for name in names[1:]:
            views.append(cache.view(name, rng.permutation(cache.n_samples)))

        total = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                total += cache.pair_stat(views[i], views[j], names[i], names[j])
        out[it] = total / (m * (m - 1) / 2.0)
    return out


# ---------------------------------------------------------------------------
# Naive reference implementations
#
# Kept as the equivalence reference for the fast path (and as the fallback for
# any measure removed from FAST_MEASURES). They rebuild every operator / kNN
# graph from raw activations on every permutation.
# ---------------------------------------------------------------------------

def _naive_permutation_null(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    measure: str = "mutual_knn",
    n_perm: int = 200,
    seed: int = 0,
    k: int = 10,
    n_components: int = 10,
    knn: int = 35,
) -> np.ndarray:
    """``permutation_null`` by full rebuild. Same RNG stream as the fast path."""
    from manylatents.metrics.platonic_convergence import alignment_matrix

    rng = np.random.default_rng(seed)
    a = _ensure_2d(np.asarray(acts_a))
    b = _ensure_2d(np.asarray(acts_b))
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


def _naive_zoo_permutation_null(
    model_acts: Dict[str, np.ndarray],
    measure: str = "mutual_knn",
    n_perm: int = 200,
    seed: int = 0,
    k: int = 10,
    n_components: int = 10,
    knn: int = 35,
) -> np.ndarray:
    """``zoo_permutation_null`` by full rebuild. Same RNG stream as the fast path."""
    from manylatents.metrics.platonic_convergence import alignment_matrix

    rng = np.random.default_rng(seed)
    names = list(model_acts.keys())
    acts = {name: _ensure_2d(np.asarray(model_acts[name])) for name in names}
    n_samples = acts[names[0]].shape[0]

    out = np.empty(n_perm, dtype=float)
    for it in range(n_perm):
        shuffled = {names[0]: acts[names[0]]}
        for name in names[1:]:
            shuffled[name] = acts[name][rng.permutation(n_samples)]
        _, mat = alignment_matrix(
            shuffled, measure=measure, k=k, n_components=n_components, knn=knn,
        )
        iu = np.triu_indices(len(names), k=1)
        out[it] = float(mat[iu].mean())
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
