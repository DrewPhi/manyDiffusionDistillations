"""Null controls that separate real alignment from the CLT floor."""
import numpy as np
import pytest

from manylatents.metrics.nulls import (
    empirical_p,
    permutation_null,
    spectral_distance,
    split_half_spectral_null,
)


def test_permutation_null_length_and_range():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 4)).astype(np.float32)
    b = rng.normal(size=(40, 4)).astype(np.float32)
    null = permutation_null(a, b, measure="mutual_knn", n_perm=25, seed=0, k=5)
    assert null.shape == (25,)
    assert np.all((null >= 0.0) & (null <= 1.0))


def test_permutation_null_is_deterministic_given_seed():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(30, 3)).astype(np.float32)
    b = rng.normal(size=(30, 3)).astype(np.float32)
    kw = dict(measure="mutual_knn", n_perm=10, seed=7, k=4)
    np.testing.assert_allclose(permutation_null(a, b, **kw), permutation_null(a, b, **kw))


def test_identical_models_beat_their_permutation_null():
    """The signal case: perfect correspondence must be significant."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(60, 5)).astype(np.float32)
    null = permutation_null(a, a.copy(), measure="mutual_knn", n_perm=50, seed=0, k=5)
    p = empirical_p(observed=1.0, null_samples=null, higher_is_better=True)
    assert p < 0.05


def test_independent_models_do_not_beat_their_null():
    """The CLT case: unrelated models must NOT look significant."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(60, 5)).astype(np.float32)
    b = rng.normal(size=(60, 5)).astype(np.float32)
    from manylatents.metrics.mutual_knn import mutual_knn_pairwise

    observed = float(mutual_knn_pairwise(a, b, k=5).mean())
    null = permutation_null(a, b, measure="mutual_knn", n_perm=100, seed=0, k=5)
    assert empirical_p(observed, null, higher_is_better=True) > 0.05


def test_empirical_p_counts_ties_conservatively():
    null = np.array([0.1, 0.2, 0.3, 0.4])
    # observed equals a null draw -> counted as "at least as extreme"
    assert empirical_p(0.4, null, higher_is_better=True) == pytest.approx((1 + 1) / (4 + 1))
    assert empirical_p(0.9, null, higher_is_better=True) == pytest.approx(1 / 5)
    assert empirical_p(0.05, null, higher_is_better=False) == pytest.approx(1 / 5)


def test_spectral_distance_zero_for_identical():
    rng = np.random.default_rng(4)
    from manylatents.metrics.diffop_alignment import build_operator

    p = build_operator(rng.normal(size=(30, 4)).astype(np.float32), knn=5)
    assert spectral_distance(p, p.copy(), n_components=3) == pytest.approx(0.0, abs=1e-12)


def test_split_half_null_is_nonzero_and_shrinks_with_n():
    """The CLT floor itself: same model, disjoint halves, distance > 0 but
    smaller for larger probe sets."""
    rng = np.random.default_rng(5)
    small = rng.normal(size=(60, 4)).astype(np.float32)
    large = rng.normal(size=(240, 4)).astype(np.float32)
    null_small = split_half_spectral_null(small, n_splits=8, n_components=3, knn=5, seed=0)
    null_large = split_half_spectral_null(large, n_splits=8, n_components=3, knn=5, seed=0)
    assert null_small.shape == (8,)
    assert float(null_small.mean()) > 0.0
    assert float(null_large.mean()) < float(null_small.mean())


def test_split_half_null_uses_disjoint_halves():
    """Guard against accidentally comparing overlapping samples, which would
    understate the floor."""
    rng = np.random.default_rng(6)
    acts = rng.normal(size=(40, 3)).astype(np.float32)
    seen = split_half_spectral_null(acts, n_splits=3, n_components=2, knn=5, seed=0,
                                    _return_index_pairs=True)
    for idx_a, idx_b in seen:
        assert len(set(idx_a) & set(idx_b)) == 0
        assert len(idx_a) + len(idx_b) <= 40
