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


def test_split_half_rejects_bandwidth_wider_than_half():
    """A knn wider than the half-split silently over-smooths and biases the
    CLT floor downward, so it must raise rather than clamp."""
    rng = np.random.default_rng(9)
    acts = rng.normal(size=(20, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="must be < half"):
        split_half_spectral_null(acts, n_splits=2, n_components=2, knn=10, seed=0)


# ---------------------------------------------------------------------------
# Permutation-equivariant fast path (must equal the naive rebuild exactly)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("measure", ["mutual_knn", "diffop_frobenius", "diffop_angles"])
def test_fast_null_matches_naive_path(measure):
    """The optimization is correctness-critical: caching operators and
    eigenvectors across permutations must reproduce the full-rebuild path."""
    from manylatents.metrics.nulls import _naive_permutation_null

    rng = np.random.default_rng(11)
    a = rng.normal(size=(50, 5)).astype(np.float32)
    b = rng.normal(size=(50, 5)).astype(np.float32)
    kw = dict(measure=measure, n_perm=12, seed=3, k=5, n_components=4, knn=8)
    fast = permutation_null(a, b, **kw)
    naive = _naive_permutation_null(a, b, **kw)
    np.testing.assert_allclose(fast, naive, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("measure", ["mutual_knn", "diffop_frobenius", "diffop_angles"])
def test_fast_zoo_null_matches_naive_path(measure):
    from manylatents.metrics.nulls import _naive_zoo_permutation_null, zoo_permutation_null

    rng = np.random.default_rng(12)
    zoo = {
        "a": rng.normal(size=(50, 5)).astype(np.float32),
        "b": rng.normal(size=(50, 5)).astype(np.float32),
        "c": rng.normal(size=(50, 5)).astype(np.float32),
    }
    kw = dict(measure=measure, n_perm=10, seed=5, k=5, n_components=4, knn=8)
    fast = zoo_permutation_null(zoo, **kw)
    naive = _naive_zoo_permutation_null(zoo, **kw)
    np.testing.assert_allclose(fast, naive, rtol=1e-9, atol=1e-9)


def test_zoo_null_reduces_to_pairwise_for_two_models():
    """With M=2 the upper triangle is one cell, so the aggregate null is the
    pairwise null — same RNG stream, same numbers."""
    from manylatents.metrics.nulls import zoo_permutation_null

    rng = np.random.default_rng(13)
    a = rng.normal(size=(40, 4)).astype(np.float32)
    b = rng.normal(size=(40, 4)).astype(np.float32)
    kw = dict(measure="mutual_knn", n_perm=15, seed=2, k=5)
    np.testing.assert_allclose(
        zoo_permutation_null({"a": a, "b": b}, **kw), permutation_null(a, b, **kw)
    )


def test_zoo_null_is_on_the_same_footing_as_the_zoo_score():
    """The bug this replaces: a 3-model mean-of-upper-triangle score was compared
    against a null built from a single pair, which is not a test of that score.

    A one-pair null is the wrong reference on two counts: it is centred on that
    one pair's geometry, and it has the spread of a single pair rather than of a
    mean over M(M-1)/2 of them.
    """
    from manylatents.metrics.nulls import zoo_permutation_null

    rng = np.random.default_rng(20)
    centres = rng.normal(size=(4, 5)) * 6.0
    zoo = {
        "a": rng.normal(size=(60, 5)).astype(np.float32),
        "b": rng.normal(size=(60, 5)).astype(np.float32),
        # Clustered, so its permutation null sits at a different level than a-b's.
        "c": (centres[rng.integers(0, 4, 60)] + 0.3 * rng.normal(size=(60, 5))).astype(np.float32),
    }
    kw = dict(measure="diffop_angles", n_components=4, knn=8)
    names = list(zoo)

    zoo_null = zoo_permutation_null(zoo, n_perm=60, seed=0, **kw)
    pair_nulls = {
        (names[i], names[j]): permutation_null(zoo[names[i]], zoo[names[j]],
                                               n_perm=60, seed=0, **kw)
        for i in range(3) for j in range(i + 1, 3)
    }

    # Same footing: the aggregate null is centred where the mean of the per-pair
    # nulls is, not where any single pair's null is.
    assert zoo_null.mean() == pytest.approx(
        float(np.mean([v.mean() for v in pair_nulls.values()])), abs=0.01
    )
    # ...and it is strictly tighter, because it averages three pairs.
    assert zoo_null.std() < min(v.std() for v in pair_nulls.values())

    # Consequence: the p-value for the aggregate score genuinely differs
    # depending on which null it is read against.
    from manylatents.metrics.platonic_convergence import PlatonicConvergence
    observed = PlatonicConvergence(embeddings=zoo, **kw)["score"]
    p_zoo = empirical_p(observed, zoo_null, higher_is_better=True)
    p_pair = empirical_p(observed, pair_nulls[("a", "b")], higher_is_better=True)
    assert p_zoo != pytest.approx(p_pair)


def test_zoo_null_rejects_single_model():
    from manylatents.metrics.nulls import zoo_permutation_null

    rng = np.random.default_rng(15)
    with pytest.raises(ValueError, match="at least 2 models"):
        zoo_permutation_null({"a": rng.normal(size=(20, 3))}, measure="mutual_knn", n_perm=2)


# ---------------------------------------------------------------------------
# Operator-input zoo null
#
# Row-permuting activations yields exactly Pi op Pi.T, so on cached operators
# this null is exact — not an approximation of the activation-based null.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("measure", ["diffop_frobenius", "diffop_angles"])
def test_operator_zoo_null_equals_activation_zoo_null(measure):
    """Load-bearing: identical null draws, not merely a similar distribution."""
    from manylatents.metrics.diffop_alignment import build_operator
    from manylatents.metrics.nulls import (
        zoo_permutation_null,
        zoo_permutation_null_from_operators,
    )

    rng = np.random.default_rng(20)
    base = rng.normal(size=(50, 5)).astype(np.float32)
    zoo = {
        "a": base,
        "b": rng.normal(size=(50, 5)).astype(np.float32),
        "c": (base + 0.05 * rng.normal(size=(50, 5))).astype(np.float32),
    }
    ops = {name: build_operator(acts, knn=6) for name, acts in zoo.items()}

    from_acts = zoo_permutation_null(zoo, measure=measure, n_perm=25, seed=3,
                                     n_components=4, knn=6)
    from_ops = zoo_permutation_null_from_operators(ops, measure=measure, n_perm=25,
                                                   seed=3, n_components=4)
    np.testing.assert_allclose(from_ops, from_acts, rtol=0, atol=0)


def test_operator_zoo_null_rejects_mutual_knn_and_single_model():
    from manylatents.metrics.diffop_alignment import build_operator
    from manylatents.metrics.nulls import zoo_permutation_null_from_operators

    rng = np.random.default_rng(21)
    ops = {
        "a": build_operator(rng.normal(size=(30, 3)).astype(np.float32), knn=5),
        "b": build_operator(rng.normal(size=(30, 3)).astype(np.float32), knn=5),
    }
    with pytest.raises(ValueError, match="mutual_knn"):
        zoo_permutation_null_from_operators(ops, measure="mutual_knn", n_perm=2)
    with pytest.raises(ValueError, match="at least 2 models"):
        zoo_permutation_null_from_operators({"a": ops["a"]}, measure="diffop_angles",
                                            n_perm=2)


def test_operator_zoo_null_is_deterministic_and_correctly_shaped():
    from manylatents.metrics.diffop_alignment import build_operator
    from manylatents.metrics.nulls import zoo_permutation_null_from_operators

    rng = np.random.default_rng(22)
    ops = {
        name: build_operator(rng.normal(size=(40, 4)).astype(np.float32), knn=6)
        for name in ("a", "b", "c")
    }
    kw = dict(measure="diffop_angles", n_perm=12, seed=9, n_components=3)
    first = zoo_permutation_null_from_operators(ops, **kw)
    assert first.shape == (12,)
    np.testing.assert_allclose(first, zoo_permutation_null_from_operators(ops, **kw))


def test_observed_beats_operator_null_when_correspondence_is_real():
    """Sanity: near-identical operators clear their own permutation null."""
    from manylatents.metrics.diffop_alignment import build_operator
    from manylatents.metrics.nulls import zoo_permutation_null_from_operators
    from manylatents.metrics.platonic_convergence import alignment_matrix_from_operators

    rng = np.random.default_rng(23)
    base = rng.normal(size=(60, 5)).astype(np.float32)
    ops = {
        "a": build_operator(base, knn=8),
        "b": build_operator(base + 0.02 * rng.normal(size=(60, 5)).astype(np.float32), knn=8),
    }
    names, mat = alignment_matrix_from_operators(ops, measure="diffop_angles", n_components=4)
    observed = float(mat[np.triu_indices(len(names), k=1)].mean())
    null = zoo_permutation_null_from_operators(ops, measure="diffop_angles", n_perm=99,
                                               seed=0, n_components=4)
    assert empirical_p(observed, null, higher_is_better=True) < 0.05


def test_split_half_null_accepts_a_fixed_bandwidth():
    """The floor must be measurable at the same bandwidth as the scores.

    Measured at the adaptive kNN bandwidth, both halves are near-uniform
    operators whose spectra almost coincide, so the floor is biased toward zero
    — the same over-smoothing that motivated the fixed bandwidth in the first
    place. A floor built at a different bandwidth from the score it gates is
    not a control.
    """
    rng = np.random.default_rng(21)
    acts = rng.normal(size=(120, 6)).astype(np.float32)

    adaptive = split_half_spectral_null(acts, n_splits=5, n_components=3, knn=5, seed=0)
    localized = split_half_spectral_null(
        acts, n_splits=5, n_components=3, knn=5, seed=0, sigma_scale=0.25
    )
    over_smoothed = split_half_spectral_null(
        acts, n_splits=5, n_components=3, knn=5, seed=0, sigma_scale=5.0
    )

    assert localized.shape == (5,)
    assert np.all(np.isfinite(localized))
    assert not np.allclose(localized, adaptive)
    # The bias the fixed bandwidth exists to avoid: at sigma_scale=5 the halves
    # are near-uniform matrices whose spectra almost coincide, and the floor
    # collapses by two orders of magnitude.
    assert float(over_smoothed.mean()) < 0.1 * float(localized.mean())


def test_split_half_null_default_is_unchanged_by_the_sigma_scale_argument():
    rng = np.random.default_rng(22)
    acts = rng.normal(size=(80, 4)).astype(np.float32)

    implicit = split_half_spectral_null(acts, n_splits=4, n_components=3, knn=5, seed=0)
    explicit = split_half_spectral_null(
        acts, n_splits=4, n_components=3, knn=5, seed=0, sigma_scale=None
    )

    assert np.array_equal(implicit, explicit)


def test_split_half_bandwidth_guard_still_applies_under_fixed_bandwidth():
    """``knn`` is inert once ``sigma_scale`` is set, but the guard is cheap and
    a caller passing an over-wide knn is signalling a misconfigured call."""
    rng = np.random.default_rng(23)
    acts = rng.normal(size=(20, 3)).astype(np.float32)

    with pytest.raises(ValueError, match="must be < half"):
        split_half_spectral_null(acts, n_splits=2, n_components=2, knn=10, seed=0,
                                 sigma_scale=0.25)
