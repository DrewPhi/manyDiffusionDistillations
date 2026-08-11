"""Bandwidth control and effective-neighbour diagnostics for the diffusion operator.

Two things are covered here:

1. **Regression guard.** Adding the fixed-bandwidth (``sigma_scale``) mode must
   not perturb a single bit of the two pre-existing bandwidth paths, because
   every cached operator artifact in the downstream projects was built with
   them. The guard has two layers:

   * ``_legacy_gauge`` reimplements the exact pre-change kernel arithmetic
     inside this file, and every legacy case is compared with
     ``np.array_equal`` — bit-exact, no tolerance, and portable across
     numpy/scipy versions.
   * ``LEGACY_SHA256`` records the SHA-256 of the raw output bytes as observed
     when the change was made, so byte identity is asserted directly. If these
     digests ever drift *while* the ``np.array_equal`` layer still passes, the
     cause is a numpy/scipy/BLAS change, not a regression in this module.

2. **New behaviour.** The fixed global bandwidth ``sigma = sigma_scale * median``
   and the per-row participation ratio ``effective_neighbors``.
"""
import hashlib

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from manylatents.callbacks.diffusion_operator import (
    DiffusionGauge,
    build_diffusion_operator,
    effective_neighbors,
    solve_sigma_scale,
)
from manylatents.utils.kernel_utils import symmetric_diffusion_operator


def _golden_X() -> np.ndarray:
    """The fixed probe matrix the recorded digests were taken on."""
    return np.random.default_rng(0).normal(size=(40, 7))


def _legacy_gauge(X, knn=35, alpha=1.0, symmetric=False, metric="euclidean"):
    """Verbatim copy of DiffusionGauge.__call__ from before sigma_scale existed."""
    distances = squareform(pdist(X, metric=metric))

    if knn is not None:
        sorted_dists = np.sort(distances, axis=1)
        sigma = sorted_dists[:, min(knn, distances.shape[0] - 1)]
        sigma = np.maximum(sigma, 1e-10)
        kernel = np.exp(-distances**2 / (sigma[:, None] * sigma[None, :]))
    else:
        sigma = np.median(distances[distances > 0])
        kernel = np.exp(-distances**2 / (2 * sigma**2))

    np.fill_diagonal(kernel, 0)

    if symmetric:
        return symmetric_diffusion_operator(kernel, alpha=alpha)
    row_sums = kernel.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    return kernel / row_sums


LEGACY_KWARGS = {
    "knn_default": dict(),
    "knn_5": dict(knn=5),
    "knn_5_sym": dict(knn=5, symmetric=True),
    "median": dict(knn=None),
    "median_sym": dict(knn=None, symmetric=True),
    "knn_5_cosine": dict(knn=5, metric="cosine"),
    "median_cosine": dict(knn=None, metric="cosine"),
    "knn_5_alpha0_sym": dict(knn=5, alpha=0.0, symmetric=True),
}

# sha256 of the raw float64 bytes produced by the pre-sigma_scale implementation
# on _golden_X(), recorded before the change was written.
LEGACY_SHA256 = {
    "knn_default": "5734482cad52496cfdfde6be1046b0ba6000f23a66a0411e0dae62b188556527",
    "knn_5": "8200b454e211ce2f52de2fff68456bfffc010a906292f5047663c83b4070e9b4",
    "knn_5_sym": "95086270425020748fd5e91be9803b6c99104df9b1f230efbaa1c843c4d56e3e",
    "median": "99e017f4f774da0a81d85e2dc9e0222b57fbf10cb38ea815a5019b5bed16dc07",
    "median_sym": "f5331e6405a009d47edbd17852289d3e2296b2c3924f7adfc707e031ed6ccb79",
    "knn_5_cosine": "eb324623c132faf66968e3bb621c6f471d96f6500b325bf3a99bc92f2d20cdd7",
    "median_cosine": "c72f43c581490f5c92334580419f698c7865d993d266fe1334436016d3588abc",
    "knn_5_alpha0_sym": "ce084b8fda0cfca1031908d0b0a23fa8b5376e746a0d83b0de7dc65a774488b0",
}


# =============================================================================
# Regression guard: sigma_scale=None must be byte-identical to the old code
# =============================================================================

@pytest.mark.parametrize("case", sorted(LEGACY_KWARGS))
def test_legacy_paths_are_bit_identical(case):
    """knn and median bandwidths reproduce the pre-change arithmetic exactly."""
    X = _golden_X()
    kwargs = LEGACY_KWARGS[case]

    op = DiffusionGauge(**kwargs)(X)
    expected = _legacy_gauge(X, **kwargs)

    assert op.dtype == expected.dtype
    assert np.array_equal(op, expected), f"{case} drifted from the legacy formula"


@pytest.mark.parametrize("case", sorted(LEGACY_KWARGS))
def test_legacy_paths_match_recorded_digests(case):
    """Raw output bytes are unchanged from the recorded pre-change digests."""
    op = DiffusionGauge(**LEGACY_KWARGS[case])(_golden_X())

    assert hashlib.sha256(op.tobytes()).hexdigest() == LEGACY_SHA256[case]


@pytest.mark.parametrize("case", sorted(LEGACY_KWARGS))
def test_explicit_sigma_scale_none_is_bit_identical(case):
    """Passing sigma_scale=None explicitly is the same as not passing it."""
    X = _golden_X()

    op = DiffusionGauge(sigma_scale=None, **LEGACY_KWARGS[case])(X)

    assert np.array_equal(op, _legacy_gauge(X, **LEGACY_KWARGS[case]))


def test_build_diffusion_operator_default_is_bit_identical():
    """The public entry point's default is unchanged too."""
    X = _golden_X()

    op = build_diffusion_operator(X, method="diffusion", knn=5)

    assert np.array_equal(op, _legacy_gauge(X, knn=5))
    assert hashlib.sha256(op.tobytes()).hexdigest() == LEGACY_SHA256["knn_5"]


# =============================================================================
# Fixed global bandwidth
# =============================================================================

def _reference_fixed_sigma_op(X, c, metric="euclidean"):
    d = squareform(pdist(X, metric=metric))
    sigma = c * np.median(d[d > 0])
    k = np.exp(-d**2 / (2 * sigma**2))
    np.fill_diagonal(k, 0)
    return k / np.maximum(k.sum(axis=1, keepdims=True), 1e-10)


def test_sigma_scale_matches_c_times_median():
    """sigma = sigma_scale * median(nonzero distances), a single global scalar."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(48, 5))

    op = DiffusionGauge(sigma_scale=0.25)(X)

    assert np.array_equal(op, _reference_fixed_sigma_op(X, 0.25))


def test_sigma_scale_one_matches_median_path():
    """sigma_scale=1.0 reduces to the existing global-median bandwidth."""
    X = _golden_X()

    op = DiffusionGauge(sigma_scale=1.0)(X)

    np.testing.assert_allclose(op, _legacy_gauge(X, knn=None), rtol=1e-12, atol=0)


def test_sigma_scale_takes_precedence_over_knn():
    """When both are set, sigma_scale wins and knn is ignored."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 6))

    with_knn = DiffusionGauge(knn=5, sigma_scale=0.25)(X)
    without_knn = DiffusionGauge(knn=None, sigma_scale=0.25)(X)

    assert np.array_equal(with_knn, without_knn)
    assert not np.allclose(with_knn, DiffusionGauge(knn=5)(X))


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.25])
def test_sigma_scale_must_be_positive(bad):
    with pytest.raises(ValueError, match="sigma_scale"):
        DiffusionGauge(sigma_scale=bad)


def test_small_sigma_scale_concentrates_the_operator():
    """A small c sharpens the kernel: fewer effective neighbours than median or kNN."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(128, 8))

    tight = effective_neighbors(DiffusionGauge(sigma_scale=0.25)(X)).mean()
    loose = effective_neighbors(DiffusionGauge(sigma_scale=1.0)(X)).mean()
    adaptive = effective_neighbors(DiffusionGauge(knn=35)(X)).mean()

    assert tight < loose
    assert tight < adaptive


def test_build_diffusion_operator_threads_sigma_scale():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 5))

    op = build_diffusion_operator(X, method="diffusion", sigma_scale=0.25)

    assert np.array_equal(op, _reference_fixed_sigma_op(X, 0.25))


def test_build_diffusion_operator_accepts_torch_tensor_with_sigma_scale():
    import torch

    X = torch.randn(32, 4, generator=torch.Generator().manual_seed(0))

    op = build_diffusion_operator(X, method="diffusion", sigma_scale=0.25)

    assert op.shape == (32, 32)
    np.testing.assert_allclose(op.sum(axis=1), 1.0, rtol=1e-10)


def test_sigma_scale_still_row_stochastic_and_symmetric_mode_works():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 4))

    op = DiffusionGauge(sigma_scale=0.25)(X)
    sym = DiffusionGauge(sigma_scale=0.25, symmetric=True)(X)

    np.testing.assert_allclose(op.sum(axis=1), 1.0, rtol=1e-10)
    np.testing.assert_allclose(sym, sym.T, rtol=1e-10)


# =============================================================================
# effective_neighbors
# =============================================================================

def test_effective_neighbors_uniform_row_gives_n():
    n = 16
    op = np.full((3, n), 1.0 / n)

    np.testing.assert_allclose(effective_neighbors(op), [n, n, n], rtol=0, atol=1e-12)


def test_effective_neighbors_one_hot_row_gives_one():
    op = np.eye(5)

    np.testing.assert_allclose(effective_neighbors(op), np.ones(5), rtol=0, atol=1e-12)


def test_effective_neighbors_two_equal_weights_gives_two():
    op = np.array([[0.5, 0.5, 0.0, 0.0]])

    np.testing.assert_allclose(effective_neighbors(op), [2.0], rtol=0, atol=1e-12)


def test_effective_neighbors_mixed_rows():
    op = np.array([
        [0.25, 0.25, 0.25, 0.25],   # -> 4
        [1.0, 0.0, 0.0, 0.0],       # -> 1
        [0.5, 0.5, 0.0, 0.0],       # -> 2
    ])

    np.testing.assert_allclose(effective_neighbors(op), [4.0, 1.0, 2.0], atol=1e-12)


def test_effective_neighbors_normalizes_unnormalized_rows():
    """Rows that do not sum to 1 are normalized first, so scale is irrelevant."""
    op = np.array([[2.0, 2.0, 2.0, 2.0], [7.0, 7.0, 0.0, 0.0]])

    np.testing.assert_allclose(effective_neighbors(op), [4.0, 2.0], atol=1e-12)


def test_effective_neighbors_shape_and_bounds():
    rng = np.random.default_rng(8)
    op = DiffusionGauge(knn=5)(rng.normal(size=(64, 6)))

    eff = effective_neighbors(op)

    assert eff.shape == (64,)
    assert np.all(eff >= 1.0 - 1e-9)
    assert np.all(eff <= 64.0 + 1e-9)


def test_effective_neighbors_all_zero_row_does_not_nan():
    op = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])

    eff = effective_neighbors(op)

    assert np.isfinite(eff).all()
    np.testing.assert_allclose(eff[1], 2.0, atol=1e-12)


def test_effective_neighbors_rejects_non_2d():
    with pytest.raises(ValueError):
        effective_neighbors(np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Bandwidth solver
#
# A single fixed ``sigma_scale`` does NOT smooth every representation equally:
# on real activations, random-init cells sit at ~3x the effective-N of trained
# cells at the same c. Comparing two arms at one global c therefore partly
# measures how differently they were smoothed. These tests cover the solver
# that equalizes the instrument instead — bisecting c per cell until the mean
# effective-neighbour fraction hits a common target.
# ---------------------------------------------------------------------------

def _clustered(n_per: int = 40, d: int = 8, seed: int = 3) -> np.ndarray:
    """Three well-separated Gaussian blobs — structure the operator can see."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0] * d, [6.0] + [0.0] * (d - 1), [0.0, 6.0] + [0.0] * (d - 2)])
    return np.vstack([c + rng.normal(scale=0.5, size=(n_per, d)) for c in centers])


def _frac(acts: np.ndarray, c: float) -> float:
    op = build_diffusion_operator(acts, method="diffusion", sigma_scale=c)
    return float(effective_neighbors(op).mean() / acts.shape[0])


def test_effective_n_fraction_is_monotone_in_sigma_scale():
    """Bisection is only valid if the fraction rises with the bandwidth.

    Asserted rather than assumed: the solver's correctness rests entirely on it.
    """
    acts = _clustered()
    grid = np.linspace(0.05, 1.5, 25)

    fracs = np.array([_frac(acts, c) for c in grid])

    assert np.all(np.diff(fracs) > -1e-9), f"non-monotone: {fracs}"
    assert fracs[0] < fracs[-1]


def test_solve_hits_the_target_fraction_within_tol():
    acts = _clustered()

    scale, achieved = solve_sigma_scale(acts, target_frac=0.35, tol=0.01)

    assert abs(achieved - 0.35) <= 0.01
    assert 0.05 <= scale <= 1.5


def test_solved_scale_reproduces_the_reported_fraction():
    """The returned pair must describe the operator a caller then builds."""
    acts = _clustered()

    scale, achieved = solve_sigma_scale(acts, target_frac=0.4, tol=0.01)

    assert _frac(acts, scale) == pytest.approx(achieved, abs=1e-12)


def test_solve_is_deterministic():
    acts = _clustered()

    first = solve_sigma_scale(acts, target_frac=0.35, tol=0.01)
    second = solve_sigma_scale(acts, target_frac=0.35, tol=0.01)

    assert first == second


def test_tighter_target_gives_smaller_sigma_scale():
    acts = _clustered()

    tight, _ = solve_sigma_scale(acts, target_frac=0.2, tol=0.005)
    loose, _ = solve_sigma_scale(acts, target_frac=0.5, tol=0.005)

    assert tight < loose


def test_unreachable_target_raises_reporting_the_bracket():
    """Above the bracket's reach: the error must say what WAS achievable."""
    acts = _clustered()

    with pytest.raises(ValueError) as exc:
        solve_sigma_scale(acts, target_frac=0.999, lo=0.05, hi=0.2, tol=1e-4)

    message = str(exc.value)
    assert "0.05" in message and "0.2" in message
    assert "0.999" in message


def test_unreachable_target_below_the_bracket_raises():
    acts = _clustered()

    with pytest.raises(ValueError) as exc:
        solve_sigma_scale(acts, target_frac=1e-4, lo=0.5, hi=1.5, tol=1e-6)

    assert "1e-04" in str(exc.value) or "0.0001" in str(exc.value)


def test_endpoint_inside_tolerance_is_returned_without_bisecting():
    acts = _clustered()
    lo = 0.05
    frac_lo = _frac(acts, lo)

    scale, achieved = solve_sigma_scale(acts, target_frac=frac_lo, lo=lo, hi=1.5, tol=0.01)

    assert scale == lo
    assert achieved == pytest.approx(frac_lo, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_frac": 0.0},
        {"target_frac": 1.5},
        {"lo": 0.0},
        {"lo": 1.0, "hi": 0.5},
        {"tol": 0.0},
        {"max_iter": 0},
    ],
)
def test_solver_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        solve_sigma_scale(_clustered(n_per=10), **kwargs)


def test_solver_accepts_3d_singleton_activations():
    """(N, 1, D) snapshots are handled elsewhere in the stack; match that."""
    acts = _clustered(n_per=20)

    flat, _ = solve_sigma_scale(acts, target_frac=0.35, tol=0.01)
    nested, _ = solve_sigma_scale(acts[:, None, :], target_frac=0.35, tol=0.01)

    assert flat == nested
