"""Principal-angle alignment and Frobenius distance between diffusion operators."""
import numpy as np
import pytest

from manylatents.metrics.diffop_alignment import (
    DiffopAlignment,
    build_operator,
    diffop_frobenius_distance,
    diffop_subspace_alignment,
    principal_angle_cosines,
    top_eigvecs,
)


def test_principal_angles_hand_computed():
    """span{e1,e2} vs span{e1, cos(60°)e2 + sin(60°)e3} -> cosines [1, 0.5]."""
    t = np.pi / 3.0
    u = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    v = np.array([[1.0, 0.0], [0.0, np.cos(t)], [0.0, np.sin(t)]])
    np.testing.assert_allclose(principal_angle_cosines(u, v), [1.0, 0.5], atol=1e-10)


def test_identical_subspace_all_cosines_one():
    rng = np.random.default_rng(0)
    u = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    np.testing.assert_allclose(principal_angle_cosines(u, u.copy()), np.ones(3), atol=1e-10)


def test_orthogonal_subspaces_all_cosines_zero():
    u = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    v = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(principal_angle_cosines(u, v), np.zeros(2), atol=1e-10)


def test_frobenius_distance_zero_for_identical():
    rng = np.random.default_rng(1)
    p = build_operator(rng.normal(size=(30, 4)).astype(np.float32), knn=5)
    assert diffop_frobenius_distance(p, p.copy()) == pytest.approx(0.0, abs=1e-12)


def test_frobenius_distance_is_scale_free():
    """Relative distance must not change when both operators are rescaled."""
    rng = np.random.default_rng(2)
    p = build_operator(rng.normal(size=(30, 4)).astype(np.float32), knn=5)
    q = build_operator(rng.normal(size=(30, 4)).astype(np.float32), knn=5)
    assert diffop_frobenius_distance(p, q) == pytest.approx(
        diffop_frobenius_distance(3.0 * p, 3.0 * q)
    )


def test_operator_is_symmetric():
    rng = np.random.default_rng(3)
    p = build_operator(rng.normal(size=(20, 3)).astype(np.float32), knn=5)
    np.testing.assert_allclose(p, p.T, atol=1e-10)


def test_subspace_alignment_identical_is_one():
    rng = np.random.default_rng(4)
    p = build_operator(rng.normal(size=(30, 4)).astype(np.float32), knn=5)
    assert diffop_subspace_alignment(p, p.copy(), n_components=3) == pytest.approx(1.0, abs=1e-8)


def test_spectra_can_match_while_subspaces_do_not():
    """The property the old 'spectral' branch misses: identical eigenvalues,
    different eigenvectors. Frobenius/angle must see the difference."""
    eigvals = np.array([3.0, 2.0, 1.0, 0.5])
    rng = np.random.default_rng(5)
    q1 = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    q2 = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    p = q1 @ np.diag(eigvals) @ q1.T
    q = q2 @ np.diag(eigvals) @ q2.T
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(p)), np.sort(np.linalg.eigvalsh(q)), atol=1e-8
    )
    assert diffop_subspace_alignment(p, q, n_components=2) < 0.99
    assert diffop_frobenius_distance(p, q) > 1e-3


def test_top_eigvecs_shape_and_orthonormal():
    rng = np.random.default_rng(6)
    p = build_operator(rng.normal(size=(25, 3)).astype(np.float32), knn=5)
    u = top_eigvecs(p, 4)
    assert u.shape == (25, 4)
    np.testing.assert_allclose(u.T @ u, np.eye(4), atol=1e-8)


def test_registered_metric_returns_directional_dict():
    rng = np.random.default_rng(7)
    acts = {
        "m1": rng.normal(size=(30, 4)).astype(np.float32),
        "m2": rng.normal(size=(30, 4)).astype(np.float32),
    }
    out = DiffopAlignment(embeddings=acts, measure="diffop_angles", knn=5, n_components=3)
    assert out["higher_is_better"] is True
    assert 0.0 <= out["score"] <= 1.0

    out_fro = DiffopAlignment(embeddings=acts, measure="diffop_frobenius", knn=5)
    assert out_fro["higher_is_better"] is False
    assert out_fro["score"] >= 0.0


def test_unknown_measure_raises():
    rng = np.random.default_rng(8)
    acts = {"m1": rng.normal(size=(10, 2)).astype(np.float32),
            "m2": rng.normal(size=(10, 2)).astype(np.float32)}
    with pytest.raises(ValueError, match="Unknown measure"):
        DiffopAlignment(embeddings=acts, measure="nope", knn=3)


def test_build_operator_accepts_singleton_middle_axis():
    """(N, 1, D) activations must squeeze, matching mutual_knn's convention."""
    from manylatents.metrics.diffop_alignment import build_operator

    rng = np.random.default_rng(31)
    acts = rng.normal(size=(20, 4)).astype(np.float32)
    np.testing.assert_allclose(
        build_operator(acts[:, None, :], knn=5), build_operator(acts, knn=5), atol=1e-12
    )
