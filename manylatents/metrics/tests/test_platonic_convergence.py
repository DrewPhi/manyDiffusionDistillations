"""Model x model alignment matrix and the scalar convergence score."""
import numpy as np
import pytest

from manylatents.metrics.platonic_convergence import (
    MEASURES,
    PlatonicConvergence,
    alignment_matrix,
)


def _zoo(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(40, 5)).astype(np.float32)
    return {
        "a": base,
        "b": base.copy(),                                   # identical to a
        "c": rng.normal(size=(40, 5)).astype(np.float32),   # unrelated
    }


@pytest.mark.parametrize("measure", MEASURES)
def test_matrix_shape_and_symmetry(measure):
    names, mat = alignment_matrix(_zoo(), measure=measure, k=5, n_components=3, knn=5)
    assert names == ["a", "b", "c"]
    assert mat.shape == (3, 3)
    np.testing.assert_allclose(mat, mat.T, atol=1e-10)


def test_identical_models_score_best_under_each_measure():
    """a and b are identical, so the a-b cell must beat the a-c cell."""
    names, mat = alignment_matrix(_zoo(), measure="mutual_knn", k=5)
    assert mat[0, 1] > mat[0, 2]

    names, mat = alignment_matrix(_zoo(), measure="diffop_angles", n_components=3, knn=5)
    assert mat[0, 1] > mat[0, 2]

    names, mat = alignment_matrix(_zoo(), measure="diffop_frobenius", knn=5)
    assert mat[0, 1] < mat[0, 2]  # distance: lower is better


def test_diagonal_is_self_alignment():
    _, mat = alignment_matrix(_zoo(), measure="diffop_frobenius", knn=5)
    np.testing.assert_allclose(np.diag(mat), np.zeros(3), atol=1e-10)

    _, mat = alignment_matrix(_zoo(), measure="mutual_knn", k=5)
    np.testing.assert_allclose(np.diag(mat), np.ones(3), atol=1e-10)


def test_score_is_mean_of_upper_triangle():
    out = PlatonicConvergence(embeddings=_zoo(), measure="mutual_knn", k=5)
    mat = np.asarray(out["matrix"])
    iu = np.triu_indices(3, k=1)
    assert out["score"] == pytest.approx(float(mat[iu].mean()))


def test_output_declares_direction():
    assert PlatonicConvergence(embeddings=_zoo(), measure="mutual_knn", k=5)["higher_is_better"]
    assert not PlatonicConvergence(
        embeddings=_zoo(), measure="diffop_frobenius", knn=5
    )["higher_is_better"]


def test_matrix_is_json_serialisable():
    import json

    out = PlatonicConvergence(embeddings=_zoo(), measure="mutual_knn", k=5)
    assert json.loads(json.dumps(out))["models"] == ["a", "b", "c"]


def test_requires_two_models():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least 2 models"):
        alignment_matrix({"only": rng.normal(size=(10, 2)).astype(np.float32)}, measure="mutual_knn", k=3)


def test_mismatched_sample_counts_raise():
    rng = np.random.default_rng(0)
    acts = {
        "a": rng.normal(size=(10, 2)).astype(np.float32),
        "b": rng.normal(size=(11, 2)).astype(np.float32),
    }
    with pytest.raises(ValueError, match="Sample count mismatch"):
        alignment_matrix(acts, measure="mutual_knn", k=3)


@pytest.mark.parametrize("measure", MEASURES)
def test_all_measures_accept_singleton_middle_axis(measure):
    """(N, 1, D) snapshots must work identically for every measure.

    mutual_knn squeezes them via _ensure_2d; before the fix the diffop measures
    passed the 3-D array straight to scipy and died with 'A 2-dimensional array
    must be passed.'
    """
    zoo2d = _zoo(seed=1)
    zoo3d = {name: arr[:, None, :] for name, arr in zoo2d.items()}
    kw = dict(measure=measure, k=5, n_components=3, knn=5)

    names_3d, mat_3d = alignment_matrix(zoo3d, **kw)
    names_2d, mat_2d = alignment_matrix(zoo2d, **kw)
    assert names_3d == names_2d
    np.testing.assert_allclose(mat_3d, mat_2d, atol=1e-12)


# ---------------------------------------------------------------------------
# Operator-input adapter
#
# The 12-model zoo's diffusion operators are already on disk, so the metric has
# to be reachable without the activations that produced them. These tests pin
# that path to the activation path it replaces.
# ---------------------------------------------------------------------------

DIFFOP_MEASURES = ("diffop_frobenius", "diffop_angles")


@pytest.mark.parametrize("measure", DIFFOP_MEASURES)
def test_operator_path_equals_activation_path(measure):
    """Load-bearing: the adapter must reproduce ``alignment_matrix`` exactly.

    Build operators from activations with the same ``build_operator`` the
    activation path uses, then feed them to the adapter. Any disagreement means
    the cached-operator numbers are not the numbers the tested stack computes.
    """
    from manylatents.metrics.diffop_alignment import build_operator
    from manylatents.metrics.platonic_convergence import alignment_matrix_from_operators

    zoo = _zoo(seed=3)
    ops = {name: build_operator(acts, knn=5) for name, acts in zoo.items()}

    names_act, mat_act = alignment_matrix(zoo, measure=measure, n_components=3, knn=5)
    names_op, mat_op = alignment_matrix_from_operators(ops, measure=measure, n_components=3)

    assert names_op == names_act
    np.testing.assert_allclose(mat_op, mat_act, rtol=0, atol=0)


def test_operator_path_rejects_mutual_knn():
    """mutual_knn needs the point cloud; an operator cannot supply it."""
    from manylatents.metrics.diffop_alignment import build_operator
    from manylatents.metrics.platonic_convergence import alignment_matrix_from_operators

    ops = {name: build_operator(acts, knn=5) for name, acts in _zoo(seed=4).items()}
    with pytest.raises(ValueError, match="mutual_knn"):
        alignment_matrix_from_operators(ops, measure="mutual_knn")


def test_operator_path_diagonal_and_score_conventions():
    """Same diagonal convention and same strict-upper-triangle score."""
    from manylatents.metrics.diffop_alignment import build_operator
    from manylatents.metrics.platonic_convergence import alignment_matrix_from_operators

    ops = {name: build_operator(acts, knn=5) for name, acts in _zoo(seed=5).items()}

    _, mat_fro = alignment_matrix_from_operators(ops, measure="diffop_frobenius")
    np.testing.assert_allclose(np.diag(mat_fro), 0.0, atol=1e-12)
    np.testing.assert_allclose(mat_fro, mat_fro.T, atol=1e-12)

    _, mat_ang = alignment_matrix_from_operators(ops, measure="diffop_angles", n_components=3)
    np.testing.assert_allclose(np.diag(mat_ang), 1.0, atol=1e-12)


def test_operator_path_symmetrizes_row_stochastic_input():
    """Cached operators on disk are row-stochastic, not symmetric.

    ``np.linalg.eigh`` would silently read only one triangle of them, so the
    adapter symmetrizes; feeding it ``op`` must equal feeding it ``(op+op.T)/2``.
    """
    from manylatents.metrics.diffop_alignment import symmetrize_operator
    from manylatents.metrics.platonic_convergence import alignment_matrix_from_operators

    rng = np.random.default_rng(6)
    raw = {}
    for name in ("a", "b", "c"):
        w = rng.random((20, 20)) + 1e-3
        raw[name] = w / w.sum(axis=1, keepdims=True)   # row-stochastic, asymmetric
    assert not np.allclose(raw["a"], raw["a"].T)

    pre = {name: symmetrize_operator(op) for name, op in raw.items()}
    for measure in DIFFOP_MEASURES:
        _, mat_raw = alignment_matrix_from_operators(raw, measure=measure, n_components=3)
        _, mat_pre = alignment_matrix_from_operators(pre, measure=measure, n_components=3)
        np.testing.assert_allclose(mat_raw, mat_pre, rtol=0, atol=0)


def test_operator_path_rejects_shape_mismatch_and_single_model():
    from manylatents.metrics.platonic_convergence import alignment_matrix_from_operators

    rng = np.random.default_rng(7)
    ok = np.eye(8) + 0.01 * rng.normal(size=(8, 8))
    with pytest.raises(ValueError, match="at least 2 models"):
        alignment_matrix_from_operators({"a": ok}, measure="diffop_angles")
    with pytest.raises(ValueError, match="[Ss]hape mismatch"):
        alignment_matrix_from_operators(
            {"a": ok, "b": np.eye(9)}, measure="diffop_angles"
        )
    with pytest.raises(ValueError, match="square"):
        alignment_matrix_from_operators(
            {"a": np.eye(8)[:, :7], "b": np.eye(8)[:, :7]}, measure="diffop_angles"
        )
