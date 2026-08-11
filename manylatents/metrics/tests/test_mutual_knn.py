"""Hand-computed mutual-kNN overlap.

A = [0, 1, 2, 100], B = [0, 1, 2, 3], k=2 (self excluded):
  A neighbours: i0={1,2}  i1={0,2}  i2={1,0}  i3={2,1}
  B neighbours: i0={1,2}  i1={0,2}  i2={1,3}  i3={2,1}
  overlap/k:      1.0       1.0       0.5       1.0   -> mean 0.875
"""
import numpy as np
import pytest

from manylatents.metrics.mutual_knn import MutualKNN, mutual_knn_pairwise


def test_mutual_knn_pairwise_hand_computed():
    a = np.array([[0.0], [1.0], [2.0], [100.0]], dtype=np.float32)
    b = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    per_sample = mutual_knn_pairwise(a, b, k=2)
    np.testing.assert_allclose(per_sample, [1.0, 1.0, 0.5, 1.0])
    assert float(per_sample.mean()) == pytest.approx(0.875)


def test_identical_embeddings_score_one():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 5)).astype(np.float32)
    assert float(mutual_knn_pairwise(a, a.copy(), k=5).mean()) == pytest.approx(1.0)


def test_differs_from_jaccard():
    """|A∩B|/k and |A∩B|/|A∪B| must not be the same number on partial overlap."""
    from manylatents.metrics.cross_modal_jaccard import cross_modal_jaccard_pairwise

    a = np.array([[0.0], [1.0], [2.0], [100.0]], dtype=np.float32)
    b = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    assert mutual_knn_pairwise(a, b, k=2)[2] != cross_modal_jaccard_pairwise(a, b, k=2)[2]


def test_registered_metric_over_model_dict():
    rng = np.random.default_rng(1)
    acts = {
        "m1": rng.normal(size=(30, 4)).astype(np.float32),
        "m2": rng.normal(size=(30, 6)).astype(np.float32),
    }
    score = MutualKNN(embeddings=acts, k=3)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_k_must_be_less_than_n():
    a = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="must be < n_samples"):
        mutual_knn_pairwise(a, a, k=4)


def test_sample_count_mismatch_raises():
    with pytest.raises(ValueError, match="Sample count mismatch"):
        mutual_knn_pairwise(
            np.zeros((4, 2), dtype=np.float32), np.zeros((5, 2), dtype=np.float32), k=2
        )
