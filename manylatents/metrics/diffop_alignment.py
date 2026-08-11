"""Diffusion-operator alignment: relative Frobenius distance and principal angles.

`compute_multi_model_spread`'s "spectral" mode compares sorted eigenvalue
vectors, which is blind to eigenvectors: two operators with identical spectra
but orthogonal eigenspaces score as perfectly aligned. The subspace alignment
here compares the eigenspaces themselves via principal angles.
"""

from typing import Any, Dict, Union

import numpy as np

from manylatents.callbacks.diffusion_operator import DiffusionGauge
from manylatents.metrics.registry import register_metric


def build_operator(acts: np.ndarray, knn: int = 35, alpha: float = 1.0) -> np.ndarray:
    """Build a symmetric diffusion operator from (N, D) activations.

    Symmetric so that ``np.linalg.eigh`` returns real eigenvectors; the default
    row-stochastic operator is non-symmetric and would need a complex solver.
    """
    gauge = DiffusionGauge(knn=knn, alpha=alpha, symmetric=True)
    op = np.asarray(gauge(acts), dtype=float)
    return 0.5 * (op + op.T)


def top_eigvecs(op: np.ndarray, n_components: int) -> np.ndarray:
    """Eigenvectors of the ``n_components`` largest-magnitude eigenvalues."""
    sym = 0.5 * (op + op.T)
    n_components = min(n_components, sym.shape[0])
    eigvals, eigvecs = np.linalg.eigh(sym)
    order = np.argsort(np.abs(eigvals))[::-1][:n_components]
    return eigvecs[:, order]


def principal_angle_cosines(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the column spaces of u and v.

    Returns values in [0, 1], descending. 1.0 means a shared direction;
    0.0 means orthogonal.
    """
    qu, _ = np.linalg.qr(np.asarray(u, dtype=float))
    qv, _ = np.linalg.qr(np.asarray(v, dtype=float))
    sing = np.linalg.svd(qu.T @ qv, compute_uv=False)
    return np.clip(sing, 0.0, 1.0)


def diffop_frobenius_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Relative Frobenius distance ||P-Q||_F / mean(||P||_F, ||Q||_F).

    Lower is better; 0.0 means identical. Relative rather than raw so that
    operator magnitude cannot drive the comparison.
    """
    num = float(np.linalg.norm(p - q, "fro"))
    den = 0.5 * (float(np.linalg.norm(p, "fro")) + float(np.linalg.norm(q, "fro")))
    return num / den if den > 0 else 0.0


def diffop_subspace_alignment(p: np.ndarray, q: np.ndarray, n_components: int = 10) -> float:
    """Mean cosine of principal angles between the top eigen-subspaces.

    Higher is better; 1.0 means the leading eigenspaces coincide.
    """
    cosines = principal_angle_cosines(top_eigvecs(p, n_components), top_eigvecs(q, n_components))
    return float(cosines.mean())


@register_metric(
    aliases=["diffop_alignment"],
    default_params={"measure": "diffop_angles", "n_components": 10, "knn": 35},
    description="Diffusion-operator alignment (relative Frobenius or principal angles)",
)
def DiffopAlignment(
    embeddings: Union[np.ndarray, Dict[str, np.ndarray]],
    dataset=None,
    module=None,
    measure: str = "diffop_angles",
    n_components: int = 10,
    knn: int = 35,
) -> Dict[str, Any]:
    """Mean pairwise diffusion-operator alignment across models.

    Args:
        embeddings: Dict mapping model name to an index-aligned (N, D) array.
        dataset: Unused; present for the metric protocol.
        module: Unused; present for the metric protocol.
        measure: "diffop_angles" (higher better) or "diffop_frobenius" (lower better).
        n_components: Eigen-subspace size for the angle measure.
        knn: Adaptive-bandwidth neighbour count for operator construction.

    Returns:
        {"score": float, "measure": str, "higher_is_better": bool,
         "pairs": {pair_name: float}}
    """
    if measure not in ("diffop_angles", "diffop_frobenius"):
        raise ValueError(f"Unknown measure: {measure}")

    names = list(embeddings.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 models for diffop alignment")

    ops = {name: build_operator(embeddings[name], knn=knn) for name in names}

    pairs: Dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}_{names[j]}"
            if measure == "diffop_angles":
                pairs[key] = diffop_subspace_alignment(
                    ops[names[i]], ops[names[j]], n_components=n_components
                )
            else:
                pairs[key] = diffop_frobenius_distance(ops[names[i]], ops[names[j]])

    return {
        "score": float(np.mean(list(pairs.values()))),
        "measure": measure,
        "higher_is_better": measure == "diffop_angles",
        "pairs": pairs,
    }
