# manylatents/callbacks/diffusion_operator.py
"""Diffusion operator construction and trajectory analysis.

Build diffusion operators from representations for spectral analysis.
These are used by the activation tracker callback during training and
by standalone spectral diagnostic pipelines.

Usage:
    # Direct computation
    from manylatents.callbacks.diffusion_operator import build_diffusion_operator
    diff_op = build_diffusion_operator(embeddings, method="diffusion")

    # Fixed small bandwidth (c x median) instead of the adaptive kNN one
    diff_op = build_diffusion_operator(embeddings, sigma_scale=0.25)

    # Check the bandwidth actually localized the operator
    from manylatents.callbacks.diffusion_operator import effective_neighbors
    effective_neighbors(diff_op).mean()  # << N means real geometry

    # As Lightning callback (see lightning/callbacks/activation_tracker.py)
"""
import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from torch import Tensor

from manylatents.utils.kernel_utils import symmetric_diffusion_operator


# =============================================================================
# Core dispatch
# =============================================================================

@functools.singledispatch
def build_diffusion_operator(
    source: Any,
    /,
    method: str = "diffusion",
    **kwargs,
) -> np.ndarray:
    """Build diffusion operator from representations.

    Args:
        source: Input representations (N samples x D features)
        method: Construction method ("diffusion", future: "sae", "attention")
        **kwargs: Method-specific parameters. For method="diffusion" these are
            the :class:`DiffusionGauge` fields: ``knn`` (default 35, adaptive
            k-th-NN bandwidth), ``sigma_scale`` (default None; when set, a fixed
            global bandwidth of ``sigma_scale * median(distances)`` that takes
            precedence over ``knn``), ``alpha``, ``symmetric``, ``metric``.

    Returns:
        Diffusion operator (N, N)
    """
    raise NotImplementedError(
        f"build_diffusion_operator() not implemented for type {type(source)}. "
        f"Expected np.ndarray or torch.Tensor."
    )


@build_diffusion_operator.register(np.ndarray)
def _from_ndarray(source: np.ndarray, /, method: str = "diffusion", **kwargs) -> np.ndarray:
    if method == "diffusion":
        gauge = DiffusionGauge(**kwargs)
        return gauge(source)
    else:
        raise ValueError(f"Unknown method: {method}")


@build_diffusion_operator.register(Tensor)
def _from_tensor(source: Tensor, /, method: str = "diffusion", **kwargs) -> np.ndarray:
    if method == "diffusion":
        gauge = DiffusionGauge(**kwargs)
        return gauge(source)
    else:
        raise ValueError(f"Unknown method: {method}")


# Backward-compatible alias
probe = build_diffusion_operator


# =============================================================================
# Diffusion operator builder
# =============================================================================

@dataclass
class DiffusionGauge:
    """Compute diffusion operator from representations.

    Pipeline:
        representations (N, D) -> pairwise distances -> Gaussian kernel ->
        affinity matrix -> diffusion operator

    Bandwidth (three mutually exclusive modes, checked in this order):
        1. ``sigma_scale`` set -> fixed global bandwidth
           ``sigma = sigma_scale * median(nonzero pairwise distances)``,
           kernel ``exp(-d^2 / (2 sigma^2))``. Takes precedence over ``knn``.
        2. ``knn`` set -> adaptive k-th-nearest-neighbour bandwidth.
        3. neither -> global median bandwidth (``sigma_scale = 1`` in effect).

    Mode 1 is the ``c x median`` parametrization: small ``c`` (c <~ 0.25) keeps
    the operator localized. Both the median bandwidth and the adaptive kNN
    bandwidth over-smooth on high-dimensional activation clouds — measure with
    :func:`effective_neighbors` rather than assuming.

    Attributes:
        knn: Number of neighbors for adaptive bandwidth. If None, uses global bandwidth.
        sigma_scale: If set, use a fixed global bandwidth of this multiple of the
            median nonzero pairwise distance. Overrides ``knn``. Must be > 0.
        alpha: Diffusion normalization parameter (0=graph Laplacian, 1=Laplace-Beltrami)
        symmetric: If True, return symmetric operator D^{-1/2} K D^{-1/2}
        metric: Distance metric for pairwise computation
    """
    knn: Optional[int] = 35
    alpha: float = 1.0
    symmetric: bool = False
    metric: str = "euclidean"
    sigma_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sigma_scale is not None and self.sigma_scale <= 0:
            raise ValueError(
                f"sigma_scale must be > 0 (got {self.sigma_scale}); it multiplies "
                f"the median pairwise distance to give the fixed bandwidth."
            )

    def __call__(self, representations: Any) -> np.ndarray:
        """Compute diffusion operator.

        Args:
            representations: Array/Tensor of shape (N, D)

        Returns:
            Diffusion operator of shape (N, N)
        """
        if isinstance(representations, Tensor):
            representations = representations.detach().cpu().numpy()

        distances = squareform(pdist(representations, metric=self.metric))

        if self.sigma_scale is not None:
            sigma = self.sigma_scale * np.median(distances[distances > 0])
            kernel = np.exp(-distances**2 / (2 * sigma**2))
        elif self.knn is not None:
            sorted_dists = np.sort(distances, axis=1)
            sigma = sorted_dists[:, min(self.knn, distances.shape[0] - 1)]
            sigma = np.maximum(sigma, 1e-10)
            kernel = np.exp(-distances**2 / (sigma[:, None] * sigma[None, :]))
        else:
            sigma = np.median(distances[distances > 0])
            kernel = np.exp(-distances**2 / (2 * sigma**2))

        np.fill_diagonal(kernel, 0)

        if self.symmetric:
            return symmetric_diffusion_operator(kernel, alpha=self.alpha)
        else:
            row_sums = kernel.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            return kernel / row_sums


# =============================================================================
# Bandwidth diagnostics
# =============================================================================

def effective_neighbors(op: np.ndarray) -> np.ndarray:
    """Per-row effective number of neighbours (participation ratio).

    For a row-stochastic operator ``P``, row ``i`` has effective neighbour count

        n_eff(i) = 1 / sum_j p_ij**2

    A row spread uniformly over ``N`` entries gives exactly ``N``; a one-hot row
    gives exactly 1. This is the measurement to use when picking a bandwidth:
    an operator whose mean ``n_eff`` approaches ``N`` is the uniform matrix in
    disguise and carries no geometry.

    Rows are **normalized to sum to 1 before the ratio is taken**, so operators
    that were saved unnormalized (or with a symmetric normalization, which is
    not row-stochastic) still give a meaningful, scale-invariant answer. Rows
    that sum to 0 are reported as 0 effective neighbours rather than NaN.

    Args:
        op: Operator of shape (N, N) — or any (rows, cols) matrix of weights.

    Returns:
        Array of shape (N,) with the effective neighbour count of each row.
    """
    op = np.asarray(op, dtype=np.float64)
    if op.ndim != 2:
        raise ValueError(f"effective_neighbors expects a 2-D matrix, got shape {op.shape}")

    row_sums = op.sum(axis=1, keepdims=True)
    safe = np.where(np.abs(row_sums) > 0, row_sums, 1.0)
    p = op / safe

    sq = (p**2).sum(axis=1)
    out = np.zeros_like(sq)
    np.divide(1.0, sq, out=out, where=sq > 0)
    return out


def solve_sigma_scale(
    acts: Any,
    target_frac: float = 0.35,
    lo: float = 0.05,
    hi: float = 1.5,
    tol: float = 0.01,
    max_iter: int = 25,
    **gauge_kwargs: Any,
) -> Tuple[float, float]:
    """Bandwidth that puts one representation at a chosen effective-N fraction.

    A single fixed ``sigma_scale`` does not smooth every representation equally.
    Measured at the penultimate layer over a 2048-item probe, ``c = 0.25`` gives
    19% effective neighbours on a trained BERT but 32% on the same model at
    random init — so a trained-vs-init comparison run at one global ``c`` partly
    measures how differently the two arms were smoothed. Solving each cell to a
    common ``target_frac`` removes that confound: the instrument is equalized,
    and what is left to differ is the geometry.

    Bisects ``sigma_scale`` on ``mean(effective_neighbors(op)) / N``, which is
    monotonically increasing in ``sigma_scale`` (larger bandwidth spreads every
    row over more points; in the limit the operator is uniform and the fraction
    is 1). ``test_effective_n_fraction_is_monotone_in_sigma_scale`` asserts that
    on real structure rather than leaving it as an assumption.

    Args:
        acts: (N, D) — or (N, 1, D) — representations.
        target_frac: Desired mean effective neighbours as a fraction of N, in
            (0, 1]. ~0.35 is the regime a prior study found the operator usable
            in; near 1 the operator is the uniform matrix and carries no geometry.
        lo: Lower end of the search bracket. Must be > 0.
        hi: Upper end of the search bracket. Must be > ``lo``.
        tol: Absolute tolerance on the achieved fraction.
        max_iter: Maximum bisection steps after the two bracket evaluations.
        **gauge_kwargs: Forwarded to :func:`build_diffusion_operator` (``alpha``,
            ``symmetric``, ``metric``). ``sigma_scale`` is the solved variable
            and is rejected here.

    Returns:
        ``(sigma_scale, achieved_frac)`` — the fraction is the one measured at
        the returned scale, so a caller that rebuilds at that scale gets exactly
        the operator described.

    Raises:
        ValueError: On invalid arguments; when ``target_frac`` lies outside the
            fractions reachable in ``[lo, hi]`` (the message reports the bracket
            and both achieved endpoints); or when bisection has not reached
            ``tol`` within ``max_iter`` steps.
    """
    if not 0.0 < target_frac <= 1.0:
        raise ValueError(f"target_frac must be in (0, 1], got {target_frac}")
    if lo <= 0.0:
        raise ValueError(f"lo must be > 0, got {lo}")
    if hi <= lo:
        raise ValueError(f"hi must be > lo, got lo={lo}, hi={hi}")
    if tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if "sigma_scale" in gauge_kwargs:
        raise ValueError("sigma_scale is what this function solves for; do not pass it")

    x = np.asarray(acts)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.ndim != 2:
        raise ValueError(f"acts must be (N, D) or (N, 1, D), got shape {x.shape}")
    n = x.shape[0]

    def achieved(scale: float) -> float:
        op = build_diffusion_operator(x, method="diffusion", sigma_scale=scale, **gauge_kwargs)
        return float(effective_neighbors(op).mean() / n)

    frac_lo = achieved(lo)
    if abs(frac_lo - target_frac) <= tol:
        return lo, frac_lo
    frac_hi = achieved(hi)
    if abs(frac_hi - target_frac) <= tol:
        return hi, frac_hi

    if not frac_lo < target_frac < frac_hi:
        raise ValueError(
            f"target_frac={target_frac} is unreachable in the bracket "
            f"[{lo}, {hi}]: sigma_scale={lo} gives {frac_lo:.4f} and "
            f"sigma_scale={hi} gives {frac_hi:.4f}. Widen the bracket."
        )

    a, b = lo, hi
    frac_mid = frac_lo
    mid = a
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        frac_mid = achieved(mid)
        if abs(frac_mid - target_frac) <= tol:
            return mid, frac_mid
        if frac_mid < target_frac:
            a = mid
        else:
            b = mid

    raise ValueError(
        f"bisection did not reach target_frac={target_frac} within tol={tol} "
        f"in max_iter={max_iter} steps; last sigma_scale={mid} gave "
        f"{frac_mid:.4f}, bracket narrowed to [{a}, {b}]"
    )


# =============================================================================
# Trajectory analysis (for analyzing operator outputs over time/models)
# =============================================================================

@dataclass
class TrajectoryVisualizer:
    """Embed probe trajectories for visualization.

    Takes a sequence of (step, operator) pairs and embeds them
    in low-dimensional space using PHATE on pairwise distances.
    """
    n_components: int = 2
    distance_metric: Literal["frobenius", "spectral"] = "frobenius"
    phate_knn: int = 5
    phate_t: int = 10

    def compute_distances(self, trajectory: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """Compute pairwise distances between operators in trajectory."""
        operators = [op for _, op in trajectory]

        if self.distance_metric == "frobenius":
            flat = [op.flatten() for op in operators]
            return squareform(pdist(flat, metric="euclidean"))
        elif self.distance_metric == "spectral":
            spectra = []
            for op in operators:
                eigvals = np.linalg.eigvalsh(op)
                eigvals = np.sort(np.abs(eigvals))[::-1]
                spectra.append(eigvals)
            return squareform(pdist(spectra, metric="euclidean"))
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

    def fit_transform(self, trajectory: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """Embed trajectory in low-dimensional space."""
        from manylatents.algorithms.latent.phate import PHATEModule

        distances = self.compute_distances(trajectory)
        sigma = np.median(distances[distances > 0])
        if sigma == 0:
            sigma = 1.0
        similarities = np.exp(-distances**2 / (2 * sigma**2))

        phate = PHATEModule(
            n_components=self.n_components,
            knn=min(self.phate_knn, len(trajectory) - 1),
            t=self.phate_t,
        )
        sim_tensor = torch.from_numpy(similarities).float()
        phate.fit(sim_tensor)
        embedding = phate.transform(sim_tensor)

        return embedding.numpy() if hasattr(embedding, 'numpy') else np.array(embedding)

    def compute_spread(self, trajectory: List[Tuple[int, np.ndarray]]) -> float:
        """Compute spread metric (average pairwise distance)."""
        distances = self.compute_distances(trajectory)
        upper_tri = distances[np.triu_indices(len(trajectory), k=1)]
        return float(np.mean(upper_tri))


def compute_multi_model_spread(
    trajectories: List[List[Tuple[int, np.ndarray]]],
    distance_metric: str = "frobenius",
) -> np.ndarray:
    """Compute spread across models at each timestep.

    Lower spread indicates models are converging to similar representations.
    """
    n_steps = len(trajectories[0])
    n_models = len(trajectories)
    spreads = []

    for step_idx in range(n_steps):
        operators = [traj[step_idx][1] for traj in trajectories]

        if distance_metric == "frobenius":
            flat = [op.flatten() for op in operators]
            if n_models > 1:
                dists = pdist(flat, metric="euclidean")
                spread = float(np.mean(dists))
            else:
                spread = 0.0
        elif distance_metric == "spectral":
            spectra = []
            for op in operators:
                eigvals = np.linalg.eigvalsh(op)
                eigvals = np.sort(np.abs(eigvals))[::-1]
                spectra.append(eigvals)
            if n_models > 1:
                dists = pdist(spectra, metric="euclidean")
                spread = float(np.mean(dists))
            else:
                spread = 0.0
        else:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")

        spreads.append(spread)

    return np.array(spreads)
