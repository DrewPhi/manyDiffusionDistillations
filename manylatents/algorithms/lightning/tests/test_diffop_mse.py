"""Unit tests for the differentiable minibatch diffusion-operator MSE loss.

Soft fixed-sigma Gaussian (no hard kNN) -> row-normalized B×B operator on each of
student and teacher activations; MSE between the operators. Dim-agnostic (operator
is over samples), backprops to the student, numerically safe at tiny distances.
"""
from __future__ import annotations

import torch

from manylatents.algorithms.lightning.distillation import _diffop_mse_loss


def test_zero_when_same_acts():
    X = torch.randn(16, 8)
    loss = _diffop_mse_loss(X, X.clone(), sigma_scale=0.15)
    assert loss.ndim == 0
    assert abs(float(loss)) < 1e-10


def test_positive_when_different():
    s = torch.randn(16, 8)
    t = torch.randn(16, 8)
    loss = _diffop_mse_loss(s, t, sigma_scale=0.15)
    assert float(loss) > 0.0


def test_dim_agnostic():
    # student 1024-dim, teacher 4096-dim, same B -> finite scalar (operator is B×B)
    s = torch.randn(24, 1024)
    t = torch.randn(24, 4096)
    loss = _diffop_mse_loss(s, t, sigma_scale=0.15)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_grad_flows_to_student():
    s = torch.randn(16, 8, requires_grad=True)
    t = torch.randn(16, 8)
    loss = _diffop_mse_loss(s, t, sigma_scale=0.15)
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert float(s.grad.abs().sum()) > 0.0


def test_teacher_detached():
    # teacher acts requiring grad must not receive gradient (built under no_grad)
    s = torch.randn(16, 8, requires_grad=True)
    t = torch.randn(16, 8, requires_grad=True)
    loss = _diffop_mse_loss(s, t, sigma_scale=0.15)
    loss.backward()
    assert t.grad is None


def test_no_nan_on_degenerate_input():
    # all points identical -> all pairwise distances 0 -> sigma floor must prevent NaN
    X = torch.ones(12, 8)
    loss = _diffop_mse_loss(X, X.clone(), sigma_scale=0.15, sigma_floor=1e-6)
    assert torch.isfinite(loss)


def test_rows_are_stochastic():
    # internal sanity: the operator rows should sum to ~1 (row-normalized)
    from manylatents.algorithms.lightning.distillation import _soft_diffop
    P = _soft_diffop(torch.randn(20, 8), sigma_scale=0.2)
    rowsums = P.sum(dim=1)
    assert torch.allclose(rowsums, torch.ones(20), atol=1e-5)
