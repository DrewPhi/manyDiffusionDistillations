"""Unit tests for the logit-KL distillation loss helper (_logit_kl_loss).

Temperature-scaled KL(teacher || student) over non-pad positions — the standard
Hinton KD term (T^2 scaling), used by the logit-KL control.
"""
from __future__ import annotations

import math

import torch

from manylatents.algorithms.lightning.distillation import _logit_kl_loss


def test_zero_when_identical():
    logits = torch.randn(2, 5, 7)
    mask = torch.ones(2, 5)
    loss = _logit_kl_loss(logits, logits.clone(), mask, temperature=1.0)
    assert loss.ndim == 0
    assert float(loss) == 0.0 or abs(float(loss)) < 1e-6


def test_known_value_T1():
    # teacher logits [0,0] -> p=[.5,.5]; student [2,0] -> p=[.8808,.1192]
    # KL(p_t||p_s) = .5*ln(.5/.8808)+.5*ln(.5/.1192) = 0.43378; T=1 -> x1
    teacher = torch.tensor([[[0.0, 0.0]]])
    student = torch.tensor([[[2.0, 0.0]]])
    mask = torch.ones(1, 1)
    loss = _logit_kl_loss(student, teacher, mask, temperature=1.0)
    assert abs(float(loss) - 0.43378) < 1e-3, float(loss)


def test_known_value_T2():
    # same logits, T=2: p_t=[.5,.5]; p_s=softmax([1,0])=[.7311,.2689]
    # KL=.5*ln(.5/.7311)+.5*ln(.5/.2689)=0.12017; x T^2=4 -> 0.48068
    teacher = torch.tensor([[[0.0, 0.0]]])
    student = torch.tensor([[[2.0, 0.0]]])
    mask = torch.ones(1, 1)
    loss = _logit_kl_loss(student, teacher, mask, temperature=2.0)
    assert abs(float(loss) - 0.48068) < 1e-3, float(loss)


def test_respects_mask():
    teacher = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    student = torch.tensor([[[2.0, 0.0], [5.0, -5.0]]])
    # position 1 is padding -> must not affect the loss
    mask = torch.tensor([[1.0, 0.0]])
    loss = _logit_kl_loss(student, teacher, mask, temperature=1.0)
    # equals the single-position known value
    assert abs(float(loss) - 0.43378) < 1e-3, float(loss)


def test_averages_over_valid_positions():
    # two identical valid positions -> average == single-position value
    teacher = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    student = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
    mask = torch.ones(1, 2)
    loss = _logit_kl_loss(student, teacher, mask, temperature=1.0)
    assert abs(float(loss) - 0.43378) < 1e-3, float(loss)


def test_scalar_shape_from_BTV():
    loss = _logit_kl_loss(torch.randn(3, 4, 9), torch.randn(3, 4, 9),
                          torch.ones(3, 4), temperature=1.5)
    assert loss.ndim == 0 and torch.isfinite(loss)
