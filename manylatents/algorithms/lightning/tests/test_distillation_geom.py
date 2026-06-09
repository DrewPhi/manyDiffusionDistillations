"""Minibatch diffop-MSE (M2) integration into Distillation.

Increment 4a of docs/plans/logit-kl-control.md: the resident teacher is forwarded
on the live training minibatch (output_hidden_states=True); a chosen hidden layer
is masked-mean-pooled over tokens for both student and teacher; the geom term is
_diffop_mse_loss between the two B×D pooled activations. geom_weight=0 (default)
must be byte-identical to the task/alignment path (regression guard). Shares the
resident-teacher plumbing with the logit-KL control.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from manylatents.algorithms.lightning.distillation import Distillation
from manylatents.lightning.activation_snapshot import ActivationSnapshot


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 100, hidden: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lin = nn.Linear(hidden, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False):
        h = self.lin(self.embed(input_ids))
        x = self.ln(h)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        hs = (h, x) if output_hidden_states else None
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hs)


def _snap(n=4, hidden=8):
    return ActivationSnapshot(
        input_ids=torch.randint(0, 100, (n, 6)), attention_mask=torch.ones(n, 6, dtype=torch.long),
        sample_ids=list(range(n)), activations={"layers.1": torch.randn(n, hidden)}, reduction="mean")


def _batch(b=6, t=6, v=100):
    ids = torch.randint(0, v, (b, t))
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def _mod(**kw):
    return Distillation(datamodule=None, student=_TinyLM(), activation_snapshot=_snap(),
                        layer_pairs=[], optimizer={"learning_rate": 1e-3}, alignment_weight=0.0, **kw)


def _frozen_teacher():
    t = _TinyLM()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def test_geom_default_off_is_task_only():
    torch.manual_seed(0)
    mod = _mod()
    batch = _batch()
    total = mod.training_step(batch, 0)
    task = mod.student(**batch).loss
    assert torch.allclose(total, task)   # no geom term when geom_weight=0


def test_geom_weight_requires_teacher():
    with pytest.raises(ValueError):
        _mod(geom_weight=1.0)             # no teacher provided


def test_geom_branch_increases_total():
    mod = _mod(geom_weight=1.0, teacher=_frozen_teacher())
    batch = _batch()
    total = mod.training_step(batch, 0)
    task = float(mod.student(**batch).loss)
    assert float(total) > task           # positive geom term added (teacher != student)
    assert total.requires_grad


def test_geom_grad_flows_to_student():
    mod = _mod(geom_weight=1.0, teacher=_frozen_teacher())
    batch = _batch()
    total = mod.training_step(batch, 0)
    total.backward()
    grads = [p.grad for p in mod.student.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_from_spec_reads_geom_block():
    spec = {"reproducibility": {
        "optimizer": {"learning_rate": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8},
        "alignment": {"batch_size": 16}, "seeds": {"global_seed": 42},
        "training": {"max_steps": 100,
                     "geom_distillation": {"weight": 0.5, "sigma_scale": 0.2,
                                           "student_layer": -2, "teacher_layer": -2}}}}
    mod = Distillation.from_spec(spec, student=_TinyLM(), activation_snapshot=_snap(),
                                 layer_pairs=[], datamodule=None, teacher=_frozen_teacher())
    assert mod.geom_weight == 0.5 and mod.geom_sigma_scale == 0.2


def test_from_spec_geom_absent_defaults_off():
    spec = {"reproducibility": {
        "optimizer": {"learning_rate": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8},
        "alignment": {"batch_size": 16}, "seeds": {"global_seed": 42}, "training": {"max_steps": 100}}}
    mod = Distillation.from_spec(spec, student=_TinyLM(), activation_snapshot=_snap(),
                                 layer_pairs=[], datamodule=None)
    assert mod.geom_weight == 0.0


def test_resident_teacher_not_in_state_dict_geom():
    mod = _mod(geom_weight=1.0, teacher=_frozen_teacher())
    keys = list(mod.state_dict().keys())
    assert not any("teacher" in k for k in keys)
    assert any(k.startswith("student.") for k in keys)
