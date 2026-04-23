"""Unit tests for Distillation LightningModule (task-only path)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from manylatents.algorithms.lightning.distillation import Distillation
from manylatents.lightning.activation_snapshot import ActivationSnapshot


# ---- Test fixtures -----------------------------------------------------------


class _TinyLMStudent(nn.Module):
    """HF-shaped tiny student: returns an object with .loss.

    Used to avoid HF downloads in unit tests. Shape mirrors an MLM model:
    ``forward(input_ids, attention_mask, labels=None) -> (loss, logits)``.
    """

    def __init__(self, vocab: int = 100, hidden: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=True) for _ in range(n_layers)]
        )
        # Include a LayerNorm so weight-decay partitioning has something to find.
        self.final_ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def _make_snapshot(n: int = 4, hidden: int = 8) -> ActivationSnapshot:
    torch.manual_seed(0)
    return ActivationSnapshot(
        input_ids=torch.randint(0, 100, (n, 6), dtype=torch.long),
        attention_mask=torch.ones(n, 6, dtype=torch.long),
        sample_ids=list(range(n)),
        activations={"layers.1": torch.randn(n, hidden)},
        reduction="mean",
    )


def _make_task_batch(batch_size: int = 4, seq_len: int = 6, vocab: int = 100):
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


# ---- Tests -------------------------------------------------------------------


def test_construct_with_zero_alignment_weight() -> None:
    student = _TinyLMStudent()
    snap = _make_snapshot()
    mod = Distillation(
        datamodule=None,
        student=student,
        activation_snapshot=snap,
        layer_pairs=[],  # empty is fine when alignment_weight=0
        optimizer={"learning_rate": 1e-3},
        alignment_weight=0.0,
    )
    assert mod.alignment_weight == 0.0
    assert mod.student is student
    assert mod.snapshot is snap


def test_training_step_task_only_finite() -> None:
    student = _TinyLMStudent()
    snap = _make_snapshot()
    mod = Distillation(
        datamodule=None,
        student=student,
        activation_snapshot=snap,
        layer_pairs=[],
        optimizer={"learning_rate": 1e-3},
        alignment_weight=0.0,
    )
    batch = _make_task_batch()
    loss = mod.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss).item()
    assert loss.requires_grad


def test_training_step_returns_task_loss_when_alignment_zero() -> None:
    """When alignment_weight=0, training_step must equal the task loss exactly
    (no alignment term added)."""
    student = _TinyLMStudent()
    snap = _make_snapshot()
    mod = Distillation(
        datamodule=None,
        student=student,
        activation_snapshot=snap,
        layer_pairs=[],
        optimizer={"learning_rate": 1e-3},
        alignment_weight=0.0,
    )
    batch = _make_task_batch()
    total = mod.training_step(batch, batch_idx=0)
    with torch.no_grad():
        raw_task = mod(**batch).loss
    assert torch.allclose(total.detach(), raw_task)


def test_configure_optimizers_two_param_groups_bias_ln_no_decay() -> None:
    student = _TinyLMStudent()
    snap = _make_snapshot()
    mod = Distillation(
        datamodule=None,
        student=student,
        activation_snapshot=snap,
        layer_pairs=[],
        optimizer={"learning_rate": 1e-3, "weight_decay": 0.01},
        alignment_weight=0.0,
    )
    optimizer = mod.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.01
    assert no_decay_group["weight_decay"] == 0.0

    # Build a name lookup from parameter object identity so we can check which
    # params ended up where.
    id_to_name = {id(p): n for n, p in student.named_parameters()}
    no_decay_names = [id_to_name[id(p)] for p in no_decay_group["params"]]
    decay_names = [id_to_name[id(p)] for p in decay_group["params"]]

    # Every bias and LayerNorm weight should be in no_decay.
    for name in decay_names:
        assert not name.endswith(".bias"), f"{name} should be in no_decay"
    assert any(n.endswith(".bias") for n in no_decay_names)
    # The final_ln (nn.LayerNorm) parameters must be in no_decay.
    norm_param_ids = {
        id(p) for p in student.final_ln.parameters()
    }
    no_decay_param_ids = {id(p) for p in no_decay_group["params"]}
    assert norm_param_ids.issubset(no_decay_param_ids), (
        "LayerNorm params must be in no_decay group"
    )


def test_configure_optimizers_filters_requires_grad_false() -> None:
    """Frozen params must not appear in optimizer — this is what makes
    StagedTrainingCallback's freeze semantic actually freeze."""
    student = _TinyLMStudent()
    # Freeze layer 0
    for p in student.layers[0].parameters():
        p.requires_grad = False

    snap = _make_snapshot()
    mod = Distillation(
        datamodule=None,
        student=student,
        activation_snapshot=snap,
        layer_pairs=[],
        optimizer={"learning_rate": 1e-3},
        alignment_weight=0.0,
    )
    optimizer = mod.configure_optimizers()
    all_opt_params = [p for g in optimizer.param_groups for p in g["params"]]
    frozen_ids = {id(p) for p in student.layers[0].parameters()}
    for p in all_opt_params:
        assert id(p) not in frozen_ids, "frozen param leaked into optimizer"


def test_layer_pairs_empty_with_nonzero_alignment_raises() -> None:
    student = _TinyLMStudent()
    snap = _make_snapshot()
    with pytest.raises(ValueError, match=r"layer_pairs must be non-empty"):
        Distillation(
            datamodule=None,
            student=student,
            activation_snapshot=snap,
            layer_pairs=[],
            optimizer={"learning_rate": 1e-3},
            alignment_weight=1.0,
        )


def test_layer_pairs_teacher_missing_from_snapshot_raises() -> None:
    student = _TinyLMStudent()
    snap = _make_snapshot()  # keys: {"layers.1"}
    with pytest.raises(ValueError, match=r"not in snapshot\.activations"):
        Distillation(
            datamodule=None,
            student=student,
            activation_snapshot=snap,
            layer_pairs=[{"student": "layers.0", "teacher": "layers.99", "weight": 1.0}],
            optimizer={"learning_rate": 1e-3},
            alignment_weight=1.0,
        )


def test_layer_pairs_invalid_student_path_raises() -> None:
    student = _TinyLMStudent()
    snap = _make_snapshot()
    with pytest.raises(ValueError, match=r"does not\s+resolve"):
        Distillation(
            datamodule=None,
            student=student,
            activation_snapshot=snap,
            layer_pairs=[
                {"student": "nonexistent.path", "teacher": "layers.1", "weight": 1.0}
            ],
            optimizer={"learning_rate": 1e-3},
            alignment_weight=1.0,
        )


def test_snapshot_tensors_registered_as_buffers() -> None:
    student = _TinyLMStudent()
    snap = _make_snapshot()
    mod = Distillation(
        datamodule=None,
        student=student,
        activation_snapshot=snap,
        layer_pairs=[],
        optimizer={"learning_rate": 1e-3},
        alignment_weight=0.0,
    )
    # input_ids + attention_mask + one-per-layer activation target.
    buffer_names = dict(mod.named_buffers())
    assert "_probe_input_ids" in buffer_names
    assert "_probe_attention_mask" in buffer_names
    # Layer "layers.1" → sanitized buffer name.
    assert any("target" in n and "layers_1" in n for n in buffer_names)


def test_snapshot_buffers_move_with_to() -> None:
    """.to('meta') should move snapshot tensors just like any other buffer."""
    student = _TinyLMStudent()
    snap = _make_snapshot()
    mod = Distillation(
        datamodule=None,
        student=student,
        activation_snapshot=snap,
        layer_pairs=[],
        optimizer={"learning_rate": 1e-3},
        alignment_weight=0.0,
    )
    mod_meta = mod.to("meta")
    assert mod_meta._probe_input_ids.device.type == "meta"
    assert mod_meta._probe_attention_mask.device.type == "meta"


def test_snapshot_property_returns_input() -> None:
    student = _TinyLMStudent()
    snap = _make_snapshot()
    mod = Distillation(
        datamodule=None,
        student=student,
        activation_snapshot=snap,
        layer_pairs=[],
        optimizer={"learning_rate": 1e-3},
        alignment_weight=0.0,
    )
    assert mod.snapshot is snap
    # Reduction must be readable from the snapshot, not duplicated on the module.
    assert mod.snapshot.reduction == "mean"


def test_alignment_path_raises_not_implemented_in_step_6() -> None:
    """The alignment regularizer lands in the next commit. Constructing with
    alignment_weight>0 is fine; calling training_step with it currently raises
    a clear NotImplementedError. This test locks that contract."""
    student = _TinyLMStudent()
    snap = _make_snapshot()
    mod = Distillation(
        datamodule=None,
        student=student,
        activation_snapshot=snap,
        layer_pairs=[{"student": "layers.0", "teacher": "layers.1", "weight": 1.0}],
        optimizer={"learning_rate": 1e-3},
        alignment_weight=1.0,
    )
    batch = _make_task_batch()
    with pytest.raises(NotImplementedError, match=r"alignment path lands"):
        mod.training_step(batch, batch_idx=0)


def test_setup_seeds_deterministically() -> None:
    student = _TinyLMStudent()
    snap = _make_snapshot()
    mod = Distillation(
        datamodule=None,
        student=student,
        activation_snapshot=snap,
        layer_pairs=[],
        optimizer={"learning_rate": 1e-3},
        alignment_weight=0.0,
        init_seed=1234,
    )
    mod.setup()
    a = torch.randn(3)
    mod.setup()
    b = torch.randn(3)
    assert torch.equal(a, b), "setup must reseed so torch.randn is deterministic"
