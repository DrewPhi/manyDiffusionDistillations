"""Unit tests for ActivationSnapshot invariants and save/load."""
from __future__ import annotations

import pytest
import torch

from manylatents.lightning.activation_snapshot import (
    ActivationSnapshot,
    SNAPSHOT_SCHEMA_VERSION,
)


def _make_valid_fields(n: int = 4, seq_len: int = 8, hidden: int = 16):
    return {
        "input_ids": torch.zeros(n, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        "sample_ids": list(range(n)),
        "activations": {"encoder.layer.0": torch.zeros(n, hidden)},
        "reduction": "mean",
    }


def test_construct_happy_path() -> None:
    snap = ActivationSnapshot(**_make_valid_fields())
    assert snap.reduction == "mean"
    assert "encoder.layer.0" in snap.activations


def test_post_init_rejects_shape_mismatch_attention_mask() -> None:
    fields = _make_valid_fields(n=4)
    fields["attention_mask"] = torch.ones(3, 8, dtype=torch.long)
    with pytest.raises(ValueError, match=r"attention_mask\.shape\[0\]"):
        ActivationSnapshot(**fields)


def test_post_init_rejects_shape_mismatch_sample_ids() -> None:
    fields = _make_valid_fields(n=4)
    fields["sample_ids"] = [0, 1, 2]
    with pytest.raises(ValueError, match=r"len\(sample_ids\)"):
        ActivationSnapshot(**fields)


def test_post_init_rejects_duplicate_sample_ids() -> None:
    fields = _make_valid_fields(n=4)
    fields["sample_ids"] = [0, 1, 1, 2]
    with pytest.raises(ValueError, match=r"unique"):
        ActivationSnapshot(**fields)


def test_post_init_rejects_wrong_activation_first_dim() -> None:
    fields = _make_valid_fields(n=4, hidden=16)
    fields["activations"] = {"encoder.layer.0": torch.zeros(3, 16)}
    with pytest.raises(ValueError, match=r"activations.*shape\[0\]"):
        ActivationSnapshot(**fields)


def test_post_init_rejects_unknown_reduction() -> None:
    fields = _make_valid_fields()
    fields["reduction"] = "gibberish"
    with pytest.raises(ValueError, match=r"reduction must be one of"):
        ActivationSnapshot(**fields)


def test_post_init_accepts_all_valid_reductions() -> None:
    fields = _make_valid_fields()
    for reduction in ("mean", "last_token", "cls", "first_token", "none"):
        fields_copy = {**fields, "reduction": reduction}
        ActivationSnapshot(**fields_copy)


def test_post_init_rejects_device_mismatch_attention_mask() -> None:
    fields = _make_valid_fields()
    # meta device is always available, avoids CUDA requirement
    fields["attention_mask"] = fields["attention_mask"].to("meta")
    with pytest.raises(ValueError, match=r"attention_mask device"):
        ActivationSnapshot(**fields)


def test_post_init_rejects_device_mismatch_activations() -> None:
    fields = _make_valid_fields()
    fields["activations"] = {
        "encoder.layer.0": fields["activations"]["encoder.layer.0"].to("meta")
    }
    with pytest.raises(ValueError, match=r"activations.*device"):
        ActivationSnapshot(**fields)


def test_len_returns_n_samples() -> None:
    snap = ActivationSnapshot(**_make_valid_fields(n=7))
    assert len(snap) == 7


def test_multiple_layers_all_validated() -> None:
    """Every activation tensor must pass shape+device checks, not just the first."""
    fields = _make_valid_fields(n=4, hidden=16)
    fields["activations"] = {
        "encoder.layer.0": torch.zeros(4, 16),
        "encoder.layer.11": torch.zeros(3, 16),  # wrong
    }
    with pytest.raises(ValueError, match=r"encoder\.layer\.11"):
        ActivationSnapshot(**fields)


def test_frozen_cannot_mutate() -> None:
    snap = ActivationSnapshot(**_make_valid_fields())
    with pytest.raises((AttributeError, Exception)):
        snap.reduction = "cls"  # type: ignore[misc]


def test_save_load_roundtrip(tmp_path) -> None:
    fields = _make_valid_fields(n=5, hidden=8)
    fields["input_ids"] = torch.randint(0, 100, (5, 8), dtype=torch.long)
    fields["sample_ids"] = [10, 20, 30, 40, 50]
    fields["activations"] = {
        "encoder.layer.0": torch.randn(5, 8),
        "encoder.layer.11": torch.randn(5, 8),
    }
    snap = ActivationSnapshot(**fields)

    path = tmp_path / "snap.pt"
    snap.save(path)
    loaded = ActivationSnapshot.load(path)

    assert torch.equal(loaded.input_ids, snap.input_ids)
    assert torch.equal(loaded.attention_mask, snap.attention_mask)
    assert loaded.sample_ids == snap.sample_ids
    assert loaded.reduction == snap.reduction
    assert set(loaded.activations.keys()) == set(snap.activations.keys())
    for k in snap.activations:
        assert torch.equal(loaded.activations[k], snap.activations[k])


def test_save_accepts_str_path(tmp_path) -> None:
    snap = ActivationSnapshot(**_make_valid_fields())
    path_str = str(tmp_path / "snap.pt")
    snap.save(path_str)
    loaded = ActivationSnapshot.load(path_str)
    assert len(loaded) == len(snap)


def test_load_rejects_unknown_version(tmp_path) -> None:
    path = tmp_path / "future.pt"
    snap = ActivationSnapshot(**_make_valid_fields())
    blob = {
        "_version": SNAPSHOT_SCHEMA_VERSION + 99,
        "input_ids": snap.input_ids,
        "attention_mask": snap.attention_mask,
        "sample_ids": snap.sample_ids,
        "activations": snap.activations,
        "reduction": snap.reduction,
    }
    torch.save(blob, str(path))
    with pytest.raises(ValueError, match=r"unknown ActivationSnapshot schema _version"):
        ActivationSnapshot.load(path)


def test_load_rejects_missing_keys(tmp_path) -> None:
    path = tmp_path / "malformed.pt"
    torch.save({"_version": SNAPSHOT_SCHEMA_VERSION, "input_ids": torch.zeros(2, 4)}, str(path))
    with pytest.raises(ValueError, match=r"missing keys"):
        ActivationSnapshot.load(path)


def test_load_rejects_non_dict(tmp_path) -> None:
    path = tmp_path / "weird.pt"
    torch.save(torch.zeros(3), str(path))
    with pytest.raises(ValueError, match=r"expected a dict"):
        ActivationSnapshot.load(path)


def test_load_validates_on_read(tmp_path) -> None:
    """A dict with valid schema but broken invariants should raise __post_init__."""
    path = tmp_path / "broken.pt"
    blob = {
        "_version": SNAPSHOT_SCHEMA_VERSION,
        "input_ids": torch.zeros(4, 8, dtype=torch.long),
        "attention_mask": torch.ones(4, 8, dtype=torch.long),
        "sample_ids": [0, 1, 1, 2],  # duplicate — __post_init__ should reject
        "activations": {"encoder.layer.0": torch.zeros(4, 16)},
        "reduction": "mean",
    }
    torch.save(blob, str(path))
    with pytest.raises(ValueError, match=r"unique"):
        ActivationSnapshot.load(path)
