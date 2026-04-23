"""Frozen per-layer activation snapshot.

An ActivationSnapshot pairs a fixed set of tokenized inputs with pre-computed,
already-pooled activations at named layers. Used as a reference artifact for
alignment-style training losses (see
``manylatents.algorithms.lightning.distillation``), probing, and analysis that
requires stable target activations against a known input set.

The snapshot declares, via the ``reduction`` field, how its per-layer tensors
were pooled (``mean``, ``cls``, ``last_token``, ``first_token``, ``none``).
Consumers that perform a fresh student forward for a regularizer use this field
to drive matching pooling of the live activations; they do not re-state
producer-side recipes.

Contract for ``sample_ids``: when a snapshot is built against a probe split
from a consumer's ``LightningDataModule`` (e.g. ``TextDataModule``), its
``sample_ids`` must be the same identifiers the datamodule emits at training
time (e.g. ``batch["probe_ids"]``). Mismatched ID spaces cause silent
target-lookup errors. This is a consumer-side convention; we enforce uniqueness
on the snapshot side and document the rest.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor

from manylatents.lightning.hooks import VALID_REDUCE

__all__ = ["ActivationSnapshot", "SNAPSHOT_SCHEMA_VERSION"]

# Bump when the on-disk format changes incompatibly. `load` rejects unknown
# versions rather than silently mis-deserializing. Migration paths (if any) go
# inside `load` keyed on the version field.
SNAPSHOT_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class ActivationSnapshot:
    """Fixed batch of tokenized inputs with pre-computed, already-pooled
    activations at named layers.

    Attributes:
        input_ids: (N, L) int64 token IDs.
        attention_mask: (N, L) int64 attention mask.
        sample_ids: length-N list of unique identifiers (see contract above).
        activations: dict mapping layer path (str) to a (N, hidden) tensor of
            activations already pooled with ``reduction``.
        reduction: the single pooling strategy applied to every tensor in
            ``activations``. One of the entries in
            ``manylatents.lightning.hooks.VALID_REDUCE``.
    """

    input_ids: Tensor
    attention_mask: Tensor
    sample_ids: List[int]
    activations: Dict[str, Tensor]
    reduction: str

    def __post_init__(self) -> None:
        n = self.input_ids.shape[0]

        if self.attention_mask.shape[0] != n:
            raise ValueError(
                f"attention_mask.shape[0] ({self.attention_mask.shape[0]}) "
                f"must equal input_ids.shape[0] ({n})"
            )
        if len(self.sample_ids) != n:
            raise ValueError(
                f"len(sample_ids) ({len(self.sample_ids)}) must equal "
                f"input_ids.shape[0] ({n})"
            )
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")

        if self.reduction not in VALID_REDUCE:
            raise ValueError(
                f"reduction must be one of {VALID_REDUCE}, got {self.reduction!r}"
            )

        input_device = self.input_ids.device
        if self.attention_mask.device != input_device:
            raise ValueError(
                f"attention_mask device ({self.attention_mask.device}) must "
                f"equal input_ids device ({input_device})"
            )

        for layer, acts in self.activations.items():
            if acts.shape[0] != n:
                raise ValueError(
                    f"activations[{layer!r}].shape[0] ({acts.shape[0]}) must "
                    f"equal input_ids.shape[0] ({n})"
                )
            if acts.device != input_device:
                raise ValueError(
                    f"activations[{layer!r}] device ({acts.device}) must "
                    f"equal input_ids device ({input_device})"
                )

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def save(self, path: Path | str) -> None:
        """Serialize to disk as a single ``torch.save`` blob.

        The on-disk format is a versioned dict. ``load`` validates the version
        and reconstructs the dataclass (re-triggering ``__post_init__``).
        """
        torch.save(
            {
                "_version": SNAPSHOT_SCHEMA_VERSION,
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask,
                "sample_ids": list(self.sample_ids),
                "activations": dict(self.activations),
                "reduction": self.reduction,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: Path | str) -> "ActivationSnapshot":
        """Load a snapshot previously written by ``save``.

        Raises:
            ValueError: if the file's ``_version`` does not match
                ``SNAPSHOT_SCHEMA_VERSION`` or the dict is missing required keys.
        """
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(blob, dict):
            raise ValueError(
                f"expected a dict at {path!s}, got {type(blob).__name__}"
            )
        version = blob.get("_version")
        if version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unknown ActivationSnapshot schema _version={version!r} at "
                f"{path!s}; this build expects {SNAPSHOT_SCHEMA_VERSION}"
            )
        required = {"input_ids", "attention_mask", "sample_ids", "activations", "reduction"}
        missing = required - blob.keys()
        if missing:
            raise ValueError(
                f"malformed snapshot at {path!s}: missing keys {sorted(missing)}"
            )
        return cls(
            input_ids=blob["input_ids"],
            attention_mask=blob["attention_mask"],
            sample_ids=list(blob["sample_ids"]),
            activations=dict(blob["activations"]),
            reduction=blob["reduction"],
        )
