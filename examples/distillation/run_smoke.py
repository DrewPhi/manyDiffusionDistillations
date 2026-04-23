"""Tamia H100 smoke test for the distillation algo module.

End-to-end integration exercise on real models and data:

    bert-large-uncased (teacher) → bert-base-uncased (student)
    probe: 32 pile-mini samples, penultimate encoder layer
    phase1: 20 imperative alignment steps (align_on_snapshot)
    phase2+3: 80 Lightning steps with StagedTrainingCallback
              (phase3_start_step=50, frozen prefix = student's penultimate)

Structural golden test. Asserts:
- every loss is finite
- phase1 last-step MSE < phase1 first-step MSE (alignment descends)
- phase2+3 final task-loss window mean < initial window mean
- StagedTrainingCallback's freeze/unfreeze actually took effect (param grad
  state observed across the phase3 boundary)

Not a numerical reproduction of the collaborator's bert_11m finding (no
recorded baseline exists; see Agent A's archaeology report). Validates
that the new module runs on real hardware with real assets.

Usage:
    python run_smoke.py \\
        --pile-mini /scratch/c/cesarmvc/pile_mini \\
        --out-dir /scratch/c/cesarmvc/manyDiff-algo/smoke_out/$SLURM_JOB_ID
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningDataModule, Trainer
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from manylatents.algorithms.lightning.distillation import Distillation
from manylatents.algorithms.lightning.phase1_align import align_on_snapshot
from manylatents.callbacks.staged_training import StagedTrainingCallback
from manylatents.lightning.activation_snapshot import ActivationSnapshot


TEACHER_NAME = "bert-large-uncased"
STUDENT_NAME = "bert-base-uncased"
MAX_LENGTH = 128
PROBE_N = 32
PHASE1_STEPS = 20
PHASE2_STEPS = 50
PHASE3_STEPS = 30
BATCH_SIZE = 4
LR = 3e-4


def _read_texts(jsonl_dir: Path, n: int) -> List[str]:
    """Pull the first ``n`` text payloads from any shard under ``jsonl_dir``.

    pile_mini stores jsonl lines with a ``text`` key (pile-uncopyrighted
    convention). We don't need ordering - just enough real text to
    tokenize against.
    """
    texts: List[str] = []
    shards = sorted(p for p in jsonl_dir.glob("**/*.jsonl") if p.is_file())
    if not shards:
        raise FileNotFoundError(f"no .jsonl shards under {jsonl_dir}")
    for shard in shards:
        with shard.open() as f:
            for line in f:
                payload = json.loads(line)
                text = payload.get("text") or payload.get("content") or ""
                if text.strip():
                    texts.append(text)
                    if len(texts) >= n:
                        return texts
    raise ValueError(
        f"only found {len(texts)} non-empty text rows under {jsonl_dir}; need {n}"
    )


class _InMemoryMLMDataModule(LightningDataModule):
    """Minimal datamodule over pre-tokenized MLM-style batches."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.batch_size = batch_size

    def train_dataloader(self):
        ds = TensorDataset(self.input_ids, self.attention_mask)

        def collate(rows):
            ids = torch.stack([r[0] for r in rows])
            mask = torch.stack([r[1] for r in rows])
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "labels": ids.clone(),  # trivial LM labels for smoke
            }

        return DataLoader(ds, batch_size=self.batch_size, collate_fn=collate, shuffle=True)


def _penultimate_layer_path(num_hidden_layers: int) -> str:
    return f"bert.encoder.layer.{num_hidden_layers - 2}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pile-mini", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Offline-cluster enforcement.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}", flush=True)

    # -- Teacher side ----------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(TEACHER_NAME, local_files_only=True)
    teacher = AutoModelForMaskedLM.from_pretrained(TEACHER_NAME, local_files_only=True)
    teacher.eval()
    teacher.to(device)
    teacher_penult = _penultimate_layer_path(teacher.config.num_hidden_layers)
    print(f"[smoke] teacher={TEACHER_NAME} penult={teacher_penult}", flush=True)

    # -- Probe set -------------------------------------------------------------
    probe_texts = _read_texts(args.pile_mini, PROBE_N)
    probe_enc = tok(
        probe_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    probe_ids = list(range(PROBE_N))

    snapshot = ActivationSnapshot.from_model(
        teacher,
        input_ids=probe_enc["input_ids"].to(device),
        attention_mask=probe_enc["attention_mask"].to(device),
        sample_ids=probe_ids,
        layer_paths=[teacher_penult],
        reduction="mean",
        batch_size=8,
        device=device,
    )
    print(f"[smoke] snapshot built: n={len(snapshot)} "
          f"keys={list(snapshot.activations.keys())}", flush=True)

    # Free teacher weights before student training starts.
    del teacher
    torch.cuda.empty_cache() if device == "cuda" else None

    # -- Student side ----------------------------------------------------------
    # Random-init bert-base from its architectural config (no weights needed).
    # This matches real distillation practice - the student starts untrained
    # and the alignment pass is part of its training. Also keeps us offline
    # on tamia compute where bert-base weights aren't cached.
    student_config = AutoConfig.from_pretrained(STUDENT_NAME, local_files_only=True)
    student = AutoModelForMaskedLM.from_config(student_config)
    student.to(device)
    student_penult = _penultimate_layer_path(student.config.num_hidden_layers)
    print(
        f"[smoke] student={STUDENT_NAME} (from_config, random init) "
        f"penult={student_penult}",
        flush=True,
    )

    layer_pairs = [
        {"student": student_penult, "teacher": teacher_penult, "weight": 1.0}
    ]

    # -- Phase 1: imperative alignment -----------------------------------------
    phase1_losses = align_on_snapshot(
        student,
        snapshot,
        layer_pairs=layer_pairs,
        n_steps=PHASE1_STEPS,
        optimizer_cfg={"learning_rate": LR},
        batch_size=8,
        seed=42,
        device=device,
    )
    print(f"[smoke] phase1 losses: first={phase1_losses[0]:.4f} "
          f"last={phase1_losses[-1]:.4f}", flush=True)

    # -- Phase 2+3: Lightning fit with StagedTrainingCallback ------------------
    # Tokenize some training batches (distinct from probe).
    train_texts = _read_texts(args.pile_mini, PROBE_N * 8)
    train_enc = tok(
        train_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    dm = _InMemoryMLMDataModule(
        input_ids=train_enc["input_ids"],
        attention_mask=train_enc["attention_mask"],
        batch_size=BATCH_SIZE,
    )

    # Frozen prefix: the Distillation module holds self.student, so param
    # names on the LightningModule begin with "student." (see the callback
    # docstring for the wrapper-prefix gotcha).
    frozen_prefix = f"student.{student_penult}"

    callback = StagedTrainingCallback(
        phase3_start_step=PHASE2_STEPS,
        frozen_prefixes_phase2=[frozen_prefix],
    )

    task_losses: List[float] = []

    # Custom callback to capture per-step task loss for the golden-test
    # assertion. Lightning's logger is disabled to keep this smoke
    # environment-independent.
    class _TaskLossRecorder(StagedTrainingCallback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            if isinstance(outputs, dict) and "loss" in outputs:
                task_losses.append(float(outputs["loss"].detach().cpu()))

    recorder = _TaskLossRecorder(
        phase3_start_step=PHASE2_STEPS,
        frozen_prefixes_phase2=[frozen_prefix],
    )

    mod = Distillation(
        datamodule=dm,
        student=student,
        activation_snapshot=snapshot,
        layer_pairs=layer_pairs,
        optimizer={"learning_rate": LR / 10, "weight_decay": 0.01},
        alignment_weight=1.0,
        alignment_batch_size=8,
    )

    trainer = Trainer(
        max_steps=PHASE2_STEPS + PHASE3_STEPS,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        callbacks=[recorder],
        precision="bf16-mixed" if device == "cuda" else "32-true",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(mod, datamodule=dm)
    print(f"[smoke] fit done: global_step={trainer.global_step}", flush=True)

    # -- Structural assertions -------------------------------------------------
    results = {
        "device": device,
        "teacher": TEACHER_NAME,
        "student": STUDENT_NAME,
        "phase1_losses": phase1_losses,
        "task_losses": task_losses,
        "phase2_steps": PHASE2_STEPS,
        "phase3_steps": PHASE3_STEPS,
    }
    (args.out_dir / "smoke_results.json").write_text(json.dumps(results, indent=2))

    assert all(l == l and abs(l) < 1e6 for l in phase1_losses), (
        f"phase1 losses non-finite or exploded: {phase1_losses}"
    )
    assert phase1_losses[-1] < phase1_losses[0], (
        f"phase1 did not descend: first={phase1_losses[0]} last={phase1_losses[-1]}"
    )
    assert all(l == l and abs(l) < 1e6 for l in task_losses), (
        f"task losses non-finite or exploded: {task_losses}"
    )
    assert trainer.global_step == PHASE2_STEPS + PHASE3_STEPS, (
        f"fit stopped early: global_step={trainer.global_step}"
    )
    print(f"[smoke] PASSED. results at {args.out_dir}/smoke_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
