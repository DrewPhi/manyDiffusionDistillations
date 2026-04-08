"""Distillation sweep stage for LM + alignment training."""
from __future__ import annotations

import csv
import gc
import itertools
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW

try:
    import wandb
    wandb.init
except (ImportError, AttributeError):
    wandb = None

from manylatents.data.text import TextDataModule
from manylatents.lightning.hf_trainer import HFTrainerConfig, HFTrainerModule
from manylatents.lightning.hooks import ActivationExtractor, LayerSpec
from manylatents.pipeline.family_resolution import (
    resolve_layer_alias,
    resolve_layer_aliases,
    validate_family_layers,
)
from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult


class DistillationSweepStage(PipelineStage):
    """Run Cartesian sweeps for student distillation with LM + alignment loss."""

    RESULT_SCHEMA_VERSION = "distill_sweep_row_v4"

    @staticmethod
    def _wandb_log(metrics: Dict[str, float], step: int | None = None) -> None:
        if wandb is None or wandb.run is None:
            return
        wandb.log(metrics, step=step)

    @staticmethod
    def _sanitize_layer_name(layer: str) -> str:
        """Match the layer-name normalization used by probe/PHATE artifacts."""
        return layer.replace(".", "_").replace("[", "_").replace("]", "_").replace("-", "m")

    @staticmethod
    def _infer_layer_scheme_name(layer_pairs: Sequence[Dict[str, Any]]) -> str:
        if len(layer_pairs) == 1:
            return "penultimate_only"
        if len(layer_pairs) == 2:
            return "second_plus_penultimate"
        return "custom"

    @staticmethod
    def _json_list(values: Sequence[Any]) -> str:
        return json.dumps(list(values), separators=(",", ":"))

    def __init__(
        self,
        stage_name: str,
        teacher_stage: str = "probe_teacher",
        teacher_target_stage: str = "phate_teacher_target",
        teacher_target_key: str = "aligned_targets",
        teacher_probe_ids_key: str = "probe_ids_path",
        target_probe_ids_key: str = "aligned_probe_ids_path",
        target_primary_key: str = "aligned_target_path",
        target_paths_by_layer_key: str = "aligned_target_paths_by_layer",
        target_index: int = 0,
        output_subdir: str | None = None,
        teacher_model_name_or_path: str = "EleutherAI/pythia-1.4b",
        teacher_model_revision: str | None = None,
        student_model_name_or_path: str = "EleutherAI/pythia-70m",
        student_model_revision: str | None = None,
        student_trust_remote_code: bool = False,
        student_model_family: str = "causal_lm",
        student_model_config_name_or_path: str | None = None,
        student_model_config_overrides: Dict[str, Any] | None = None,
        init_from_scratch: bool = True,
        family_name: str | None = None,
        family_architecture: str | None = None,
        alignment_side: str | None = None,
        penultimate_layer_teacher: str = "transformer.h[-2]",
        penultimate_layer_student: str = "transformer.h[-2]",
        teacher_layer_specs: Sequence[str] | None = None,
        student_layer_specs: Sequence[str] | None = None,
        layer_loss_weights: Dict[str, float] | Sequence[float] | None = None,
        penultimate_dim_student: int = 512,
        dataset_name: str = "wikitext",
        dataset_config: str | None = "wikitext-2-raw-v1",
        dataset_path: str | None = None,
        dataset_revision: str | None = None,
        text_field: str = "text",
        probe_ids_path: str | None = None,
        tokenizer_name: str | None = None,
        max_length: int = 128,
        token_budget: int | None = 0,
        train_example_offset: int = 0,
        train_split: str = "train",
        val_split: str = "validation",
        test_split: str | None = "test",
        probe_split: str | None = None,
        exclude_probe_from_train: bool = False,
        precision: str = "bf16",
        global_batch_size: int = 128,
        micro_batch_size: int = 8,
        grad_accum_steps: int = 16,
        max_steps: int | None = None,
        eval_every_n_steps: int = 1000,
        save_every_n_steps: int = 1000,
        save_top_k: int = 1,
        analysis_checkpoint_steps: Sequence[int] | None = None,
        gradient_clip_norm: float = 1.0,
        train_history_every_n_steps: int = 10,
        optimizer: Dict[str, Any] | None = None,
        lr_scheduler: Dict[str, Any] | None = None,
        regularization: Dict[str, Any] | None = None,
        alignment: Dict[str, Any] | None = None,
        staged_training: Dict[str, Any] | None = None,
        seeds: Dict[str, int] | None = None,
        sweep: Dict[str, Sequence[Any]] | None = None,
        sweep_tied_keys: Sequence[Sequence[str] | str] | None = None,
        eval_max_batches: int = 100,
        lm_loss_weight: float = 1.0,
        training_regime: str = "staged",
        student_init_checkpoint_path: str | None = None,
    ):
        super().__init__(stage_name=stage_name)
        self.teacher_stage = teacher_stage
        self.teacher_target_stage = teacher_target_stage
        self.teacher_target_key = teacher_target_key
        self.teacher_probe_ids_key = teacher_probe_ids_key
        self.target_probe_ids_key = target_probe_ids_key
        self.target_primary_key = target_primary_key
        self.target_paths_by_layer_key = target_paths_by_layer_key
        self.target_index = target_index
        self.output_subdir = output_subdir or stage_name

        self.teacher_model_name_or_path = teacher_model_name_or_path
        self.teacher_model_revision = teacher_model_revision
        self.student_model_name_or_path = student_model_name_or_path
        self.student_model_revision = student_model_revision
        self.student_trust_remote_code = bool(student_trust_remote_code)
        self.student_model_family = str(student_model_family)
        self.student_model_config_name_or_path = student_model_config_name_or_path
        self.student_model_config_overrides = dict(student_model_config_overrides or {})
        self.init_from_scratch = init_from_scratch
        self.family_name = family_name
        self.family_architecture = family_architecture
        self.alignment_side = alignment_side
        self.penultimate_layer_teacher = resolve_layer_alias(penultimate_layer_teacher, family_name)
        self.penultimate_layer_student = resolve_layer_alias(penultimate_layer_student, family_name)
        self.teacher_layer_specs = resolve_layer_aliases(
            teacher_layer_specs or [self.penultimate_layer_teacher],
            family_name,
        )
        self.student_layer_specs = resolve_layer_aliases(
            student_layer_specs or [self.penultimate_layer_student],
            family_name,
        )
        if len(self.teacher_layer_specs) != len(self.student_layer_specs):
            raise ValueError(
                "teacher_layer_specs and student_layer_specs must have the same length "
                f"({len(self.teacher_layer_specs)} vs {len(self.student_layer_specs)})"
            )
        if not self.student_layer_specs:
            raise ValueError("student_layer_specs must contain at least one layer path")
        validate_family_layers(
            family_name=self.family_name,
            architecture=self.family_architecture,
            alignment_side=self.alignment_side,
            teacher_layers=self.teacher_layer_specs,
            student_layers=self.student_layer_specs,
        )
        self.layer_loss_weights_cfg = layer_loss_weights
        self.penultimate_dim_student = penultimate_dim_student

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_path = dataset_path
        self.dataset_revision = dataset_revision
        self.text_field = text_field
        self.probe_ids_path = probe_ids_path
        self.tokenizer_name = tokenizer_name or student_model_name_or_path
        self.max_length = max_length
        self.token_budget = token_budget
        self.train_example_offset = int(train_example_offset)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.probe_split = probe_split or val_split
        self.exclude_probe_from_train = exclude_probe_from_train

        self.precision = precision
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_steps = max_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        self.save_top_k = save_top_k
        self.analysis_checkpoint_steps = sorted({int(step) for step in (analysis_checkpoint_steps or []) if int(step) > 0})
        self.analysis_checkpoint_step_set = set(self.analysis_checkpoint_steps)
        self.gradient_clip_norm = gradient_clip_norm
        self.train_history_every_n_steps = train_history_every_n_steps

        self.optimizer_cfg = optimizer or {
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
        }
        self.lr_scheduler_cfg = lr_scheduler or {
            "type": "cosine",
            "warmup_steps": 0,
            "min_lr": 0.0,
        }
        self.regularization_cfg = regularization or {}
        self.alignment_cfg = alignment or {
            "lambda_align": 0.0,
            "lambda_schedule": "constant",
            "lambda_ramp_steps": 0,
            "batch_size": 16,
            "sample_every_n_steps": 1,
            "probe_train_fraction": 0.8,
            "probe_eval_fraction": 0.2,
            "mse_reduction": "mean",
        }
        self.staged_training_cfg = {
            "enabled": False,
            "phase1": {
                "objective": "alignment_only",
                "min_steps": 0,
                "max_steps": 0,
                "eval_every_n_steps": 0,
                "early_stop_patience": 0,
                "early_stop_min_delta": 0.0,
            },
            "phase2": {"objective": "task_only_frozen", "max_steps": 0},
            "phase3": {"objective": "task_only_unfrozen", "max_steps": 0},
            "checkpoint_analysis": {
                "enabled": False,
                "phase1_snapshots": 0,
                "phase2_snapshots": 0,
                "phase3_snapshots": 0,
                "include_phase_end": True,
            },
        }
        if staged_training:
            for key, value in staged_training.items():
                if isinstance(value, dict) and isinstance(self.staged_training_cfg.get(key), dict):
                    merged = dict(self.staged_training_cfg[key])
                    merged.update(value)
                    self.staged_training_cfg[key] = merged
                else:
                    self.staged_training_cfg[key] = value
        self.seeds = seeds or {"global_seed": 42}
        self.sweep = sweep or {}
        self.sweep_tied_keys = [
            [str(v) for v in group] if not isinstance(group, str) else [group]
            for group in (sweep_tied_keys or [])
        ]
        self.eval_max_batches = eval_max_batches
        self.lm_loss_weight = float(lm_loss_weight)
        self.training_regime = str(training_regime)
        self.student_init_checkpoint_path = student_init_checkpoint_path

    def _resolve_stage_seed(self, key: str, combo_seed: int) -> int:
        val = self.seeds.get(key)
        if val is None:
            return int(combo_seed)
        return int(val)

    @staticmethod
    def _to_2d(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.reshape(-1, 1)
        if x.ndim == 2:
            return x
        return x.reshape(x.shape[0], -1)

    @staticmethod
    def _resolve_list_artifact(value: Any) -> List[Path]:
        if value is None:
            return []
        if isinstance(value, str):
            return [Path(value)]
        if isinstance(value, Sequence):
            return [Path(v) for v in value]
        raise TypeError(f"Unsupported artifact type: {type(value)}")

    @staticmethod
    def _expand_sweep_grid(
        sweep: Dict[str, Sequence[Any]],
        tied_key_groups: Sequence[Sequence[str]] | None = None,
    ) -> List[Dict[str, Any]]:
        if not sweep:
            return [{}]

        groups: List[List[str]] = []
        used_keys: set[str] = set()
        for group in tied_key_groups or []:
            normalized = [str(k) for k in group]
            if not normalized:
                continue
            missing = [k for k in normalized if k not in sweep]
            if missing:
                raise KeyError(f"sweep_tied_keys references missing sweep keys: {missing}")
            for key in normalized:
                if key in used_keys:
                    raise ValueError(f"sweep key '{key}' appears in multiple tied groups")
                used_keys.add(key)
            lengths = []
            for key in normalized:
                seq = sweep[key]
                if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
                    raise TypeError(f"sweep['{key}'] must be a sequence of values")
                if len(seq) == 0:
                    raise ValueError(f"sweep['{key}'] is empty")
                lengths.append(len(seq))
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"All tied sweep keys must have the same length for group {normalized}: {lengths}"
                )
            groups.append(normalized)

        for key in sweep.keys():
            if key not in used_keys:
                groups.append([key])

        grouped_values: List[List[Dict[str, Any]]] = []
        for group in groups:
            if len(group) == 1:
                key = group[0]
                seq = sweep[key]
                if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
                    raise TypeError(f"sweep['{key}'] must be a sequence of values")
                if len(seq) == 0:
                    raise ValueError(f"sweep['{key}'] is empty")
                grouped_values.append([{key: value} for value in seq])
                continue

            rows = []
            for idx in range(len(sweep[group[0]])):
                rows.append({key: sweep[key][idx] for key in group})
            grouped_values.append(rows)

        combos: List[Dict[str, Any]] = []
        for parts in itertools.product(*grouped_values):
            merged: Dict[str, Any] = {}
            for part in parts:
                merged.update(part)
            combos.append(merged)
        return combos

    def _resolve_combo(self, combo: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "teacher_model_name_or_path": combo.get("teacher_model", self.teacher_model_name_or_path),
            "teacher_model_revision": combo.get("teacher_model_revision", self.teacher_model_revision),
            "student_model_name_or_path": combo.get("student_model", self.student_model_name_or_path),
            "student_model_revision": combo.get("student_model_revision", self.student_model_revision),
            "student_model_family": combo.get("student_model_family", self.student_model_family),
            "student_model_config_name_or_path": combo.get(
                "student_model_config_name_or_path",
                self.student_model_config_name_or_path,
            ),
            "student_model_config_overrides": dict(
                combo.get("student_model_config_overrides", self.student_model_config_overrides) or {}
            ),
            "seed": int(combo.get("seed", self.seeds.get("global_seed", 42))),
            "lambda_align": float(combo.get("lambda_align", self.alignment_cfg.get("lambda_align", 0.0))),
            "learning_rate": float(combo.get("learning_rate", self.optimizer_cfg.get("learning_rate", 3e-4))),
            "max_length": int(combo.get("max_length", self.max_length)),
            "token_budget": (
                None
                if combo.get("token_budget", self.token_budget) is None
                else int(combo.get("token_budget", self.token_budget))
            ),
            "train_example_offset": int(combo.get("train_example_offset", self.train_example_offset)),
            "lm_loss_weight": float(combo.get("lm_loss_weight", self.lm_loss_weight)),
            "student_init_checkpoint_path": combo.get(
                "student_init_checkpoint_path",
                self.student_init_checkpoint_path,
            ),
        }
        return out

    def _effective_global_batch_size(self) -> int:
        effective = int(self.micro_batch_size) * int(self.grad_accum_steps)
        if int(self.global_batch_size) != effective:
            raise ValueError(
                "Inconsistent batch configuration: "
                f"global_batch_size={self.global_batch_size} but "
                f"micro_batch_size*grad_accum_steps={self.micro_batch_size}*{self.grad_accum_steps}={effective}"
            )
        return effective

    def _compute_max_steps(
        self,
        token_budget: int | None,
        max_length: int,
        train_batches: int | None = None,
    ) -> int:
        if self.max_steps is not None:
            return int(self.max_steps)
        if token_budget is None:
            if train_batches is None:
                raise ValueError("train_batches is required when token_budget is null and max_steps is unset")
            return max(1, int(math.ceil(float(train_batches) / float(self.grad_accum_steps))))
        if token_budget <= 0:
            return 100
        effective_batch = self._effective_global_batch_size()
        tokens_per_step = max(1, int(effective_batch * max_length))
        return max(1, token_budget // tokens_per_step)

    @staticmethod
    def _select_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in batch.items()}

    def _resolve_layer_pairs(self) -> List[Dict[str, Any]]:
        if isinstance(self.layer_loss_weights_cfg, dict):
            weights = [
                float(
                    self.layer_loss_weights_cfg.get(student_layer,
                    self.layer_loss_weights_cfg.get(teacher_layer, 1.0))
                )
                for teacher_layer, student_layer in zip(self.teacher_layer_specs, self.student_layer_specs)
            ]
        elif self.layer_loss_weights_cfg is not None:
            weights = [float(v) for v in self.layer_loss_weights_cfg]
            if len(weights) != len(self.student_layer_specs):
                raise ValueError(
                    "layer_loss_weights sequence must match number of layer specs "
                    f"({len(weights)} vs {len(self.student_layer_specs)})"
                )
        else:
            weights = [1.0] * len(self.student_layer_specs)

        return [
            {
                "teacher_layer": teacher_layer,
                "student_layer": student_layer,
                "weight": float(weight),
            }
            for teacher_layer, student_layer, weight in zip(
                self.teacher_layer_specs,
                self.student_layer_specs,
                weights,
            )
        ]

    def _make_probe_buffers(
        self,
        dm: TextDataModule,
        aligned_targets: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        probe_dataset = dm.probe_dataset
        if probe_dataset is None:
            raise RuntimeError("TextDataModule has no probe_dataset; call setup() first")

        if not aligned_targets:
            raise ValueError("aligned_targets must contain at least one layer target matrix")

        n = min(len(probe_dataset), min(target.shape[0] for target in aligned_targets.values()))
        if n < 2:
            raise ValueError("Need at least 2 probe samples to compute alignment")

        probe_input_ids = []
        probe_attention_mask = []
        for i in range(n):
            item = probe_dataset[i]
            probe_input_ids.append(item["input_ids"])
            probe_attention_mask.append(item["attention_mask"])

        input_ids = torch.stack(probe_input_ids, dim=0)
        attention_mask = torch.stack(probe_attention_mask, dim=0)
        targets = {
            layer: torch.from_numpy(target[:n]).float()
            for layer, target in aligned_targets.items()
        }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
            "n": n,
        }

    @staticmethod
    def _load_probe_ids_from_artifacts(artifacts: Dict[str, Any], key: str) -> List[int]:
        path = artifacts.get(key)
        if isinstance(path, str) and path:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise TypeError(f"Probe IDs payload at '{path}' is not a list")
            return [int(v) for v in payload]
        return []

    @staticmethod
    def _probe_ids_from_datamodule(dm: TextDataModule) -> List[int]:
        if getattr(dm, "probe_source_ids", None):
            return [int(v) for v in dm.probe_source_ids]
        probe_dataset = dm.probe_dataset
        if probe_dataset is not None and hasattr(probe_dataset, "indices"):
            base_dataset = getattr(probe_dataset, "dataset", None)
            if base_dataset is not None and hasattr(base_dataset, "source_id_for_index"):
                return [
                    int(base_dataset.source_id_for_index(int(v)))
                    for v in getattr(probe_dataset, "indices")
                ]
            return [int(v) for v in getattr(probe_dataset, "indices")]
        if probe_dataset is not None:
            if hasattr(probe_dataset, "source_id_for_index"):
                return [
                    int(probe_dataset.source_id_for_index(i))
                    for i in range(len(probe_dataset))
                ]
            return list(range(len(probe_dataset)))
        return []

    def _materialize_targets_for_datamodule(
        self,
        dm: TextDataModule,
        aligned_targets: Dict[str, np.ndarray],
        expected_probe_ids: List[int],
    ) -> Dict[str, np.ndarray]:
        dm_probe_ids = self._probe_ids_from_datamodule(dm)
        if not dm_probe_ids:
            return aligned_targets

        n_targets = {layer: len(target) for layer, target in aligned_targets.items()}
        if len(set(n_targets.values())) != 1:
            raise ValueError(f"All aligned target matrices must have the same row count: {n_targets}")
        target_rows = next(iter(n_targets.values()))
        if len(expected_probe_ids) < target_rows:
            expected_probe_ids = expected_probe_ids[:target_rows]
        if len(expected_probe_ids) != target_rows:
            raise ValueError(
                "Aligned targets row count does not match expected probe IDs length "
                f"({target_rows} vs {len(expected_probe_ids)})"
            )

        id_to_idx = {pid: i for i, pid in enumerate(expected_probe_ids)}
        reordered: Dict[str, np.ndarray] = {}
        for layer, target in aligned_targets.items():
            reordered_rows = []
            for pid in dm_probe_ids:
                if pid not in id_to_idx:
                    raise ValueError(
                        f"Datamodule probe ID {pid} not present in aligned-target probe ID contract"
                    )
                reordered_rows.append(target[id_to_idx[pid]])
            reordered[layer] = np.asarray(reordered_rows, dtype=np.float32)

        return reordered

    def _split_probe_indices(self, n: int, seed: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        train_frac = float(self.alignment_cfg.get("probe_train_fraction", 0.8))
        n_train = max(1, min(n - 1, int(round(n * train_frac))))
        train_idx = idx[:n_train]
        eval_idx = idx[n_train:]
        if len(eval_idx) == 0:
            eval_idx = idx[-1:]
            train_idx = idx[:-1]
        return {"train": train_idx, "eval": eval_idx}

    @staticmethod
    def _lambda_value(step: int, base_lambda: float, schedule: str, ramp_steps: int) -> float:
        if schedule == "constant":
            return base_lambda
        if schedule == "ramp":
            if ramp_steps <= 0:
                return base_lambda
            frac = min(1.0, step / float(ramp_steps))
            return base_lambda * frac
        return base_lambda

    @staticmethod
    def _require_matching_alignment_width(
        acts: torch.Tensor,
        targets: torch.Tensor,
        *,
        student_layer: str,
        teacher_layer: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if acts.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                "Alignment expects 2D mean-pooled representations, got "
                f"student shape {tuple(acts.shape)} and target shape {tuple(targets.shape)}"
            )
        if acts.shape[1] != targets.shape[1]:
            raise ValueError(
                "Alignment width mismatch between student activations and PHATE target: "
                f"student layer '{student_layer}' has width {acts.shape[1]}, "
                f"teacher target '{teacher_layer}' has width {targets.shape[1]}"
            )
        return acts, targets

    def _update_lr(self, optimizer: AdamW, step: int, total_steps: int, base_lr: float) -> None:
        sched_type = str(self.lr_scheduler_cfg.get("type", "none")).lower()
        warmup_steps = int(self.lr_scheduler_cfg.get("warmup_steps", 0))
        min_lr = float(self.lr_scheduler_cfg.get("min_lr", 0.0))

        if sched_type == "none":
            lr = base_lr
        else:
            if step <= warmup_steps and warmup_steps > 0:
                lr = base_lr * (step / float(warmup_steps))
            else:
                if sched_type == "cosine":
                    progress_num = max(0, step - warmup_steps)
                    progress_den = max(1, total_steps - warmup_steps)
                    progress = min(1.0, progress_num / float(progress_den))
                    lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
                else:
                    lr = base_lr

        for group in optimizer.param_groups:
            group["lr"] = lr

    @staticmethod
    def _safe_perplexity(loss_val: float) -> float:
        if not np.isfinite(loss_val):
            return float("nan")
        if loss_val > 50:
            return float("inf")
        return float(math.exp(loss_val))

    def _is_analysis_checkpoint_step(self, step: int) -> bool:
        return int(step) in self.analysis_checkpoint_step_set

    def _save_periodic_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        run_output_dir: Path,
        step: int,
        eval_loss_for_rank: float | None,
        best_checkpoints: List[tuple[float, Path]],
    ) -> None:
        ckpt_step_path = run_output_dir / f"student_step{step}.pt"
        torch.save(model.state_dict(), ckpt_step_path)

        if self._is_analysis_checkpoint_step(step):
            analysis_ckpt_step_path = run_output_dir / f"student_analysis_step{step}.pt"
            torch.save(model.state_dict(), analysis_ckpt_step_path)

        if eval_loss_for_rank is None:
            return

        best_checkpoints.append((float(eval_loss_for_rank), ckpt_step_path))
        best_checkpoints.sort(key=lambda x: x[0])
        while len(best_checkpoints) > int(self.save_top_k):
            _loss, path_to_remove = best_checkpoints.pop()
            if path_to_remove.exists():
                path_to_remove.unlink()

    @staticmethod
    def _save_analysis_checkpoint(
        *,
        model: torch.nn.Module,
        run_output_dir: Path,
        step: int,
    ) -> Path:
        analysis_ckpt_step_path = run_output_dir / f"student_analysis_step{step}.pt"
        torch.save(model.state_dict(), analysis_ckpt_step_path)
        return analysis_ckpt_step_path

    @staticmethod
    def _path_segments(path: str) -> List[tuple[str, List[int]]]:
        segments: List[tuple[str, List[int]]] = []
        for part in str(path).split("."):
            if not part:
                continue
            name = ""
            indices: List[int] = []
            cursor = 0
            while cursor < len(part):
                if part[cursor] == "[":
                    end = part.index("]", cursor)
                    indices.append(int(part[cursor + 1:end]))
                    cursor = end + 1
                    continue
                name += part[cursor]
                cursor += 1
            segments.append((name, indices))
        return segments

    @classmethod
    def _resolve_module_by_path(cls, root: torch.nn.Module, path: str) -> torch.nn.Module:
        current: Any = root
        for name, indices in cls._path_segments(path):
            if name:
                current = getattr(current, name)
            for index in indices:
                current = current[index]
        if not isinstance(current, torch.nn.Module):
            raise TypeError(f"Resolved path '{path}' is not a torch.nn.Module")
        return current

    @staticmethod
    def _set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
        for param in module.parameters():
            param.requires_grad = trainable

    def _freeze_aligned_layers(self, model: torch.nn.Module) -> List[str]:
        forward_model = self._activation_forward_model(model, alignment_side=self.alignment_side)
        frozen_paths: List[str] = []
        for layer_path in self.student_layer_specs:
            normalized_path = self._normalize_layer_path_for_model(
                forward_model=forward_model,
                original_model=model,
                layer_path=layer_path,
                alignment_side=self.alignment_side,
            )
            module = self._resolve_module_by_path(forward_model, normalized_path)
            self._set_module_trainable(module, False)
            frozen_paths.append(layer_path)
        return frozen_paths

    @staticmethod
    def _unfreeze_all_layers(model: torch.nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = True

    def _build_optimizer_for_trainable_params(
        self,
        model: torch.nn.Module,
        *,
        learning_rate: float,
    ) -> AdamW:
        params = [param for param in model.parameters() if param.requires_grad]
        if not params:
            raise ValueError("No trainable parameters remain after applying freeze policy")
        return AdamW(
            params,
            lr=float(learning_rate),
            betas=tuple(self.optimizer_cfg.get("betas", [0.9, 0.95])),
            eps=float(self.optimizer_cfg.get("eps", 1e-8)),
            weight_decay=float(self.optimizer_cfg.get("weight_decay", 0.0)),
        )

    def _staged_training_enabled(self) -> bool:
        return bool(self.staged_training_cfg.get("enabled", False))

    def _is_control_task_only(self) -> bool:
        return str(self.training_regime).lower() == "control_task_only"

    def _control_budget_steps(self) -> tuple[int, int]:
        phase1_steps = max(0, int((self.staged_training_cfg.get("phase1") or {}).get("max_steps", 0)))
        phase2_steps = max(0, int((self.staged_training_cfg.get("phase2") or {}).get("max_steps", 0)))
        phase3_steps = max(0, int((self.staged_training_cfg.get("phase3") or {}).get("max_steps", 0)))
        return phase2_steps + phase3_steps, phase1_steps + phase2_steps + phase3_steps

    def _control_analysis_checkpoint_steps(self) -> List[int]:
        phase23_budget_steps, full_budget_steps = self._control_budget_steps()
        explicit_steps = list(self.analysis_checkpoint_steps)
        return sorted(
            {
                int(step)
                for step in [*explicit_steps, phase23_budget_steps, full_budget_steps]
                if int(step) > 0
            }
        )

    def _analysis_steps_for_phase(
        self,
        *,
        phase_start_step: int,
        phase_steps: int,
        snapshot_count: int,
        include_phase_end: bool,
    ) -> List[int]:
        if phase_steps <= 0 or snapshot_count <= 0:
            return []
        end_step = phase_start_step + phase_steps - 1
        if snapshot_count == 1:
            return [end_step] if include_phase_end else [phase_start_step]
        raw = np.linspace(phase_start_step, end_step, num=snapshot_count, endpoint=True)
        steps = sorted({int(round(value)) for value in raw})
        if include_phase_end:
            steps.append(end_step)
        return sorted({step for step in steps if phase_start_step <= step <= end_step})

    def _resolved_analysis_checkpoint_steps(self) -> List[int]:
        if self._is_control_task_only():
            return self._control_analysis_checkpoint_steps()
        if self.analysis_checkpoint_steps:
            return list(self.analysis_checkpoint_steps)
        if not self._staged_training_enabled():
            return []
        checkpoint_cfg = dict(self.staged_training_cfg.get("checkpoint_analysis") or {})
        if not checkpoint_cfg.get("enabled", False):
            return []
        include_phase_end = bool(checkpoint_cfg.get("include_phase_end", True))
        phase_specs = [
            ("phase1", int((self.staged_training_cfg.get("phase1") or {}).get("max_steps", 0)), int(checkpoint_cfg.get("phase1_snapshots", 0))),
            ("phase2", int((self.staged_training_cfg.get("phase2") or {}).get("max_steps", 0)), int(checkpoint_cfg.get("phase2_snapshots", 0))),
            ("phase3", int((self.staged_training_cfg.get("phase3") or {}).get("max_steps", 0)), int(checkpoint_cfg.get("phase3_snapshots", 0))),
        ]
        out: List[int] = []
        cursor = 1
        for _phase_name, phase_steps, snapshot_count in phase_specs:
            out.extend(
                self._analysis_steps_for_phase(
                    phase_start_step=cursor,
                    phase_steps=phase_steps,
                    snapshot_count=snapshot_count,
                    include_phase_end=include_phase_end,
                )
            )
            cursor += max(0, phase_steps)
        return sorted({int(step) for step in out if int(step) > 0})

    def _record_train_row(
        self,
        rows: List[Dict[str, float | str | bool]],
        *,
        global_step: int,
        phase_name: str,
        phase_step: int,
        train_lm_loss: float,
        train_align_loss: float,
        train_total_loss: float,
        learning_rate: float,
        aligned_layers_frozen: bool,
    ) -> None:
        rows.append(
            {
                "global_step": float(global_step),
                "phase": phase_name,
                "phase_step": float(phase_step),
                "train_lm_loss": float(train_lm_loss),
                "train_align_loss": float(train_align_loss),
                "train_total_loss": float(train_total_loss),
                "learning_rate": float(learning_rate),
                "aligned_layers_frozen": bool(aligned_layers_frozen),
            }
        )

    def _record_eval_row(
        self,
        rows: List[Dict[str, float | str | bool]],
        *,
        global_step: int,
        phase_name: str,
        phase_step: int,
        val_loss: float,
        align_mse: float,
        align_mse_per_layer: Dict[str, float],
        aligned_layers_frozen: bool,
    ) -> Dict[str, float | str | bool]:
        row: Dict[str, float | str | bool] = {
            "global_step": float(global_step),
            "phase": phase_name,
            "phase_step": float(phase_step),
            "val_loss": float(val_loss),
            "val_perplexity": float(self._safe_perplexity(val_loss)),
            "align_mse": float(align_mse),
            "aligned_layers_frozen": bool(aligned_layers_frozen),
        }
        for layer_name, value in align_mse_per_layer.items():
            safe = self._sanitize_layer_name(layer_name)
            row[f"align_mse__{safe}"] = float(value)
        rows.append(row)
        return row

    def _wandb_log_phase_metrics(
        self,
        *,
        combo_idx: int,
        phase_name: str,
        step: int,
        metrics: Dict[str, float | bool | str],
        kind: str,
    ) -> None:
        combo_prefix = f"distill_sweep/combo_{int(combo_idx):04d}"
        payload: Dict[str, float | bool | str] = {}
        for key, value in metrics.items():
            payload[f"{combo_prefix}/{kind}/{phase_name}/{key}"] = value
            payload[f"{combo_prefix}/{kind}/{key}"] = value
        self._wandb_log(payload, step=step)

    @staticmethod
    def _analysis_queue_path(run_output_dir: Path) -> Path:
        return run_output_dir / "analysis_queue.jsonl"

    @staticmethod
    def _append_analysis_queue_entry(queue_path: Path, entry: Dict[str, Any]) -> None:
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _build_analysis_queue_entry(
        self,
        *,
        combo: Dict[str, Any],
        run_output_dir: Path,
        step: int,
        phase_name: str,
        phase_step: int,
        probe_buffers: Dict[str, Any],
        probe_source_ids: List[int],
        probe_seed: int,
        data_order_seed: int,
        dataloader_seed: int,
    ) -> Dict[str, Any]:
        return {
            "queue_schema_version": "student_analysis_queue_v1",
            "run_dir": str(run_output_dir),
            "checkpoint_path": str(run_output_dir / f"student_analysis_step{int(step)}.pt"),
            "step": int(step),
            "phase": phase_name,
            "phase_step": int(phase_step),
            "family": self.family_name or "unknown",
            "student_model": combo["student_model_name_or_path"],
            "student_model_revision": combo["student_model_revision"],
            "student_trust_remote_code": bool(self.student_trust_remote_code),
            "student_model_family": combo["student_model_family"],
            "student_model_config_name_or_path": combo["student_model_config_name_or_path"],
            "student_model_config_overrides": dict(combo["student_model_config_overrides"]),
            "tokenizer_name": self.tokenizer_name or combo["student_model_name_or_path"],
            "precision": self.precision,
            "micro_batch_size": int(self.micro_batch_size),
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "dataset_path": self.dataset_path,
            "dataset_revision": self.dataset_revision,
            "text_field": self.text_field,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "probe_split": self.probe_split,
            "probe_ids_path": self.probe_ids_path,
            "exclude_probe_from_train": bool(self.exclude_probe_from_train),
            "max_length": int(combo["max_length"]),
            "probe_n_samples": int(probe_buffers["n"]),
            "probe_source_ids": [int(v) for v in probe_source_ids],
            "probe_seed": int(probe_seed),
            "data_order_seed": int(data_order_seed),
            "dataloader_seed": int(dataloader_seed),
            "student_layer_specs": list(self.student_layer_specs),
            "alignment_side": self.alignment_side,
            "wandb_parent_run_id": getattr(getattr(wandb, "run", None), "id", None) if wandb is not None else None,
            "wandb_parent_run_name": getattr(getattr(wandb, "run", None), "name", None) if wandb is not None else None,
            "wandb_project": getattr(getattr(wandb, "run", None), "project", None) if wandb is not None else None,
            "wandb_entity": getattr(getattr(wandb, "run", None), "entity", None) if wandb is not None else None,
        }

    def _precision_to_dtype(self) -> torch.dtype | None:
        p = str(self.precision).lower()
        if p in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if p in {"16", "fp16", "float16"}:
            return torch.float16
        return None

    def _autocast_ctx(self, device: torch.device):
        dtype = self._precision_to_dtype()
        if dtype is None:
            return nullcontext()
        # Keep CPU path simple and stable.
        if device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=dtype)

    @staticmethod
    def _shift_logits_and_labels(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return shift_logits, shift_labels

    def _lm_loss(
        self,
        outputs,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        label_smoothing = float(self.regularization_cfg.get("label_smoothing", 0.0))
        if label_smoothing <= 0.0:
            return outputs.loss
        logits = outputs.logits
        labels = batch["labels"]
        shift_logits, shift_labels = self._shift_logits_and_labels(logits, labels)
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )

    @staticmethod
    def _activation_forward_model(
        model: torch.nn.Module,
        alignment_side: str | None = None,
    ) -> torch.nn.Module:
        model_config = getattr(model, "config", None)
        is_encoder_decoder = bool(getattr(model_config, "is_encoder_decoder", False))
        if (
            str(alignment_side).lower() == "encoder"
            and is_encoder_decoder
            and hasattr(model, "get_encoder")
        ):
            encoder = model.get_encoder()
            if isinstance(encoder, torch.nn.Module):
                return encoder
        base_model_prefix = getattr(model, "base_model_prefix", None)
        if isinstance(base_model_prefix, str) and base_model_prefix:
            base_model = getattr(model, base_model_prefix, None)
            if isinstance(base_model, torch.nn.Module):
                return base_model
        base_model = getattr(model, "base_model", None)
        if isinstance(base_model, torch.nn.Module):
            return base_model
        return model

    @staticmethod
    def _normalize_layer_path_for_model(
        forward_model: torch.nn.Module,
        original_model: torch.nn.Module,
        layer_path: str,
        alignment_side: str | None = None,
    ) -> str:
        if forward_model is original_model:
            return layer_path

        base_model_prefix = getattr(original_model, "base_model_prefix", None)
        normalized = layer_path
        if isinstance(base_model_prefix, str) and base_model_prefix:
            prefix = f"{base_model_prefix}."
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]

        if normalized.startswith("transformer.h"):
            suffix = normalized[len("transformer.h"):]
            return f"layers{suffix}"
        if normalized.startswith("model.layers"):
            suffix = normalized[len("model.layers"):]
            return f"layers{suffix}"
        if normalized.startswith("transformer.layers"):
            suffix = normalized[len("transformer.layers"):]
            return f"layers{suffix}"
        if str(alignment_side).lower() == "encoder" and normalized.startswith("encoder."):
            model_config = getattr(original_model, "config", None)
            is_encoder_decoder = bool(getattr(model_config, "is_encoder_decoder", False))
            if is_encoder_decoder and hasattr(original_model, "get_encoder"):
                encoder = original_model.get_encoder()
                if forward_model is encoder:
                    return normalized[len("encoder."):]
        if normalized.startswith("transformer."):
            return normalized[len("transformer."):]
        if normalized.startswith("model."):
            return normalized[len("model."):]
        return normalized

    def _activations_from_batch(
        self,
        model: torch.nn.Module,
        layer_paths: Sequence[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        detach: bool = True,
    ) -> Dict[str, torch.Tensor]:
        forward_model = self._activation_forward_model(model, alignment_side=self.alignment_side)
        normalized_paths = {
            layer_path: self._normalize_layer_path_for_model(
                forward_model=forward_model,
                original_model=model,
                layer_path=layer_path,
                alignment_side=self.alignment_side,
            )
            for layer_path in layer_paths
        }
        specs = [LayerSpec(path=normalized_paths[layer_path], reduce="mean") for layer_path in layer_paths]
        extractor = ActivationExtractor(specs, detach=detach)
        guard = torch.no_grad() if detach else nullcontext()
        with guard:
            with self._autocast_ctx(device):
                with extractor.capture(forward_model):
                    forward_model(input_ids=input_ids, attention_mask=attention_mask)
        acts = extractor.get_activations()
        remapped = {
            layer_path: acts[normalized_paths[layer_path]]
            for layer_path in layer_paths
            if normalized_paths[layer_path] in acts
        }
        missing = [layer for layer in layer_paths if layer not in remapped]
        if missing:
            raise KeyError(f"Could not extract activations for layer paths: {missing}")
        return remapped

    def _eval_lm(
        self,
        model: torch.nn.Module,
        loader,
        device: torch.device,
    ) -> float:
        model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= self.eval_max_batches:
                    break
                batch = self._to_device(batch, device)
                with self._autocast_ctx(device):
                    out = model(**batch)
                losses.append(float(out.loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("nan")

    def _eval_alignment(
        self,
        model: torch.nn.Module,
        device: torch.device,
        probe_buffers: Dict[str, Any],
        eval_idx: np.ndarray,
        layer_pairs: Sequence[Dict[str, Any]],
    ) -> tuple[float, Dict[str, float]]:
        if len(eval_idx) == 0:
            return float("nan"), {}
        eval_batch_size = int(
            self.alignment_cfg.get(
                "eval_batch_size",
                self.alignment_cfg.get("batch_size", self.micro_batch_size),
            )
        )
        if eval_batch_size <= 0:
            eval_batch_size = int(self.micro_batch_size)

        per_layer_weighted_sum: Dict[str, float] = {}
        per_layer_numel: Dict[str, int] = {}
        weighted_total = 0.0
        total_weight = 0.0

        for start in range(0, len(eval_idx), eval_batch_size):
            batch_idx = eval_idx[start:start + eval_batch_size]
            ids = probe_buffers["input_ids"][batch_idx].to(device)
            mask = probe_buffers["attention_mask"][batch_idx].to(device)
            acts_by_layer = self._activations_from_batch(
                model=model,
                layer_paths=[pair["student_layer"] for pair in layer_pairs],
                input_ids=ids,
                attention_mask=mask,
                device=device,
                detach=True,
            )

            batch_weighted_total = 0.0
            batch_total_weight = 0.0
            for pair in layer_pairs:
                student_layer = str(pair["student_layer"])
                teacher_layer = str(pair["teacher_layer"])
                weight = float(pair["weight"])
                acts = acts_by_layer[student_layer].to(device)
                targets = probe_buffers["targets"][teacher_layer][batch_idx].to(device)
                acts, targets = self._require_matching_alignment_width(
                    acts,
                    targets,
                    student_layer=student_layer,
                    teacher_layer=teacher_layer,
                )
                targets = targets.to(dtype=acts.dtype)
                mse = float(F.mse_loss(acts, targets, reduction="mean").detach().cpu().item())
                numel = int(acts.numel())
                per_layer_weighted_sum[student_layer] = per_layer_weighted_sum.get(student_layer, 0.0) + (mse * numel)
                per_layer_numel[student_layer] = per_layer_numel.get(student_layer, 0) + numel
                batch_weighted_total += weight * mse
                batch_total_weight += weight

            if batch_total_weight > 0.0:
                weighted_total += (batch_weighted_total / batch_total_weight) * len(batch_idx)
                total_weight += len(batch_idx)

        per_layer = {
            student_layer: float(per_layer_weighted_sum[student_layer] / per_layer_numel[student_layer])
            for student_layer in per_layer_weighted_sum
            if per_layer_numel[student_layer] > 0
        }

        if total_weight <= 0.0:
            return float("nan"), per_layer
        return float(weighted_total / total_weight), per_layer

    @staticmethod
    def _cleanup_combo_resources(
        *,
        trainer_module: HFTrainerModule | None,
        model: torch.nn.Module | None,
        optimizer: AdamW | None,
        dm: TextDataModule | None,
        probe_buffers: Dict[str, Any] | None,
        aligned_targets_for_run: Dict[str, np.ndarray] | None,
        train_loader: Any,
        train_iter: Any,
    ) -> None:
        if trainer_module is not None:
            trainer_module.network = None
            trainer_module.tokenizer = None

        del train_iter
        del train_loader
        del probe_buffers
        del optimizer
        del model
        del aligned_targets_for_run
        del dm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_single_combo_legacy(
        self,
        combo: Dict[str, Any],
        aligned_targets: Dict[str, np.ndarray],
        expected_probe_ids: List[int],
        run_output_dir: Path,
        combo_idx: int,
    ) -> Dict[str, Any]:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        combo_seed = int(combo["seed"])
        model_init_seed = self._resolve_stage_seed("model_init_seed", combo_seed)
        data_order_seed = self._resolve_stage_seed("data_order_seed", combo_seed)
        dataloader_seed = self._resolve_stage_seed("dataloader_seed", combo_seed)
        probe_seed = self._resolve_stage_seed("global_seed", combo_seed)
        layer_pairs = self._resolve_layer_pairs()
        layer_scheme_name = self._infer_layer_scheme_name(layer_pairs)
        eval_example_limit = None
        if int(self.eval_max_batches) > 0:
            eval_example_limit = int(self.eval_max_batches) * int(self.micro_batch_size)

        random.seed(model_init_seed)
        torch.manual_seed(model_init_seed)
        np.random.seed(model_init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_init_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        trainer_module: HFTrainerModule | None = None
        model: torch.nn.Module | None = None
        optimizer: AdamW | None = None
        dm: TextDataModule | None = None
        probe_buffers: Dict[str, Any] | None = None
        aligned_targets_for_run: Dict[str, np.ndarray] | None = None
        train_loader = None
        train_iter = None

        try:
            dm = TextDataModule(
                dataset_name=self.dataset_name,
                dataset_config=self.dataset_config,
                dataset_path=self.dataset_path,
                dataset_revision=self.dataset_revision,
                train_split=self.train_split,
                val_split=self.val_split,
                test_split=self.test_split,
                probe_split=self.probe_split,
                token_budget=combo["token_budget"],
                train_example_offset=int(combo["train_example_offset"]),
                probe_ids_path=self.probe_ids_path,
                exclude_probe_from_train=self.exclude_probe_from_train,
                val_example_limit=eval_example_limit,
                test_example_limit=eval_example_limit,
                text_field=self.text_field,
                lm_objective=combo["student_model_family"],
                tokenizer_name=self.tokenizer_name or combo["student_model_name_or_path"],
                max_length=int(combo["max_length"]),
                batch_size=int(self.micro_batch_size),
                probe_n_samples=int(min(target.shape[0] for target in aligned_targets.values())),
                seed=probe_seed,
                data_order_seed=data_order_seed,
                dataloader_seed=dataloader_seed,
            )
            dm.setup()
            aligned_targets_for_run = self._materialize_targets_for_datamodule(
                dm=dm,
                aligned_targets=aligned_targets,
                expected_probe_ids=expected_probe_ids,
            )
            train_loader = dm.train_dataloader()

            model_config_overrides = dict(combo["student_model_config_overrides"])
            model_config_overrides.update(
                {
                    "hidden_dropout": float(self.regularization_cfg.get("dropout", 0.0)),
                    "attention_dropout": float(self.regularization_cfg.get("dropout", 0.0)),
                    "resid_pdrop": float(self.regularization_cfg.get("dropout", 0.0)),
                    "embd_pdrop": float(self.regularization_cfg.get("dropout", 0.0)),
                    "attn_pdrop": float(self.regularization_cfg.get("dropout", 0.0)),
                }
            )
            hf_cfg = HFTrainerConfig(
                model_name_or_path=combo["student_model_name_or_path"],
                model_revision=combo["student_model_revision"],
                init_mode="from_config" if self.init_from_scratch else "pretrained",
                model_config_name_or_path=(
                    combo["student_model_config_name_or_path"] or combo["student_model_name_or_path"]
                ),
                model_config_overrides=model_config_overrides,
                tokenizer_name=self.tokenizer_name or combo["student_model_name_or_path"],
                tokenizer_revision=combo["student_model_revision"],
                learning_rate=float(combo["learning_rate"]),
                weight_decay=float(self.optimizer_cfg.get("weight_decay", 0.0)),
                warmup_steps=int(self.lr_scheduler_cfg.get("warmup_steps", 0)),
                adam_epsilon=float(self.optimizer_cfg.get("eps", 1e-8)),
                trust_remote_code=self.student_trust_remote_code,
                torch_dtype=self._precision_to_dtype(),
                model_family=combo["student_model_family"],
            )
            trainer_module = HFTrainerModule(hf_cfg)
            trainer_module.configure_model()
            model = trainer_module.network
            assert model is not None

            init_ckpt_path = combo.get("student_init_checkpoint_path")
            if isinstance(init_ckpt_path, str) and init_ckpt_path:
                checkpoint_payload = torch.load(init_ckpt_path, map_location="cpu")
                if isinstance(checkpoint_payload, dict) and "model_state_dict" in checkpoint_payload:
                    state_dict = checkpoint_payload["model_state_dict"]
                else:
                    state_dict = checkpoint_payload
                model.load_state_dict(state_dict)

            device = self._select_device()
            model = model.to(device)

            optimizer = AdamW(
                model.parameters(),
                lr=float(combo["learning_rate"]),
                betas=tuple(self.optimizer_cfg.get("betas", [0.9, 0.95])),
                eps=float(self.optimizer_cfg.get("eps", 1e-8)),
                weight_decay=float(self.optimizer_cfg.get("weight_decay", 0.0)),
            )

            probe_buffers = self._make_probe_buffers(dm, aligned_targets=aligned_targets_for_run)
            probe_source_ids_for_analysis = self._probe_ids_from_datamodule(dm)[: int(probe_buffers["n"])]
            split = self._split_probe_indices(probe_buffers["n"], seed=probe_seed)
            probe_train_idx = split["train"]
            probe_eval_idx = split["eval"]
            rng = np.random.default_rng(probe_seed + 17)

            steps = self._compute_max_steps(
                token_budget=combo["token_budget"],
                max_length=int(combo["max_length"]),
                train_batches=len(train_loader),
            )
            effective_batch = self._effective_global_batch_size()
            lambda_schedule = str(self.alignment_cfg.get("lambda_schedule", "constant"))
            lambda_ramp_steps = int(self.alignment_cfg.get("lambda_ramp_steps", 0))
            align_every = int(self.alignment_cfg.get("sample_every_n_steps", 1))
            align_batch_size = int(self.alignment_cfg.get("batch_size", 16))
            mse_reduction = str(self.alignment_cfg.get("mse_reduction", "mean"))
            eval_every = int(self.eval_every_n_steps) if self.eval_every_n_steps is not None else 0
            save_every = int(self.save_every_n_steps) if self.save_every_n_steps is not None else 0

            train_iter = iter(train_loader)
            lm_losses: List[float] = []
            align_losses: List[float] = []
            last_lambda = 0.0
            periodic_eval_rows: List[Dict[str, float]] = []
            train_history_rows: List[Dict[str, float]] = []
            best_checkpoints: List[tuple[float, Path]] = []

            model.train()
            for step in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)
                self._update_lr(
                    optimizer=optimizer,
                    step=step,
                    total_steps=steps,
                    base_lr=float(combo["learning_rate"]),
                )

                lm_loss_values: List[float] = []
                for _ in range(int(self.grad_accum_steps)):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)

                    batch = self._to_device(batch, device)
                    with self._autocast_ctx(device):
                        outputs = model(**batch)
                        lm_loss = self._lm_loss(outputs, batch)
                    lm_loss_weight = float(combo["lm_loss_weight"])
                    if lm_loss_weight != 0.0:
                        ((lm_loss * lm_loss_weight) / float(self.grad_accum_steps)).backward()
                    lm_loss_values.append(float(lm_loss.detach().cpu().item()))

                lm_loss = torch.tensor(float(np.mean(lm_loss_values)), device=device)
                lm_loss_weight = float(combo["lm_loss_weight"])
                total_loss = lm_loss * lm_loss_weight
                current_align_loss = torch.tensor(0.0, device=device)
                current_align_loss_per_layer: Dict[str, float] = {}

                if step % align_every == 0 and len(probe_train_idx) > 0:
                    sampled_idx = rng.choice(
                        probe_train_idx,
                        size=min(align_batch_size, len(probe_train_idx)),
                        replace=len(probe_train_idx) < align_batch_size,
                    )
                    probe_ids = probe_buffers["input_ids"][sampled_idx].to(device)
                    probe_mask = probe_buffers["attention_mask"][sampled_idx].to(device)
                    acts_by_layer = self._activations_from_batch(
                        model=model,
                        layer_paths=[pair["student_layer"] for pair in layer_pairs],
                        input_ids=probe_ids,
                        attention_mask=probe_mask,
                        device=device,
                        detach=False,
                    )
                    total_weight = 0.0
                    weighted_loss = torch.tensor(0.0, device=device)
                    for pair in layer_pairs:
                        student_layer = str(pair["student_layer"])
                        teacher_layer = str(pair["teacher_layer"])
                        weight = float(pair["weight"])
                        acts = acts_by_layer[student_layer]
                        probe_targets = probe_buffers["targets"][teacher_layer][sampled_idx].to(device)
                        acts, probe_targets = self._require_matching_alignment_width(
                            acts,
                            probe_targets,
                            student_layer=student_layer,
                            teacher_layer=teacher_layer,
                        )
                        probe_targets = probe_targets.to(dtype=acts.dtype)
                        layer_loss = F.mse_loss(
                            acts.float(),
                            probe_targets.float(),
                            reduction=mse_reduction,
                        )
                        current_align_loss_per_layer[student_layer] = float(layer_loss.detach().cpu().item())
                        weighted_loss = weighted_loss + (weight * layer_loss)
                        total_weight += weight

                    if total_weight <= 0.0:
                        raise ValueError("Sum of layer alignment weights must be positive")
                    current_align_loss = weighted_loss / total_weight
                    last_lambda = self._lambda_value(
                        step=step,
                        base_lambda=float(combo["lambda_align"]),
                        schedule=lambda_schedule,
                        ramp_steps=lambda_ramp_steps,
                    )
                    lambda_tensor = torch.tensor(last_lambda, device=device, dtype=current_align_loss.dtype)
                    align_term = lambda_tensor * current_align_loss
                    total_loss = total_loss + align_term
                    align_term.backward()

                if self.gradient_clip_norm and self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                optimizer.step()

                lm_losses.append(float(lm_loss.detach().cpu().item()))
                align_losses.append(float(current_align_loss.detach().cpu().item()))
                current_lr = float(optimizer.param_groups[0]["lr"])
                if (
                    int(self.train_history_every_n_steps) > 0
                    and (step % int(self.train_history_every_n_steps) == 0 or step == 1 or step == steps)
                ):
                    train_row = {
                        "step": float(step),
                        "train_lm_loss": float(lm_loss.detach().cpu().item()),
                        "train_align_loss": float(current_align_loss.detach().cpu().item()),
                        "train_total_loss": float(total_loss.detach().cpu().item()),
                        "lambda_value": float(last_lambda),
                        "lm_loss_weight": float(combo["lm_loss_weight"]),
                        "learning_rate": current_lr,
                    }
                    train_history_rows.append(train_row)
                    combo_prefix = f"distill_sweep/combo_{int(combo_idx):04d}"
                    self._wandb_log(
                        {
                            f"{combo_prefix}/train_lm_loss": train_row["train_lm_loss"],
                            f"{combo_prefix}/train_align_loss": train_row["train_align_loss"],
                            f"{combo_prefix}/train_total_loss": train_row["train_total_loss"],
                            f"{combo_prefix}/lambda_value": train_row["lambda_value"],
                            f"{combo_prefix}/lm_loss_weight": train_row["lm_loss_weight"],
                            f"{combo_prefix}/learning_rate": train_row["learning_rate"],
                            f"{combo_prefix}/lambda_align": float(combo["lambda_align"]),
                        },
                        step=step,
                    )

                if eval_every > 0 and (step % eval_every == 0 or step == steps):
                    eval_loss = self._eval_lm(model, dm.val_dataloader(), device=device)
                    model.train()
                    eval_row = {
                        "step": float(step),
                        "val_loss": float(eval_loss),
                        "val_perplexity": float(self._safe_perplexity(eval_loss)),
                        }
                    periodic_eval_rows.append(eval_row)
                    combo_prefix = f"distill_sweep/combo_{int(combo_idx):04d}"
                    self._wandb_log(
                        {
                            f"{combo_prefix}/val_loss": eval_row["val_loss"],
                            f"{combo_prefix}/val_perplexity": eval_row["val_perplexity"],
                        },
                        step=step,
                    )

                    if save_every > 0 and (step % save_every == 0 or step == steps):
                        eval_loss_for_rank = float(periodic_eval_rows[-1]["val_loss"]) if periodic_eval_rows else None
                        self._save_periodic_checkpoint(
                            model=model,
                            run_output_dir=run_output_dir,
                            step=step,
                            eval_loss_for_rank=eval_loss_for_rank,
                            best_checkpoints=best_checkpoints,
                        )

            val_loss = self._eval_lm(model, dm.val_dataloader(), device=device)
            val_ppl = self._safe_perplexity(val_loss)
            test_loss = self._eval_lm(model, dm.test_dataloader(), device=device)
            test_ppl = self._safe_perplexity(test_loss)
            align_eval_mse, align_eval_per_layer = self._eval_alignment(
                model=model,
                device=device,
                probe_buffers=probe_buffers,
                eval_idx=probe_eval_idx,
                layer_pairs=layer_pairs,
            )

            checkpoint_path = run_output_dir / "student_last.pt"
            torch.save(model.state_dict(), checkpoint_path)

            metrics = {
                "train_lm_loss_last": lm_losses[-1] if lm_losses else float("nan"),
                "train_lm_loss_mean": float(np.mean(lm_losses)) if lm_losses else float("nan"),
                "train_align_loss_last": align_losses[-1] if align_losses else float("nan"),
                "train_align_loss_mean": float(np.mean(align_losses)) if align_losses else float("nan"),
                "train_total_loss_last": train_history_rows[-1]["train_total_loss"] if train_history_rows else float("nan"),
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "test_loss": test_loss,
                "test_perplexity": test_ppl,
                "align_mse": align_eval_mse,
                "effective_global_batch_size": int(effective_batch),
                "lm_loss_weight": float(combo["lm_loss_weight"]),
                "train_example_offset": int(combo["train_example_offset"]),
            }
            for layer_name in self.student_layer_specs:
                safe = layer_name.replace(".", "__")
                metrics[f"align_mse__{safe}"] = float(align_eval_per_layer.get(layer_name, float("nan")))
            if current_align_loss_per_layer:
                for layer_name in self.student_layer_specs:
                    safe = layer_name.replace(".", "__")
                    metrics[f"train_align_loss_last__{safe}"] = float(
                        current_align_loss_per_layer.get(layer_name, float("nan"))
                    )
            metrics_path = run_output_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
            periodic_eval_path = run_output_dir / "periodic_eval.json"
            periodic_eval_path.write_text(
                json.dumps(periodic_eval_rows, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            train_history_json_path = run_output_dir / "train_history.json"
            train_history_json_path.write_text(
                json.dumps(train_history_rows, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            train_history_csv_path = run_output_dir / "train_history.csv"
            with train_history_csv_path.open("w", encoding="utf-8", newline="") as f:
                fieldnames = [
                    "step",
                    "train_lm_loss",
                    "train_align_loss",
                    "train_total_loss",
                    "lambda_value",
                    "lm_loss_weight",
                    "learning_rate",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(train_history_rows)

            row = {
                "result_schema_version": self.RESULT_SCHEMA_VERSION,
                "family": self.family_name or "unknown",
                "family_architecture": self.family_architecture or "unknown",
                "alignment_side": self.alignment_side or "unknown",
                "layer_scheme": layer_scheme_name,
                "teacher_layer_specs": self._json_list([pair["teacher_layer"] for pair in layer_pairs]),
                "student_layer_specs": self._json_list([pair["student_layer"] for pair in layer_pairs]),
                "layer_loss_weights": self._json_list([float(pair["weight"]) for pair in layer_pairs]),
                "teacher_model": combo["teacher_model_name_or_path"],
                "teacher_model_revision": combo["teacher_model_revision"],
                "student_model": combo["student_model_name_or_path"],
                "student_model_revision": combo["student_model_revision"],
                "tokenizer_name": self.tokenizer_name or combo["student_model_name_or_path"],
                "dataset_name": self.dataset_name,
                "dataset_config": self.dataset_config,
                "dataset_path": self.dataset_path,
                "dataset_revision": self.dataset_revision,
                "text_field": self.text_field,
                "train_split": self.train_split,
                "val_split": self.val_split,
                "test_split": self.test_split,
                "probe_split": self.probe_split,
                "exclude_probe_from_train": bool(self.exclude_probe_from_train),
                "seed": int(combo_seed),
                "seed_model_init": int(model_init_seed),
                "seed_data_order": int(data_order_seed),
                "seed_dataloader": int(dataloader_seed),
                "seed_probe": int(probe_seed),
                "lambda_align": float(combo["lambda_align"]),
                "learning_rate": float(combo["learning_rate"]),
                "max_length": int(combo["max_length"]),
                "token_budget": (
                    None if combo["token_budget"] is None else int(combo["token_budget"])
                ),
                "train_example_offset": int(combo["train_example_offset"]),
                "max_steps": int(steps),
                "effective_global_batch_size": int(effective_batch),
                "lm_loss_weight": float(combo["lm_loss_weight"]),
                "lambda_schedule": lambda_schedule,
                "lambda_last": float(last_lambda),
                "val_loss": float(val_loss),
                "val_perplexity": float(val_ppl),
                "test_loss": float(test_loss),
                "test_perplexity": float(test_ppl),
                "align_mse": float(align_eval_mse),
                "train_lm_loss_last": float(metrics["train_lm_loss_last"]),
                "train_lm_loss_mean": float(metrics["train_lm_loss_mean"]),
                "train_align_loss_last": float(metrics["train_align_loss_last"]),
                "train_align_loss_mean": float(metrics["train_align_loss_mean"]),
                "train_total_loss_last": float(metrics["train_total_loss_last"]),
                "num_alignment_layers": int(len(layer_pairs)),
                "probe_size": int(probe_buffers["n"]),
                "probe_ids_path": str(self.probe_ids_path) if self.probe_ids_path is not None else "",
                "teacher_stage": self.teacher_stage,
                "teacher_target_stage": self.teacher_target_stage,
                "teacher_probe_ids_key": self.teacher_probe_ids_key,
                "target_probe_ids_key": self.target_probe_ids_key,
                "run_dir": str(run_output_dir),
                "ckpt_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
                "periodic_eval_path": str(periodic_eval_path),
                "train_history_json_path": str(train_history_json_path),
                "train_history_csv_path": str(train_history_csv_path),
            }
            for layer_name in self.student_layer_specs:
                safe = layer_name.replace(".", "__")
                row_key = f"align_mse__{safe}"
                row[row_key] = float(align_eval_per_layer.get(layer_name, float("nan")))
            combo_prefix = f"distill_sweep/combo_{int(combo_idx):04d}"
            self._wandb_log(
                {
                    f"{combo_prefix}/final_val_loss": float(val_loss),
                    f"{combo_prefix}/final_val_perplexity": float(val_ppl),
                    f"{combo_prefix}/final_test_loss": float(test_loss),
                    f"{combo_prefix}/final_test_perplexity": float(test_ppl),
                    f"{combo_prefix}/final_align_mse": float(align_eval_mse),
                    f"{combo_prefix}/lambda_align": float(combo["lambda_align"]),
                    f"{combo_prefix}/lm_loss_weight": float(combo["lm_loss_weight"]),
                },
                step=int(steps),
            )
            return row
        finally:
            self._cleanup_combo_resources(
                trainer_module=trainer_module,
                model=model,
                optimizer=optimizer,
                dm=dm,
                probe_buffers=probe_buffers,
                aligned_targets_for_run=aligned_targets_for_run,
                train_loader=train_loader,
                train_iter=train_iter,
            )

    def _run_single_combo(
        self,
        combo: Dict[str, Any],
        aligned_targets: Dict[str, np.ndarray],
        expected_probe_ids: List[int],
        run_output_dir: Path,
        combo_idx: int,
    ) -> Dict[str, Any]:
        if not self._staged_training_enabled() and not self._is_control_task_only():
            return self._run_single_combo_legacy(
                combo=combo,
                aligned_targets=aligned_targets,
                expected_probe_ids=expected_probe_ids,
                run_output_dir=run_output_dir,
                combo_idx=combo_idx,
            )

        run_output_dir.mkdir(parents=True, exist_ok=True)
        combo_seed = int(combo["seed"])
        model_init_seed = self._resolve_stage_seed("model_init_seed", combo_seed)
        data_order_seed = self._resolve_stage_seed("data_order_seed", combo_seed)
        dataloader_seed = self._resolve_stage_seed("dataloader_seed", combo_seed)
        probe_seed = self._resolve_stage_seed("global_seed", combo_seed)
        layer_pairs = self._resolve_layer_pairs()
        layer_scheme_name = self._infer_layer_scheme_name(layer_pairs)
        eval_example_limit = None
        if int(self.eval_max_batches) > 0:
            eval_example_limit = int(self.eval_max_batches) * int(self.micro_batch_size)

        random.seed(model_init_seed)
        torch.manual_seed(model_init_seed)
        np.random.seed(model_init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_init_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        trainer_module: HFTrainerModule | None = None
        model: torch.nn.Module | None = None
        optimizer: AdamW | None = None
        dm: TextDataModule | None = None
        probe_buffers: Dict[str, Any] | None = None
        aligned_targets_for_run: Dict[str, np.ndarray] | None = None
        train_loader = None
        train_iter = None

        try:
            dm = TextDataModule(
                dataset_name=self.dataset_name,
                dataset_config=self.dataset_config,
                dataset_path=self.dataset_path,
                dataset_revision=self.dataset_revision,
                train_split=self.train_split,
                val_split=self.val_split,
                test_split=self.test_split,
                probe_split=self.probe_split,
                token_budget=combo["token_budget"],
                train_example_offset=int(combo["train_example_offset"]),
                probe_ids_path=self.probe_ids_path,
                exclude_probe_from_train=self.exclude_probe_from_train,
                val_example_limit=eval_example_limit,
                test_example_limit=eval_example_limit,
                text_field=self.text_field,
                lm_objective=combo["student_model_family"],
                tokenizer_name=self.tokenizer_name or combo["student_model_name_or_path"],
                max_length=int(combo["max_length"]),
                batch_size=int(self.micro_batch_size),
                probe_n_samples=int(min(target.shape[0] for target in aligned_targets.values())),
                seed=probe_seed,
                data_order_seed=data_order_seed,
                dataloader_seed=dataloader_seed,
            )
            dm.setup()
            aligned_targets_for_run = self._materialize_targets_for_datamodule(
                dm=dm,
                aligned_targets=aligned_targets,
                expected_probe_ids=expected_probe_ids,
            )
            train_loader = dm.train_dataloader()
            train_iter = iter(train_loader)

            model_config_overrides = dict(combo["student_model_config_overrides"])
            model_config_overrides.update(
                {
                    "hidden_dropout": float(self.regularization_cfg.get("dropout", 0.0)),
                    "attention_dropout": float(self.regularization_cfg.get("dropout", 0.0)),
                    "resid_pdrop": float(self.regularization_cfg.get("dropout", 0.0)),
                    "embd_pdrop": float(self.regularization_cfg.get("dropout", 0.0)),
                    "attn_pdrop": float(self.regularization_cfg.get("dropout", 0.0)),
                }
            )
            hf_cfg = HFTrainerConfig(
                model_name_or_path=combo["student_model_name_or_path"],
                model_revision=combo["student_model_revision"],
                init_mode="from_config" if self.init_from_scratch else "pretrained",
                model_config_name_or_path=(
                    combo["student_model_config_name_or_path"] or combo["student_model_name_or_path"]
                ),
                model_config_overrides=model_config_overrides,
                tokenizer_name=self.tokenizer_name or combo["student_model_name_or_path"],
                tokenizer_revision=combo["student_model_revision"],
                learning_rate=float(combo["learning_rate"]),
                weight_decay=float(self.optimizer_cfg.get("weight_decay", 0.0)),
                warmup_steps=int(self.lr_scheduler_cfg.get("warmup_steps", 0)),
                adam_epsilon=float(self.optimizer_cfg.get("eps", 1e-8)),
                trust_remote_code=self.student_trust_remote_code,
                torch_dtype=self._precision_to_dtype(),
                model_family=combo["student_model_family"],
            )
            trainer_module = HFTrainerModule(hf_cfg)
            trainer_module.configure_model()
            model = trainer_module.network
            assert model is not None

            init_ckpt_path = combo.get("student_init_checkpoint_path")
            if isinstance(init_ckpt_path, str) and init_ckpt_path:
                checkpoint_payload = torch.load(init_ckpt_path, map_location="cpu")
                state_dict = checkpoint_payload["model_state_dict"] if isinstance(checkpoint_payload, dict) and "model_state_dict" in checkpoint_payload else checkpoint_payload
                model.load_state_dict(state_dict)

            device = self._select_device()
            model = model.to(device)

            probe_buffers = self._make_probe_buffers(dm, aligned_targets=aligned_targets_for_run)
            probe_source_ids_for_analysis = self._probe_ids_from_datamodule(dm)[: int(probe_buffers["n"])]
            split = self._split_probe_indices(probe_buffers["n"], seed=probe_seed)
            probe_train_idx = split["train"]
            probe_eval_idx = split["eval"]
            rng = np.random.default_rng(probe_seed + 17)

            phase1_cfg = dict(self.staged_training_cfg.get("phase1") or {})
            phase2_cfg = dict(self.staged_training_cfg.get("phase2") or {})
            phase3_cfg = dict(self.staged_training_cfg.get("phase3") or {})
            control_phase23_budget_step: int | None = None
            control_full_budget_step: int | None = None
            control_phase23_analysis_checkpoint_path: str = ""
            if self._is_control_task_only():
                control_phase23_budget_step, control_full_budget_step = self._control_budget_steps()
                phase_specs = [
                    {
                        "name": "control_task_only",
                        "objective": "task_only_unfrozen",
                        "steps": int(control_full_budget_step),
                        "eval_every": int(self.eval_every_n_steps or 0),
                        "min_steps": 0,
                        "patience": 0,
                        "min_delta": 0.0,
                        "freeze_layers": False,
                    }
                ]
            else:
                phase_specs = [
                    {
                        "name": "phase1",
                        "objective": str(phase1_cfg.get("objective", "alignment_only")),
                        "steps": int(phase1_cfg.get("max_steps", 0)),
                        "eval_every": int(phase1_cfg.get("eval_every_n_steps", self.eval_every_n_steps or 0)),
                        "min_steps": int(phase1_cfg.get("min_steps", 0)),
                        "patience": int(phase1_cfg.get("early_stop_patience", 0)),
                        "min_delta": float(phase1_cfg.get("early_stop_min_delta", 0.0)),
                        "freeze_layers": False,
                    },
                    {
                        "name": "phase2",
                        "objective": str(phase2_cfg.get("objective", "task_only_frozen")),
                        "steps": int(phase2_cfg.get("max_steps", 0)),
                        "eval_every": int(self.eval_every_n_steps or 0),
                        "min_steps": 0,
                        "patience": 0,
                        "min_delta": 0.0,
                        "freeze_layers": True,
                    },
                    {
                        "name": "phase3",
                        "objective": str(phase3_cfg.get("objective", "task_only_unfrozen")),
                        "steps": int(phase3_cfg.get("max_steps", 0)),
                        "eval_every": int(self.eval_every_n_steps or 0),
                        "min_steps": 0,
                        "patience": 0,
                        "min_delta": 0.0,
                        "freeze_layers": False,
                    },
                ]

            total_planned_steps = sum(max(0, int(spec["steps"])) for spec in phase_specs)
            if total_planned_steps <= 0:
                raise ValueError("Staged training requires at least one positive phase step budget")
            if self.max_steps is not None and int(self.max_steps) != int(total_planned_steps):
                raise ValueError(
                    "staged_training phase step budgets must sum to training.max_steps "
                    f"({total_planned_steps} vs {self.max_steps})"
                )
            effective_batch = self._effective_global_batch_size()
            analysis_steps = self._resolved_analysis_checkpoint_steps()
            self.analysis_checkpoint_steps = analysis_steps
            self.analysis_checkpoint_step_set = set(analysis_steps)
            analysis_queue_path = self._analysis_queue_path(run_output_dir)
            if analysis_queue_path.exists():
                analysis_queue_path.unlink()
            analysis_jobs: List[Dict[str, Any]] = []
            enqueued_analysis_steps: set[int] = set()

            lm_losses: List[float] = []
            align_losses: List[float] = []
            train_history_rows: List[Dict[str, float | str | bool]] = []
            periodic_eval_rows: List[Dict[str, float | str | bool]] = []
            best_checkpoints: List[tuple[float, Path]] = []
            phase_steps_completed: Dict[str, int] = {"phase1": 0, "phase2": 0, "phase3": 0, "control_task_only": 0}
            phase_stop_reasons: Dict[str, str] = {}
            phase_best_align: Dict[str, float] = {}
            global_step = 0
            last_align_loss_per_layer: Dict[str, float] = {}

            for phase_spec in phase_specs:
                phase_name = str(phase_spec["name"])
                phase_steps = int(phase_spec["steps"])
                if phase_steps <= 0:
                    phase_stop_reasons[phase_name] = "skipped"
                    phase_best_align.setdefault(phase_name, float("nan"))
                    continue

                self._unfreeze_all_layers(model)
                frozen_paths: List[str] = []
                if bool(phase_spec["freeze_layers"]):
                    frozen_paths = self._freeze_aligned_layers(model)
                optimizer = self._build_optimizer_for_trainable_params(
                    model=model,
                    learning_rate=float(combo["learning_rate"]),
                )
                aligned_layers_frozen = bool(frozen_paths)
                best_align_mse = float("inf")
                no_improve_evals = 0
                phase_stop_reason = "max_steps"

                self._wandb_log_phase_metrics(
                    combo_idx=combo_idx,
                    phase_name=phase_name,
                    step=global_step,
                    metrics={
                        "phase_start": 1.0,
                        "aligned_layers_frozen": aligned_layers_frozen,
                    },
                    kind="phase_boundary",
                )

                for phase_step in range(1, phase_steps + 1):
                    global_step += 1
                    optimizer.zero_grad(set_to_none=True)
                    self._update_lr(
                        optimizer=optimizer,
                        step=global_step,
                        total_steps=total_planned_steps,
                        base_lr=float(combo["learning_rate"]),
                    )

                    phase_objective = str(phase_spec["objective"])
                    lm_loss_values: List[float] = []
                    align_loss_values: List[float] = []
                    current_align_loss_per_layer: Dict[str, float] = {}

                    if phase_objective == "alignment_only":
                        align_batch_size = int(self.alignment_cfg.get("batch_size", 16))
                        mse_reduction = str(self.alignment_cfg.get("mse_reduction", "mean"))
                        for _ in range(int(self.grad_accum_steps)):
                            sampled_idx = rng.choice(
                                probe_train_idx,
                                size=min(align_batch_size, len(probe_train_idx)),
                                replace=len(probe_train_idx) < align_batch_size,
                            )
                            probe_ids = probe_buffers["input_ids"][sampled_idx].to(device)
                            probe_mask = probe_buffers["attention_mask"][sampled_idx].to(device)
                            acts_by_layer = self._activations_from_batch(
                                model=model,
                                layer_paths=[pair["student_layer"] for pair in layer_pairs],
                                input_ids=probe_ids,
                                attention_mask=probe_mask,
                                device=device,
                                detach=False,
                            )
                            total_weight = 0.0
                            weighted_loss = torch.tensor(0.0, device=device)
                            for pair in layer_pairs:
                                student_layer = str(pair["student_layer"])
                                teacher_layer = str(pair["teacher_layer"])
                                weight = float(pair["weight"])
                                acts = acts_by_layer[student_layer]
                                probe_targets = probe_buffers["targets"][teacher_layer][sampled_idx].to(device)
                                acts, probe_targets = self._require_matching_alignment_width(
                                    acts,
                                    probe_targets,
                                    student_layer=student_layer,
                                    teacher_layer=teacher_layer,
                                )
                                layer_loss = F.mse_loss(
                                    acts.float(),
                                    probe_targets.float(),
                                    reduction=mse_reduction,
                                )
                                current_align_loss_per_layer[student_layer] = float(layer_loss.detach().cpu().item())
                                weighted_loss = weighted_loss + (weight * layer_loss)
                                total_weight += weight
                            if total_weight <= 0.0:
                                raise ValueError("Sum of layer alignment weights must be positive")
                            align_loss = weighted_loss / total_weight
                            (align_loss / float(self.grad_accum_steps)).backward()
                            align_loss_values.append(float(align_loss.detach().cpu().item()))
                    else:
                        for _ in range(int(self.grad_accum_steps)):
                            try:
                                batch = next(train_iter)
                            except StopIteration:
                                train_iter = iter(train_loader)
                                batch = next(train_iter)
                            batch = self._to_device(batch, device)
                            with self._autocast_ctx(device):
                                outputs = model(**batch)
                                lm_loss = self._lm_loss(outputs, batch)
                            lm_loss_weight = float(combo["lm_loss_weight"])
                            if lm_loss_weight != 0.0:
                                ((lm_loss * lm_loss_weight) / float(self.grad_accum_steps)).backward()
                            lm_loss_values.append(float(lm_loss.detach().cpu().item()))

                    if self.gradient_clip_norm and self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                    optimizer.step()

                    mean_lm_loss = float(np.mean(lm_loss_values)) if lm_loss_values else 0.0
                    mean_align_loss = float(np.mean(align_loss_values)) if align_loss_values else 0.0
                    total_loss_value = mean_align_loss if phase_objective == "alignment_only" else mean_lm_loss * float(combo["lm_loss_weight"])
                    current_lr = float(optimizer.param_groups[0]["lr"])
                    lm_losses.append(mean_lm_loss)
                    align_losses.append(mean_align_loss)
                    phase_steps_completed[phase_name] = phase_step
                    if current_align_loss_per_layer:
                        last_align_loss_per_layer = current_align_loss_per_layer

                    if (
                        int(self.train_history_every_n_steps) > 0
                        and (phase_step % int(self.train_history_every_n_steps) == 0 or phase_step == 1 or phase_step == phase_steps)
                    ):
                        self._record_train_row(
                            train_history_rows,
                            global_step=global_step,
                            phase_name=phase_name,
                            phase_step=phase_step,
                            train_lm_loss=mean_lm_loss,
                            train_align_loss=mean_align_loss,
                            train_total_loss=total_loss_value,
                            learning_rate=current_lr,
                            aligned_layers_frozen=aligned_layers_frozen,
                        )
                        self._wandb_log_phase_metrics(
                            combo_idx=combo_idx,
                            phase_name=phase_name,
                            step=global_step,
                            metrics={
                                "global_step": float(global_step),
                                "phase_step": float(phase_step),
                                "train_lm_loss": mean_lm_loss,
                                "train_align_loss": mean_align_loss,
                                "train_total_loss": total_loss_value,
                                "learning_rate": current_lr,
                                "aligned_layers_frozen": aligned_layers_frozen,
                            },
                            kind="train",
                        )

                    eval_every = int(phase_spec["eval_every"])
                    analysis_step_due = global_step in self.analysis_checkpoint_step_set
                    should_eval = (
                        analysis_step_due
                        or (eval_every > 0 and (phase_step % eval_every == 0 or phase_step == phase_steps))
                    )
                    if should_eval:
                        val_loss = self._eval_lm(model, dm.val_dataloader(), device=device)
                        align_eval_mse, align_eval_per_layer = self._eval_alignment(
                            model=model,
                            device=device,
                            probe_buffers=probe_buffers,
                            eval_idx=probe_eval_idx,
                            layer_pairs=layer_pairs,
                        )
                        model.train()
                        eval_row = self._record_eval_row(
                            periodic_eval_rows,
                            global_step=global_step,
                            phase_name=phase_name,
                            phase_step=phase_step,
                            val_loss=val_loss,
                            align_mse=align_eval_mse,
                            align_mse_per_layer=align_eval_per_layer,
                            aligned_layers_frozen=aligned_layers_frozen,
                        )
                        self._wandb_log_phase_metrics(
                            combo_idx=combo_idx,
                            phase_name=phase_name,
                            step=global_step,
                            metrics=eval_row,
                            kind="eval",
                        )

                        save_every = int(self.save_every_n_steps) if self.save_every_n_steps is not None else 0
                        should_save_checkpoint = analysis_step_due or (
                            save_every > 0 and (global_step % save_every == 0 or phase_step == phase_steps)
                        )
                        if should_save_checkpoint:
                            self._save_periodic_checkpoint(
                                model=model,
                                run_output_dir=run_output_dir,
                                step=global_step,
                                eval_loss_for_rank=float(val_loss),
                                best_checkpoints=best_checkpoints,
                            )
                            if analysis_step_due:
                                analysis_ckpt_path = run_output_dir / f"student_analysis_step{int(global_step)}.pt"
                                if analysis_ckpt_path.exists() and global_step not in enqueued_analysis_steps:
                                    queue_entry = self._build_analysis_queue_entry(
                                        combo=combo,
                                        run_output_dir=run_output_dir,
                                        step=global_step,
                                        phase_name=phase_name,
                                        phase_step=phase_step,
                                        probe_buffers=probe_buffers,
                                        probe_source_ids=probe_source_ids_for_analysis,
                                        probe_seed=probe_seed,
                                        data_order_seed=data_order_seed,
                                        dataloader_seed=dataloader_seed,
                                    )
                                    self._append_analysis_queue_entry(analysis_queue_path, queue_entry)
                                    analysis_jobs.append(queue_entry)
                                    enqueued_analysis_steps.add(int(global_step))
                                    if (
                                        self._is_control_task_only()
                                        and control_phase23_budget_step is not None
                                        and int(global_step) == int(control_phase23_budget_step)
                                    ):
                                        control_phase23_analysis_checkpoint_path = str(analysis_ckpt_path)

                        if phase_name == "phase1":
                            best_align_mse = min(best_align_mse, float(align_eval_mse))
                            min_steps = int(phase_spec["min_steps"])
                            patience = int(phase_spec["patience"])
                            min_delta = float(phase_spec["min_delta"])
                            improved = float(align_eval_mse) < (phase_best_align.get(phase_name, float("inf")) - min_delta)
                            if improved:
                                phase_best_align[phase_name] = float(align_eval_mse)
                                no_improve_evals = 0
                            else:
                                no_improve_evals += 1
                            if phase_step >= min_steps and patience > 0 and no_improve_evals >= patience:
                                phase_stop_reason = "align_plateau"
                                break

                phase_best_align.setdefault(phase_name, best_align_mse if math.isfinite(best_align_mse) else float("nan"))
                phase_stop_reasons[phase_name] = phase_stop_reason
                actual_phase_end_step = int(global_step)
                if (
                    bool((self.staged_training_cfg.get("checkpoint_analysis") or {}).get("enabled", False))
                    and actual_phase_end_step > 0
                    and actual_phase_end_step not in enqueued_analysis_steps
                ):
                    analysis_ckpt_path = self._save_analysis_checkpoint(
                        model=model,
                        run_output_dir=run_output_dir,
                        step=actual_phase_end_step,
                    )
                    if analysis_ckpt_path.exists():
                        queue_entry = self._build_analysis_queue_entry(
                            combo=combo,
                            run_output_dir=run_output_dir,
                            step=actual_phase_end_step,
                            phase_name=phase_name,
                            phase_step=int(phase_steps_completed[phase_name]),
                            probe_buffers=probe_buffers,
                            probe_source_ids=probe_source_ids_for_analysis,
                            probe_seed=probe_seed,
                            data_order_seed=data_order_seed,
                            dataloader_seed=dataloader_seed,
                        )
                        self._append_analysis_queue_entry(analysis_queue_path, queue_entry)
                        analysis_jobs.append(queue_entry)
                        enqueued_analysis_steps.add(actual_phase_end_step)
                        if (
                            self._is_control_task_only()
                            and control_phase23_budget_step is not None
                            and int(actual_phase_end_step) == int(control_phase23_budget_step)
                        ):
                            control_phase23_analysis_checkpoint_path = str(analysis_ckpt_path)
                self._wandb_log_phase_metrics(
                    combo_idx=combo_idx,
                    phase_name=phase_name,
                    step=global_step,
                    metrics={
                        "phase_end": 1.0,
                        "phase_steps": float(phase_steps_completed[phase_name]),
                        "phase_stop_reason": phase_stop_reasons[phase_name],
                        "best_align_mse": float(phase_best_align[phase_name]),
                    },
                    kind="phase_boundary",
                )

            val_loss = self._eval_lm(model, dm.val_dataloader(), device=device)
            val_ppl = self._safe_perplexity(val_loss)
            test_loss = self._eval_lm(model, dm.test_dataloader(), device=device)
            test_ppl = self._safe_perplexity(test_loss)
            align_eval_mse, align_eval_per_layer = self._eval_alignment(
                model=model,
                device=device,
                probe_buffers=probe_buffers,
                eval_idx=probe_eval_idx,
                layer_pairs=layer_pairs,
            )

            checkpoint_path = run_output_dir / "student_last.pt"
            torch.save(model.state_dict(), checkpoint_path)

            metrics = {
                "train_lm_loss_last": lm_losses[-1] if lm_losses else float("nan"),
                "train_lm_loss_mean": float(np.mean(lm_losses)) if lm_losses else float("nan"),
                "train_align_loss_last": align_losses[-1] if align_losses else float("nan"),
                "train_align_loss_mean": float(np.mean(align_losses)) if align_losses else float("nan"),
                "train_total_loss_last": float(train_history_rows[-1]["train_total_loss"]) if train_history_rows else float("nan"),
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "test_loss": test_loss,
                "test_perplexity": test_ppl,
                "align_mse": align_eval_mse,
                "effective_global_batch_size": int(effective_batch),
                "lm_loss_weight": float(combo["lm_loss_weight"]),
                "train_example_offset": int(combo["train_example_offset"]),
                "training_regime": self.training_regime,
                "staged_training_enabled": not self._is_control_task_only(),
                "phase1_steps": int(phase_steps_completed["phase1"]),
                "phase2_steps": int(phase_steps_completed["phase2"]),
                "phase3_steps": int(phase_steps_completed["phase3"]),
                "phase1_best_align_mse": float(phase_best_align.get("phase1", float("nan"))),
                "phase1_stop_reason": str(phase_stop_reasons.get("phase1", "")),
                "control_phase23_budget_step": (
                    int(control_phase23_budget_step) if control_phase23_budget_step is not None else None
                ),
                "control_full_budget_step": (
                    int(control_full_budget_step) if control_full_budget_step is not None else None
                ),
                "control_phase23_analysis_checkpoint_path": control_phase23_analysis_checkpoint_path,
            }
            for layer_name in self.student_layer_specs:
                safe = layer_name.replace(".", "__")
                metrics[f"align_mse__{safe}"] = float(align_eval_per_layer.get(layer_name, float("nan")))
                if last_align_loss_per_layer:
                    metrics[f"train_align_loss_last__{safe}"] = float(last_align_loss_per_layer.get(layer_name, float("nan")))

            metrics_path = run_output_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
            periodic_eval_path = run_output_dir / "periodic_eval.json"
            periodic_eval_path.write_text(json.dumps(periodic_eval_rows, indent=2, sort_keys=True), encoding="utf-8")
            train_history_json_path = run_output_dir / "train_history.json"
            train_history_json_path.write_text(json.dumps(train_history_rows, indent=2, sort_keys=True), encoding="utf-8")
            train_history_csv_path = run_output_dir / "train_history.csv"
            fieldnames = sorted({key for row in train_history_rows for key in row.keys()}) if train_history_rows else [
                "global_step",
                "phase",
                "phase_step",
                "train_lm_loss",
                "train_align_loss",
                "train_total_loss",
                "learning_rate",
                "aligned_layers_frozen",
            ]
            with train_history_csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(train_history_rows)

            analysis_index_path = run_output_dir / "analysis_index.json"
            analysis_index_path.write_text(
                json.dumps(analysis_jobs, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            row = {
                "result_schema_version": self.RESULT_SCHEMA_VERSION,
                "family": self.family_name or "unknown",
                "family_architecture": self.family_architecture or "unknown",
                "alignment_side": self.alignment_side or "unknown",
                "layer_scheme": layer_scheme_name,
                "teacher_layer_specs": self._json_list([pair["teacher_layer"] for pair in layer_pairs]),
                "student_layer_specs": self._json_list([pair["student_layer"] for pair in layer_pairs]),
                "layer_loss_weights": self._json_list([float(pair["weight"]) for pair in layer_pairs]),
                "teacher_model": combo["teacher_model_name_or_path"],
                "teacher_model_revision": combo["teacher_model_revision"],
                "student_model": combo["student_model_name_or_path"],
                "student_model_revision": combo["student_model_revision"],
                "tokenizer_name": self.tokenizer_name or combo["student_model_name_or_path"],
                "dataset_name": self.dataset_name,
                "dataset_config": self.dataset_config,
                "dataset_path": self.dataset_path,
                "dataset_revision": self.dataset_revision,
                "text_field": self.text_field,
                "train_split": self.train_split,
                "val_split": self.val_split,
                "test_split": self.test_split,
                "probe_split": self.probe_split,
                "exclude_probe_from_train": bool(self.exclude_probe_from_train),
                "seed": int(combo_seed),
                "seed_model_init": int(model_init_seed),
                "seed_data_order": int(data_order_seed),
                "seed_dataloader": int(dataloader_seed),
                "seed_probe": int(probe_seed),
                "learning_rate": float(combo["learning_rate"]),
                "max_length": int(combo["max_length"]),
                "token_budget": None if combo["token_budget"] is None else int(combo["token_budget"]),
                "train_example_offset": int(combo["train_example_offset"]),
                "training_regime": self.training_regime,
                "max_steps": int(global_step),
                "effective_global_batch_size": int(effective_batch),
                "lm_loss_weight": float(combo["lm_loss_weight"]),
                "val_loss": float(val_loss),
                "val_perplexity": float(val_ppl),
                "test_loss": float(test_loss),
                "test_perplexity": float(test_ppl),
                "align_mse": float(align_eval_mse),
                "train_lm_loss_last": float(metrics["train_lm_loss_last"]),
                "train_lm_loss_mean": float(metrics["train_lm_loss_mean"]),
                "train_align_loss_last": float(metrics["train_align_loss_last"]),
                "train_align_loss_mean": float(metrics["train_align_loss_mean"]),
                "train_total_loss_last": float(metrics["train_total_loss_last"]),
                "num_alignment_layers": int(len(layer_pairs)),
                "probe_size": int(probe_buffers["n"]),
                "probe_ids_path": str(self.probe_ids_path) if self.probe_ids_path is not None else "",
                "teacher_stage": self.teacher_stage,
                "teacher_target_stage": self.teacher_target_stage,
                "teacher_probe_ids_key": self.teacher_probe_ids_key,
                "target_probe_ids_key": self.target_probe_ids_key,
                "run_dir": str(run_output_dir),
                "ckpt_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
                "periodic_eval_path": str(periodic_eval_path),
                "train_history_json_path": str(train_history_json_path),
                "train_history_csv_path": str(train_history_csv_path),
                "staged_training_enabled": not self._is_control_task_only(),
                "phase1_steps": int(phase_steps_completed["phase1"]),
                "phase2_steps": int(phase_steps_completed["phase2"]),
                "phase3_steps": int(phase_steps_completed["phase3"]),
                "phase1_best_align_mse": float(phase_best_align.get("phase1", float("nan"))),
                "phase1_stop_reason": str(phase_stop_reasons.get("phase1", "")),
                "control_phase23_budget_step": (
                    int(control_phase23_budget_step) if control_phase23_budget_step is not None else None
                ),
                "control_full_budget_step": (
                    int(control_full_budget_step) if control_full_budget_step is not None else None
                ),
                "control_phase23_analysis_checkpoint_path": control_phase23_analysis_checkpoint_path,
                "analysis_checkpoint_count": int(len(analysis_jobs)),
                "analysis_index_path": str(analysis_index_path),
                "analysis_queue_path": str(analysis_queue_path),
            }
            for layer_name in self.student_layer_specs:
                safe = layer_name.replace(".", "__")
                row[f"align_mse__{safe}"] = float(align_eval_per_layer.get(layer_name, float("nan")))
            self._wandb_log_phase_metrics(
                combo_idx=combo_idx,
                phase_name="final",
                step=global_step,
                metrics={
                    "final_val_loss": float(val_loss),
                    "final_val_perplexity": float(val_ppl),
                    "final_test_loss": float(test_loss),
                    "final_test_perplexity": float(test_ppl),
                    "final_align_mse": float(align_eval_mse),
                    "phase1_steps": float(phase_steps_completed["phase1"]),
                    "phase2_steps": float(phase_steps_completed["phase2"]),
                    "phase3_steps": float(phase_steps_completed["phase3"]),
                },
                kind="summary",
            )
            return row
        finally:
            self._cleanup_combo_resources(
                trainer_module=trainer_module,
                model=model,
                optimizer=optimizer,
                dm=dm,
                probe_buffers=probe_buffers,
                aligned_targets_for_run=aligned_targets_for_run,
                train_loader=train_loader,
                train_iter=train_iter,
            )

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        teacher_artifacts = context.artifacts.get(self.teacher_stage) or {}
        teacher_target_artifacts = context.artifacts.get(self.teacher_target_stage)
        if teacher_target_artifacts is None:
            raise KeyError(f"No artifacts found for teacher_target_stage '{self.teacher_target_stage}'")

        target_paths_by_layer_raw = teacher_target_artifacts.get(self.target_paths_by_layer_key)
        aligned_targets: Dict[str, np.ndarray] = {}
        if isinstance(target_paths_by_layer_raw, dict) and target_paths_by_layer_raw:
            for teacher_layer in self.teacher_layer_specs:
                path = target_paths_by_layer_raw.get(teacher_layer)
                if path is None:
                    normalized_teacher_layer = self._sanitize_layer_name(teacher_layer)
                    path = target_paths_by_layer_raw.get(normalized_teacher_layer)
                if path is None:
                    raise KeyError(
                        f"Missing aligned target for teacher layer '{teacher_layer}' under "
                        f"'{self.target_paths_by_layer_key}'"
                    )
                aligned_targets[teacher_layer] = self._to_2d(np.load(path))
        else:
            target_primary = teacher_target_artifacts.get(self.target_primary_key)
            if isinstance(target_primary, str) and target_primary:
                target_paths = [Path(target_primary)]
            else:
                target_paths = self._resolve_list_artifact(
                    teacher_target_artifacts.get(self.teacher_target_key)
                )
            if not target_paths:
                raise ValueError(
                    f"No teacher targets found from key '{self.teacher_target_key}' "
                    f"in stage '{self.teacher_target_stage}'"
                )

            if len(self.teacher_layer_specs) == 1:
                aligned_targets[self.teacher_layer_specs[0]] = self._to_2d(
                    np.load(target_paths[self.target_index])
                )
            elif len(target_paths) == len(self.teacher_layer_specs):
                for teacher_layer, path in zip(self.teacher_layer_specs, target_paths):
                    aligned_targets[teacher_layer] = self._to_2d(np.load(path))
            else:
                raise ValueError(
                    "Multi-layer distillation requires either aligned_target_paths_by_layer or "
                    "one target path per teacher layer in aligned_targets"
                )

        teacher_probe_ids = self._load_probe_ids_from_artifacts(
            teacher_artifacts, self.teacher_probe_ids_key
        )
        target_probe_ids = self._load_probe_ids_from_artifacts(
            teacher_target_artifacts, self.target_probe_ids_key
        )
        if teacher_probe_ids and target_probe_ids and teacher_probe_ids != target_probe_ids:
            raise ValueError(
                "Probe ID contract mismatch between teacher and aligned target stages "
                f"({self.teacher_stage}.{self.teacher_probe_ids_key} vs "
                f"{self.teacher_target_stage}.{self.target_probe_ids_key})"
            )
        contract_probe_ids = target_probe_ids or teacher_probe_ids
        target_rows = {layer: target.shape[0] for layer, target in aligned_targets.items()}
        if len(set(target_rows.values())) != 1:
            raise ValueError(f"All aligned targets must share the same row count: {target_rows}")
        n_target_rows = next(iter(target_rows.values()))
        if contract_probe_ids and len(contract_probe_ids) != n_target_rows:
            raise ValueError(
                "Aligned target row count does not match probe ID contract length "
                f"({n_target_rows} vs {len(contract_probe_ids)})"
            )
        if not contract_probe_ids:
            contract_probe_ids = list(range(int(n_target_rows)))
        self.probe_ids_path = self.probe_ids_path or teacher_artifacts.get(self.teacher_probe_ids_key)

        stage_output_dir = stage_dir / self.output_subdir
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        combos = self._expand_sweep_grid(self.sweep, tied_key_groups=self.sweep_tied_keys)
        rows: List[Dict[str, Any]] = []

        for combo_idx, combo_raw in enumerate(combos):
            combo = self._resolve_combo(combo_raw)
            run_name = f"combo_{combo_idx:04d}"
            run_output_dir = stage_output_dir / run_name
            row = self._run_single_combo(
                combo=combo,
                aligned_targets=aligned_targets,
                expected_probe_ids=contract_probe_ids,
                run_output_dir=run_output_dir,
                combo_idx=combo_idx,
            )
            row["combo_index"] = combo_idx
            rows.append(row)

        results_json = stage_output_dir / "sweep_results.json"
        results_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

        results_csv = stage_output_dir / "sweep_results.csv"
        if rows:
            fieldnames = sorted({k for row in rows for k in row.keys()})
            with results_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            results_csv.write_text("", encoding="utf-8")

        return StageResult(
            outputs={
                "stage_output_dir": str(stage_output_dir),
                "sweep_results_json": str(results_json),
                "sweep_results_csv": str(results_csv),
            },
            metadata={
                "num_combinations": len(combos),
                "num_rows": len(rows),
                "teacher_target_key": self.teacher_target_key,
                "artifact_contract": "probe_target_v1",
                "result_schema_version": self.RESULT_SCHEMA_VERSION,
            },
        )
