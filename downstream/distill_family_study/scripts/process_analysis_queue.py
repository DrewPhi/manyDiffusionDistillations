#!/usr/bin/env python3
"""Process detached student-analysis checkpoints and log geometry artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manylatents.algorithms.latent.phate import PHATEModule
from manylatents.callbacks.diffusion_operator import build_diffusion_operator
from manylatents.data.text import TextDataModule
from manylatents.lightning.hf_trainer import HFTrainerConfig, HFTrainerModule
from manylatents.pipeline.stages.distillation_sweep import DistillationSweepStage

try:
    import wandb
except (ImportError, AttributeError):  # pragma: no cover
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.0)
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser.parse_args()


def _read_queue(queue_path: Path) -> list[dict[str, Any]]:
    if not queue_path.exists():
        raise FileNotFoundError(f"Queue path does not exist: {queue_path}")
    entries: list[dict[str, Any]] = []
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _status_path(entry: dict[str, Any]) -> Path:
    run_dir = Path(entry["run_dir"])
    return run_dir / f"analysis_status_step{int(entry['step'])}.json"


def _lock_path(entry: dict[str, Any]) -> Path:
    run_dir = Path(entry["run_dir"])
    return run_dir / f"analysis_status_step{int(entry['step'])}.lock"


def _result_dir(entry: dict[str, Any]) -> Path:
    run_dir = Path(entry["run_dir"])
    outdir = run_dir / "analysis_outputs" / f"step_{int(entry['step']):07d}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _read_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _build_datamodule(entry: dict[str, Any]) -> TextDataModule:
    dm = TextDataModule(
        dataset_name=entry["dataset_name"],
        dataset_config=entry.get("dataset_config"),
        dataset_path=entry.get("dataset_path"),
        dataset_revision=entry.get("dataset_revision"),
        train_split=entry.get("train_split", "train"),
        val_split=entry.get("val_split", "validation"),
        test_split=entry.get("test_split", "test"),
        probe_split=entry["probe_split"],
        token_budget=None,
        train_example_offset=0,
        probe_ids_path=entry.get("probe_ids_path"),
        exclude_probe_from_train=bool(entry.get("exclude_probe_from_train", False)),
        text_field=entry.get("text_field", "text"),
        lm_objective=entry["student_model_family"],
        tokenizer_name=entry["tokenizer_name"],
        max_length=int(entry["max_length"]),
        batch_size=int(entry["micro_batch_size"]),
        probe_n_samples=int(entry["probe_n_samples"]),
        seed=int(entry["probe_seed"]),
        data_order_seed=int(entry["data_order_seed"]),
        dataloader_seed=int(entry["dataloader_seed"]),
    )
    dm.setup(stage="probe")
    return dm


def _load_model(entry: dict[str, Any], stage: DistillationSweepStage) -> torch.nn.Module:
    hf_cfg = HFTrainerConfig(
        model_name_or_path=entry["student_model"],
        model_revision=entry.get("student_model_revision"),
        init_mode="from_config",
        model_config_name_or_path=entry.get("student_model_config_name_or_path") or entry["student_model"],
        model_config_overrides=dict(entry.get("student_model_config_overrides") or {}),
        tokenizer_name=entry["tokenizer_name"],
        tokenizer_revision=entry.get("student_model_revision"),
        learning_rate=3e-4,
        weight_decay=0.0,
        warmup_steps=0,
        adam_epsilon=1e-8,
        trust_remote_code=bool(entry.get("student_trust_remote_code", False)),
        torch_dtype=stage._precision_to_dtype(),
        model_family=entry["student_model_family"],
    )
    trainer = HFTrainerModule(hf_cfg)
    trainer.configure_model()
    model = trainer.network
    assert model is not None
    state_dict = torch.load(entry["checkpoint_path"], map_location="cpu")
    model.load_state_dict(state_dict)
    device = stage._select_device()
    return model.to(device).eval()


def _extract_activations(
    *,
    entry: dict[str, Any],
    dm: TextDataModule,
    model: torch.nn.Module,
) -> dict[str, np.ndarray]:
    stage = DistillationSweepStage(
        stage_name="analysis_worker",
        student_layer_specs=entry["student_layer_specs"],
        alignment_side=entry.get("alignment_side"),
        precision=entry.get("precision", "bf16"),
        micro_batch_size=int(entry["micro_batch_size"]),
    )
    loader = dm.probe_dataloader() if hasattr(dm, "probe_dataloader") else dm.val_dataloader()
    device = stage._select_device()
    activations: dict[str, list[np.ndarray]] = {layer: [] for layer in entry["student_layer_specs"]}
    for batch in loader:
        batch = stage._to_device(batch, device)
        acts = stage._activations_from_batch(
            model=model,
            layer_paths=entry["student_layer_specs"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            device=device,
            detach=True,
        )
        for layer_name, layer_acts in acts.items():
            activations[layer_name].append(layer_acts.detach().float().cpu().numpy())
    return {
        layer_name: np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)
        for layer_name, chunks in activations.items()
    }


def _phate_2d(diffusion_operator: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(diffusion_operator).float()
    phate = PHATEModule(n_components=2, knn=35, t="auto", decay=40, gamma=1.0)
    coords = phate.fit_transform(x)
    return coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)


def _log_to_wandb(entry: dict[str, Any], layer_name: str, coords: np.ndarray, image_path: Path) -> None:
    if wandb is None or not entry.get("wandb_project"):
        return
    if wandb.run is None:
        run_name = entry.get("wandb_parent_run_name") or Path(entry["run_dir"]).name
        wandb.init(
            project=entry["wandb_project"],
            entity=entry.get("wandb_entity"),
            name=f"{run_name}-analysis",
            group=entry.get("wandb_parent_run_id") or run_name,
            job_type="analysis",
            reinit=True,
        )
    table = wandb.Table(columns=["probe_id", "x", "y", "step", "phase", "layer"])
    probe_ids = list(entry.get("probe_source_ids") or [])
    if len(probe_ids) != len(coords):
        probe_ids = list(range(len(coords)))
    for probe_id, (x, y) in zip(probe_ids, coords):
        table.add_data(int(probe_id), float(x), float(y), int(entry["step"]), str(entry["phase"]), str(layer_name))
    safe = layer_name.replace(".", "_").replace("[", "_").replace("]", "_")
    wandb.log(
        {
            f"analysis/{safe}/phate_table": table,
            f"analysis/{safe}/phate_image": wandb.Image(str(image_path)),
        },
        step=int(entry["step"]),
    )


def _plot_coords(coords: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("PHATE-1")
    ax.set_ylabel("PHATE-2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _process_entry(entry: dict[str, Any]) -> dict[str, Any]:
    dm = _build_datamodule(entry)
    stage = DistillationSweepStage(
        stage_name="analysis_worker",
        student_layer_specs=entry["student_layer_specs"],
        alignment_side=entry.get("alignment_side"),
        precision=entry.get("precision", "bf16"),
        micro_batch_size=int(entry["micro_batch_size"]),
    )
    model = _load_model(entry, stage)
    activations = _extract_activations(entry=entry, dm=dm, model=model)
    outdir = _result_dir(entry)

    layer_results = []
    for layer_name, acts in activations.items():
        safe = layer_name.replace(".", "_").replace("[", "_").replace("]", "_")
        acts_path = outdir / f"{safe}__activations.npy"
        diff_path = outdir / f"{safe}__diffop.npy"
        phate_path = outdir / f"{safe}__phate2d.npy"
        image_path = outdir / f"{safe}__phate2d.png"
        np.save(acts_path, acts)
        diff_op = build_diffusion_operator(acts, method="diffusion", knn=35, alpha=1.0, symmetric=False, metric="euclidean")
        np.save(diff_path, diff_op)
        coords = _phate_2d(diff_op)
        np.save(phate_path, coords)
        _plot_coords(coords, image_path, title=f"{layer_name} step {entry['step']}")
        _log_to_wandb(entry, layer_name, coords, image_path)
        layer_results.append(
            {
                "layer": layer_name,
                "activations_path": str(acts_path),
                "diffusion_operator_path": str(diff_path),
                "phate_2d_path": str(phate_path),
                "phate_image_path": str(image_path),
            }
        )

    return {
        "status": "done",
        "step": int(entry["step"]),
        "phase": entry["phase"],
        "run_dir": entry["run_dir"],
        "checkpoint_path": entry["checkpoint_path"],
        "layers": layer_results,
    }


def main() -> None:
    args = parse_args()
    processed = 0
    while True:
        entries = _read_queue(args.queue_path)
        made_progress = False
        for entry in entries:
            if args.max_jobs is not None and processed >= args.max_jobs:
                break
            status_path = _status_path(entry)
            lock_path = _lock_path(entry)
            existing_status = _read_status(status_path)
            if existing_status is not None and existing_status.get("status") == "done":
                continue
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.close(fd)
            try:
                running_payload = {
                    "status": "running",
                    "step": int(entry["step"]),
                    "phase": entry["phase"],
                    "run_dir": entry["run_dir"],
                    "checkpoint_path": entry["checkpoint_path"],
                }
                status_path.write_text(json.dumps(running_payload, indent=2, sort_keys=True), encoding="utf-8")
                result = _process_entry(entry)
                status_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
                processed += 1
                made_progress = True
            except Exception as exc:
                failure_payload = {
                    "status": "failed",
                    "step": int(entry["step"]),
                    "phase": entry["phase"],
                    "run_dir": entry["run_dir"],
                    "checkpoint_path": entry["checkpoint_path"],
                    "error": repr(exc),
                }
                status_path.write_text(json.dumps(failure_payload, indent=2, sort_keys=True), encoding="utf-8")
                made_progress = True
            finally:
                if lock_path.exists():
                    lock_path.unlink()
        if args.max_jobs is not None and processed >= args.max_jobs:
            break
        if args.poll_seconds <= 0:
            break
        if not made_progress:
            time.sleep(float(args.poll_seconds))
    if wandb is not None and wandb.run is not None:
        wandb.finish()
    print(f"Processed {processed} analysis job(s) from {args.queue_path}")


if __name__ == "__main__":
    main()
