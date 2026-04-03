#!/usr/bin/env python3
"""Benchmark startup timing for the downstream Pythia distillation pipeline path."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import numpy as np
import torch

from manylatents.data.text import TextDataModule
from manylatents.lightning.hf_trainer import HFTrainerConfig, HFTrainerModule
from manylatents.lightning.hooks import ActivationExtractor, LayerSpec


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Path to JSON timing output")
    parser.add_argument("--dataset-name", default="json")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer-name", default="EleutherAI/pythia-70m")
    parser.add_argument("--teacher-model-name-or-path", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--teacher-model-revision", default="main")
    parser.add_argument("--student-model-name-or-path", default="EleutherAI/pythia-70m")
    parser.add_argument("--student-model-revision", default="main")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--probe-split", default="train")
    parser.add_argument("--probe-ids-path", default=None)
    parser.add_argument("--exclude-probe-from-train", action="store_true")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--probe-size", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=32000)
    parser.add_argument("--train-example-offset", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-path", default="transformer.h[-2]")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--split-index-cache-dir", default=None)
    return parser.parse_args()


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_ctx(device: torch.device, precision: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)


def _make_datamodule(args: argparse.Namespace) -> TextDataModule:
    return TextDataModule(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_path=args.dataset_path,
        dataset_revision=args.dataset_revision,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        probe_split=args.probe_split,
        token_budget=args.token_budget,
        train_example_offset=args.train_example_offset,
        probe_ids_path=args.probe_ids_path,
        exclude_probe_from_train=args.exclude_probe_from_train,
        text_field=args.text_field,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        batch_size=args.micro_batch_size,
        probe_n_samples=args.probe_size,
        seed=args.seed,
        data_order_seed=args.seed,
        dataloader_seed=args.seed,
        split_index_cache_dir=args.split_index_cache_dir,
    )


def _teacher_probe_benchmark(
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    started = perf_counter()

    dm = _make_datamodule(args)
    t0 = perf_counter()
    dm.setup(stage="probe")
    timings["probe_datamodule_setup_seconds"] = perf_counter() - t0

    t0 = perf_counter()
    probe_loader = dm.probe_dataloader()
    timings["probe_dataloader_init_seconds"] = perf_counter() - t0

    teacher_cfg = HFTrainerConfig(
        model_name_or_path=args.teacher_model_name_or_path,
        model_revision=args.teacher_model_revision,
        init_mode="pretrained",
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.teacher_model_revision,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else None,
    )
    teacher = HFTrainerModule(teacher_cfg)
    t0 = perf_counter()
    teacher.configure_model()
    timings["teacher_model_configure_seconds"] = perf_counter() - t0

    network = teacher.network
    if network is None:
        raise RuntimeError("Teacher network was not initialized")

    t0 = perf_counter()
    network = network.to(device)
    network.eval()
    timings["teacher_model_to_device_seconds"] = perf_counter() - t0

    extractor = ActivationExtractor([LayerSpec(path=args.layer_path, reduce="mean")])
    batch_count = 0
    token_positions = 0
    t0 = perf_counter()
    with torch.no_grad():
        with extractor.capture(network):
            for batch in probe_loader:
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                token_positions += int(inputs["input_ids"].numel())
                with _autocast_ctx(device, args.precision):
                    network(**inputs)
                batch_count += 1
    timings["teacher_probe_forward_seconds"] = perf_counter() - t0
    timings["probe_teacher_total_seconds"] = perf_counter() - started

    acts = extractor.get_activations()
    activation_shape = tuple(acts[args.layer_path].shape) if args.layer_path in acts else None
    return {
        "timings": timings,
        "probe_batches": int(batch_count),
        "probe_examples": int(len(dm.probe_dataset)) if dm.probe_dataset is not None else 0,
        "probe_token_positions": int(token_positions),
        "activation_shape": activation_shape,
    }


def _first_train_step_benchmark(
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    started = perf_counter()

    dm = _make_datamodule(args)
    t0 = perf_counter()
    dm.setup()
    timings["train_datamodule_setup_seconds"] = perf_counter() - t0

    t0 = perf_counter()
    train_loader = dm.train_dataloader()
    timings["train_dataloader_init_seconds"] = perf_counter() - t0

    student_cfg = HFTrainerConfig(
        model_name_or_path=args.student_model_name_or_path,
        model_revision=args.student_model_revision,
        init_mode="from_config",
        model_config_name_or_path=args.student_model_name_or_path,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.student_model_revision,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else None,
    )
    student = HFTrainerModule(student_cfg)
    t0 = perf_counter()
    student.configure_model()
    timings["student_model_configure_seconds"] = perf_counter() - t0

    network = student.network
    if network is None:
        raise RuntimeError("Student network was not initialized")

    t0 = perf_counter()
    network = network.to(device)
    network.train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4)
    timings["student_model_to_device_and_optimizer_seconds"] = perf_counter() - t0

    train_iter = iter(train_loader)
    t0 = perf_counter()
    batch = next(train_iter)
    timings["first_train_batch_fetch_seconds"] = perf_counter() - t0

    batch = {k: v.to(device) for k, v in batch.items()}
    token_positions = int(batch["input_ids"].numel())
    t0 = perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with _autocast_ctx(device, args.precision):
        outputs = network(**batch)
        loss = outputs.loss
    loss.backward()
    optimizer.step()
    timings["first_train_step_seconds"] = perf_counter() - t0
    timings["time_to_first_train_step_seconds"] = perf_counter() - started

    return {
        "timings": timings,
        "train_examples": int(len(dm.train_dataset)) if dm.train_dataset is not None else 0,
        "val_examples": int(len(dm.val_dataset)) if dm.val_dataset is not None else 0,
        "test_examples": int(len(dm.test_dataset)) if dm.test_dataset is not None else 0,
        "train_token_positions_first_step": int(token_positions),
        "first_train_loss": float(loss.detach().cpu().item()),
    }


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = _select_device()

    payload = {
        "config": {
            "dataset_path": args.dataset_path,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "probe_split": args.probe_split,
            "probe_size": int(args.probe_size),
            "token_budget": int(args.token_budget),
            "max_length": int(args.max_length),
            "micro_batch_size": int(args.micro_batch_size),
            "seed": int(args.seed),
            "device": str(device),
            "precision": args.precision,
        },
        "probe_teacher": _teacher_probe_benchmark(args, device=device),
        "first_train_step": _first_train_step_benchmark(args, device=device),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
