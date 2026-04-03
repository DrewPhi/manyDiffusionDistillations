#!/usr/bin/env python3
"""Prefetch Hugging Face assets needed by a distillation run.

This script is meant to be idempotent:
- if required assets are already in the local HF cache, it exits quickly
- if assets are missing, it downloads them once

It supports either explicit model arguments or a materialized run-spec JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoConfig, AutoTokenizer


@dataclass(frozen=True)
class RepoSpec:
    repo_id: str
    revision: str | None
    trust_remote_code: bool


TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "*.tiktoken",
    "chat_template*",
    "config.json",
]

CONFIG_PATTERNS = [
    "config.json",
    "generation_config.json",
    "*.py",
]

FULL_MODEL_PATTERNS = [
    "*.json",
    "*.py",
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model*.bin.index.json",
    "model*.safetensors",
    "model*.bin",
    "*.model",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "sentencepiece.bpe.model",
    "spiece.model",
    "chat_template*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, help="Materialized run-spec JSON.")

    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-revision")
    parser.add_argument("--teacher-trust-remote-code", action="store_true")

    parser.add_argument("--student-model")
    parser.add_argument("--student-revision")
    parser.add_argument("--student-trust-remote-code", action="store_true")
    parser.add_argument("--student-init-from-scratch", action="store_true")

    parser.add_argument("--tokenizer-name")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--tokenizer-trust-remote-code", action="store_true")
    return parser.parse_args()


@contextmanager
def _online_hf_access() -> Iterable[None]:
    previous = {
        "HF_HUB_OFFLINE": os.environ.pop("HF_HUB_OFFLINE", None),
        "TRANSFORMERS_OFFLINE": os.environ.pop("TRANSFORMERS_OFFLINE", None),
    }
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _read_run_spec(path: Path) -> tuple[RepoSpec, RepoSpec, bool, RepoSpec]:
    run_spec = json.loads(path.read_text(encoding="utf-8"))
    repro = run_spec["reproducibility"]
    teacher = repro["teacher"]
    student = repro["student"]
    teacher_spec = RepoSpec(
        repo_id=str(teacher["model_name_or_path"]),
        revision=teacher.get("model_revision"),
        trust_remote_code=bool(teacher.get("trust_remote_code", False)),
    )
    student_spec = RepoSpec(
        repo_id=str(student["model_name_or_path"]),
        revision=student.get("model_revision"),
        trust_remote_code=bool(student.get("trust_remote_code", False)),
    )
    tokenizer_spec = RepoSpec(
        repo_id=str(student["tokenizer_name"]),
        revision=student.get("model_revision"),
        trust_remote_code=bool(student.get("trust_remote_code", False)),
    )
    return teacher_spec, student_spec, bool(student.get("init_from_scratch", False)), tokenizer_spec


def _read_explicit_args(args: argparse.Namespace) -> tuple[RepoSpec, RepoSpec, bool, RepoSpec]:
    if not (args.teacher_model and args.student_model and args.tokenizer_name):
        raise SystemExit(
            "Explicit mode requires --teacher-model, --student-model, and --tokenizer-name"
        )
    teacher_spec = RepoSpec(
        repo_id=args.teacher_model,
        revision=args.teacher_revision,
        trust_remote_code=bool(args.teacher_trust_remote_code),
    )
    student_spec = RepoSpec(
        repo_id=args.student_model,
        revision=args.student_revision,
        trust_remote_code=bool(args.student_trust_remote_code),
    )
    tokenizer_spec = RepoSpec(
        repo_id=args.tokenizer_name,
        revision=args.tokenizer_revision or args.student_revision,
        trust_remote_code=bool(args.tokenizer_trust_remote_code or args.student_trust_remote_code),
    )
    return teacher_spec, student_spec, bool(args.student_init_from_scratch), tokenizer_spec


def _ensure_tokenizer(spec: RepoSpec) -> str:
    try:
        AutoTokenizer.from_pretrained(
            spec.repo_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
        )
        return "cached"
    except Exception:
        with _online_hf_access():
            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                repo_type="model",
                allow_patterns=TOKENIZER_PATTERNS,
            )
        return "downloaded"


def _ensure_config(spec: RepoSpec) -> str:
    try:
        AutoConfig.from_pretrained(
            spec.repo_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
        )
        return "cached"
    except Exception:
        with _online_hf_access():
            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                repo_type="model",
                allow_patterns=CONFIG_PATTERNS,
            )
        return "downloaded"


def _ensure_model_snapshot(spec: RepoSpec) -> str:
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            repo_type="model",
            allow_patterns=FULL_MODEL_PATTERNS,
            local_files_only=True,
        )
        return "cached"
    except LocalEntryNotFoundError:
        with _online_hf_access():
            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                repo_type="model",
                allow_patterns=FULL_MODEL_PATTERNS,
            )
        return "downloaded"


def _print_status(label: str, spec: RepoSpec, status: str) -> None:
    revision = spec.revision if spec.revision is not None else "default"
    print(f"[hf-cache] {label}: {spec.repo_id} @ {revision} -> {status}")


def main() -> None:
    args = parse_args()
    if args.run_spec is not None:
        teacher_spec, student_spec, student_init_from_scratch, tokenizer_spec = _read_run_spec(args.run_spec)
    else:
        teacher_spec, student_spec, student_init_from_scratch, tokenizer_spec = _read_explicit_args(args)

    status = _ensure_tokenizer(tokenizer_spec)
    _print_status("tokenizer", tokenizer_spec, status)

    status = _ensure_config(student_spec)
    _print_status("student-config", student_spec, status)

    status = _ensure_model_snapshot(teacher_spec)
    _print_status("teacher-model", teacher_spec, status)

    if not student_init_from_scratch:
        status = _ensure_model_snapshot(student_spec)
        _print_status("student-model", student_spec, status)
    else:
        print(f"[hf-cache] student-model: {student_spec.repo_id} -> skipped (init_from_scratch=true)")


if __name__ == "__main__":
    main()
