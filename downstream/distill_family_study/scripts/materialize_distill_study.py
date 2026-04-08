#!/usr/bin/env python3
"""Materialize a declarative distillation study into canonical run manifests.

This is the bridge between the publication study config and the eventual
launcher. It expands families, students, layer schemes, and seeds into one row
per run plus the Hydra overrides needed to execute that run.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf


def _to_plain(value: Any) -> Any:
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _json_override(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _scalar_override(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _default_model_family(family_name: str, family_cfg: dict[str, Any]) -> str:
    explicit = family_cfg.get("model_family")
    if explicit is not None:
        return str(explicit)
    architecture = str(family_cfg.get("architecture", "")).lower()
    if architecture == "seq2seq_lm" or architecture == "encoder_decoder" or family_name.lower() == "t5":
        return "seq2seq_lm"
    return "causal_lm"


def _compute_total_steps(shared: dict[str, Any]) -> int | None:
    max_steps = shared["training"].get("max_steps")
    if max_steps is not None:
        return int(max_steps)
    token_budget = shared.get("token_budget")
    if token_budget is None:
        return None
    token_budget = int(token_budget)
    global_batch_size = int(shared["training"]["global_batch_size"])
    max_length = int(shared["max_length"])
    tokens_per_step = max(1, global_batch_size * max_length)
    return max(1, token_budget // tokens_per_step)


def _resolve_probe_size(shared: dict[str, Any], student_cfg: dict[str, Any]) -> int:
    probe_cfg = shared["probe"]
    fixed_size = probe_cfg.get("size")
    if fixed_size is not None:
        fixed_size = int(fixed_size)
        if fixed_size <= 0:
            raise ValueError("study.shared.probe.size must be > 0 when provided")
        return fixed_size

    size_multiplier = int(probe_cfg["size_multiplier"])
    if size_multiplier <= 0:
        raise ValueError("study.shared.probe.size_multiplier must be > 0")
    return int(student_cfg["penultimate_dim"]) * size_multiplier


@dataclass(frozen=True)
class StudyRun:
    study_name: str
    run_name: str
    family: str
    family_config: str
    teacher_model: str
    teacher_revision: str
    teacher_trust_remote_code: bool
    student_key: str
    student_model: str
    student_revision: str
    student_trust_remote_code: bool
    student_init_from_scratch: bool
    student_penultimate_dim: int
    student_model_config_name_or_path: Optional[str]
    student_model_config_overrides: Optional[dict[str, Any]]
    tokenizer_name: str
    layer_scheme: str
    training_regime: str
    staged_training_enabled: bool
    lm_loss_weight: float
    seed: int
    token_budget: Optional[int]
    max_length: int
    train_example_offset: int
    probe_size: int
    reproducibility: dict[str, Any]
    hydra_overrides: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "run_name": self.run_name,
            "family": self.family,
            "family_config": self.family_config,
            "teacher_model": self.teacher_model,
            "teacher_revision": self.teacher_revision,
            "teacher_trust_remote_code": self.teacher_trust_remote_code,
            "student_key": self.student_key,
            "student_model": self.student_model,
            "student_revision": self.student_revision,
            "student_trust_remote_code": self.student_trust_remote_code,
            "student_init_from_scratch": self.student_init_from_scratch,
            "student_penultimate_dim": self.student_penultimate_dim,
            "student_model_config_name_or_path": self.student_model_config_name_or_path,
            "student_model_config_overrides": self.student_model_config_overrides,
            "tokenizer_name": self.tokenizer_name,
            "layer_scheme": self.layer_scheme,
            "training_regime": self.training_regime,
            "staged_training_enabled": self.staged_training_enabled,
            "lm_loss_weight": self.lm_loss_weight,
            "seed": self.seed,
            "token_budget": self.token_budget,
            "max_length": self.max_length,
            "train_example_offset": self.train_example_offset,
            "probe_size": self.probe_size,
            "reproducibility": self.reproducibility,
            "hydra_overrides": self.hydra_overrides,
        }


def _build_run(
    study_name: str,
    family_name: str,
    family_cfg: dict[str, Any],
    shared: dict[str, Any],
    student_cfg: dict[str, Any],
    layer_scheme: str,
    training_regime: str,
    seed: int,
) -> StudyRun:
    lm_loss_weight = float(shared["lm_loss_weight"])
    token_budget_value = shared.get("token_budget")
    token_budget = None if token_budget_value is None else int(token_budget_value)
    max_length = int(shared["max_length"])
    train_example_offset = int(shared["train_example_offset"])
    probe_size = _resolve_probe_size(shared, student_cfg)
    total_steps = _compute_total_steps(shared)
    model_family = _default_model_family(family_name, family_cfg)
    staged_training = dict(shared["training"].get("staged_training") or {})
    runtime_staged_training_enabled = bool(staged_training.get("enabled", True))
    staged_training_enabled = runtime_staged_training_enabled and training_regime == "staged"
    run_name = (
        f"{study_name}_"
        f"{family_name}_"
        f"{student_cfg['key']}_"
        f"{layer_scheme}_"
        f"{training_regime}_"
        f"seed{seed}"
    )

    hydra_overrides = [
        "experiment=distill_family_study_template",
        f"family={family_cfg['family_config']}",
        f"layer_scheme={layer_scheme}",
        f"name={run_name}",
        f"output_dir={shared['output_dir']}",
        f"logger={shared['logger']}",
        f"stage_pipeline.params.reusable_artifacts.probe_teacher_manifest_path={_scalar_override(shared['reusable_artifacts']['probe_teacher_manifest_path'])}",
        f"stage_pipeline.params.reusable_artifacts.phate_teacher_target_manifest_path={_scalar_override(shared['reusable_artifacts']['phate_teacher_target_manifest_path'])}",
        f"stage_pipeline.params.teacher.model_name_or_path={family_cfg['teacher']['model_name_or_path']}",
        f"stage_pipeline.params.teacher.model_revision={family_cfg['teacher']['model_revision']}",
        f"stage_pipeline.params.teacher.trust_remote_code={_scalar_override(family_cfg['teacher']['trust_remote_code'])}",
        f"stage_pipeline.params.student.model_name_or_path={student_cfg['model_name_or_path']}",
        f"stage_pipeline.params.student.model_revision={student_cfg['model_revision']}",
        f"stage_pipeline.params.student.trust_remote_code={_scalar_override(student_cfg['trust_remote_code'])}",
        f"stage_pipeline.params.student.model_family={model_family}",
        f"stage_pipeline.params.student.init_from_scratch={_scalar_override(student_cfg['init_from_scratch'])}",
        f"stage_pipeline.params.student.penultimate_dim={student_cfg['penultimate_dim']}",
        f"stage_pipeline.params.data.tokenizer_name={student_cfg['tokenizer_name']}",
        f"stage_pipeline.params.data.lm_objective={model_family}",
        f"stage_pipeline.params.data.max_length={max_length}",
        f"stage_pipeline.params.data.token_budget={_scalar_override(token_budget)}",
        f"stage_pipeline.params.data.train_example_offset={train_example_offset}",
        f"stage_pipeline.params.probe.size={probe_size}",
        f"stage_pipeline.params.probe.train_fraction={shared['probe']['train_fraction']}",
        f"stage_pipeline.params.probe.eval_fraction={shared['probe']['eval_fraction']}",
        f"stage_pipeline.params.probe.sampling={shared['probe']['sampling']}",
        f"stage_pipeline.params.probe.save_ids={_scalar_override(shared['probe']['save_ids'])}",
        f"stage_pipeline.params.phate.n_components={student_cfg['penultimate_dim']}",
        f"stage_pipeline.params.phate.knn={shared['phate']['knn']}",
        f"stage_pipeline.params.phate.t={shared['phate']['t']}",
        f"stage_pipeline.params.phate.decay={shared['phate']['decay']}",
        f"stage_pipeline.params.phate.gamma={shared['phate']['gamma']}",
        f"stage_pipeline.params.phate.fit_fraction={shared['phate']['fit_fraction']}",
        f"stage_pipeline.params.training.precision={shared['training']['precision']}",
        f"stage_pipeline.params.training.global_batch_size={shared['training']['global_batch_size']}",
        f"stage_pipeline.params.training.micro_batch_size={shared['training']['micro_batch_size']}",
        f"stage_pipeline.params.training.grad_accum_steps={shared['training']['grad_accum_steps']}",
        f"stage_pipeline.params.training.max_steps={_scalar_override(shared['training']['max_steps'])}",
        f"stage_pipeline.params.training.eval_every_n_steps={shared['training']['eval_every_n_steps']}",
        f"stage_pipeline.params.training.save_every_n_steps={shared['training']['save_every_n_steps']}",
        f"stage_pipeline.params.training.save_top_k={shared['training']['save_top_k']}",
        f"stage_pipeline.params.training.analysis_checkpoint_steps={_json_override(shared['training'].get('analysis_checkpoint_steps', []))}",
        f"stage_pipeline.params.training.gradient_clip_norm={shared['training']['gradient_clip_norm']}",
        f"stage_pipeline.params.training.train_history_every_n_steps={shared['training']['train_history_every_n_steps']}",
        f"stage_pipeline.params.training.lm_loss_weight={lm_loss_weight}",
        f"stage_pipeline.params.training.training_regime={training_regime}",
        f"stage_pipeline.params.training.staged_training.enabled={_scalar_override(runtime_staged_training_enabled)}",
        f"stage_pipeline.params.optimizer.name={shared['optimizer']['name']}",
        f"stage_pipeline.params.optimizer.learning_rate={shared['optimizer']['learning_rate']}",
        f"stage_pipeline.params.optimizer.betas={_json_override(shared['optimizer']['betas'])}",
        f"stage_pipeline.params.optimizer.eps={shared['optimizer']['eps']}",
        f"stage_pipeline.params.optimizer.weight_decay={shared['optimizer']['weight_decay']}",
        f"stage_pipeline.params.lr_scheduler.type={shared['lr_scheduler']['type']}",
        f"stage_pipeline.params.lr_scheduler.warmup_steps={shared['lr_scheduler']['warmup_steps']}",
        f"stage_pipeline.params.lr_scheduler.min_lr={shared['lr_scheduler']['min_lr']}",
        f"stage_pipeline.params.regularization.dropout={shared['regularization']['dropout']}",
        f"stage_pipeline.params.regularization.label_smoothing={shared['regularization']['label_smoothing']}",
        f"stage_pipeline.params.alignment.loss={shared['alignment']['loss']}",
        f"stage_pipeline.params.alignment.mse_reduction={shared['alignment']['mse_reduction']}",
        f"stage_pipeline.params.alignment.batch_size={shared['alignment']['batch_size']}",
        f"stage_pipeline.params.alignment.eval_batch_size={shared['alignment']['eval_batch_size']}",
        f"stage_pipeline.params.alignment.sample_every_n_steps={shared['alignment']['sample_every_n_steps']}",
        f"stage_pipeline.params.seeds.global_seed={seed}",
        f"stage_pipeline.params.seeds.model_init_seed={_scalar_override(shared['seeds']['model_init_seed'])}",
        f"stage_pipeline.params.seeds.data_order_seed={_scalar_override(shared['seeds']['data_order_seed'])}",
        f"stage_pipeline.params.seeds.dataloader_seed={_scalar_override(shared['seeds']['dataloader_seed'])}",
        f"stage_pipeline.params.seeds.phate_seed={_scalar_override(shared['seeds']['phate_seed'])}",
        f"stage_pipeline.params.sweep.teacher_model={_json_override([family_cfg['teacher']['model_name_or_path']])}",
        f"stage_pipeline.params.sweep.student_model={_json_override([student_cfg['model_name_or_path']])}",
        f"stage_pipeline.params.sweep.seed={_json_override([seed])}",
        f"stage_pipeline.params.sweep.lm_loss_weight={_json_override([lm_loss_weight])}",
        f"stage_pipeline.params.sweep.learning_rate={_json_override([shared['optimizer']['learning_rate']])}",
        f"stage_pipeline.params.sweep.max_length={_json_override([max_length])}",
        f"stage_pipeline.params.sweep.token_budget={_json_override([token_budget])}",
        f"stage_pipeline.params.sweep.train_example_offset={_json_override([train_example_offset])}",
    ]

    phase1_cfg = dict(staged_training.get("phase1") or {})
    phase2_cfg = dict(staged_training.get("phase2") or {})
    phase3_cfg = dict(staged_training.get("phase3") or {})
    checkpoint_analysis_cfg = dict(staged_training.get("checkpoint_analysis") or {})
    hydra_overrides.extend(
        [
            f"stage_pipeline.params.training.staged_training.phase1.objective={_scalar_override(phase1_cfg.get('objective'))}",
            f"stage_pipeline.params.training.staged_training.phase1.min_steps={_scalar_override(phase1_cfg.get('min_steps'))}",
            f"stage_pipeline.params.training.staged_training.phase1.max_steps={_scalar_override(phase1_cfg.get('max_steps'))}",
            f"stage_pipeline.params.training.staged_training.phase1.eval_every_n_steps={_scalar_override(phase1_cfg.get('eval_every_n_steps'))}",
            f"stage_pipeline.params.training.staged_training.phase1.early_stop_patience={_scalar_override(phase1_cfg.get('early_stop_patience'))}",
            f"stage_pipeline.params.training.staged_training.phase1.early_stop_min_delta={_scalar_override(phase1_cfg.get('early_stop_min_delta'))}",
            f"stage_pipeline.params.training.staged_training.phase2.objective={_scalar_override(phase2_cfg.get('objective'))}",
            f"stage_pipeline.params.training.staged_training.phase2.max_steps={_scalar_override(phase2_cfg.get('max_steps'))}",
            f"stage_pipeline.params.training.staged_training.phase3.objective={_scalar_override(phase3_cfg.get('objective'))}",
            f"stage_pipeline.params.training.staged_training.phase3.max_steps={_scalar_override(phase3_cfg.get('max_steps'))}",
            f"stage_pipeline.params.training.staged_training.checkpoint_analysis.enabled={_scalar_override(checkpoint_analysis_cfg.get('enabled'))}",
            f"stage_pipeline.params.training.staged_training.checkpoint_analysis.phase1_snapshots={_scalar_override(checkpoint_analysis_cfg.get('phase1_snapshots'))}",
            f"stage_pipeline.params.training.staged_training.checkpoint_analysis.phase2_snapshots={_scalar_override(checkpoint_analysis_cfg.get('phase2_snapshots'))}",
            f"stage_pipeline.params.training.staged_training.checkpoint_analysis.phase3_snapshots={_scalar_override(checkpoint_analysis_cfg.get('phase3_snapshots'))}",
            f"stage_pipeline.params.training.staged_training.checkpoint_analysis.include_phase_end={_scalar_override(checkpoint_analysis_cfg.get('include_phase_end'))}",
        ]
    )

    model_config_name_or_path = student_cfg.get("model_config_name_or_path")
    if model_config_name_or_path is not None:
        hydra_overrides.append(
            "stage_pipeline.params.student.model_config_name_or_path="
            f"{_scalar_override(model_config_name_or_path)}"
        )

    for key, value in (student_cfg.get("model_config_overrides") or {}).items():
        hydra_overrides.append(
            "+stage_pipeline.params.student.model_config_overrides."
            f"{key}={_scalar_override(value)}"
        )

    reproducibility = {
        "experiment": "distill_family_study_template",
        "family": family_name,
        "family_config": family_cfg["family_config"],
        "teacher": {
            "model_name_or_path": family_cfg["teacher"]["model_name_or_path"],
            "model_revision": family_cfg["teacher"]["model_revision"],
            "trust_remote_code": bool(family_cfg["teacher"]["trust_remote_code"]),
        },
        "student": {
            "key": student_cfg["key"],
            "model_name_or_path": student_cfg["model_name_or_path"],
            "model_revision": student_cfg["model_revision"],
            "trust_remote_code": bool(student_cfg["trust_remote_code"]),
            "init_from_scratch": bool(student_cfg["init_from_scratch"]),
            "penultimate_dim": int(student_cfg["penultimate_dim"]),
            "model_config_name_or_path": model_config_name_or_path,
            "model_config_overrides": dict(student_cfg.get("model_config_overrides") or {}),
            "tokenizer_name": student_cfg["tokenizer_name"],
        },
        "layer_scheme": layer_scheme,
        "training_regime": training_regime,
        "staged_training_enabled": staged_training_enabled,
        "lm_loss_weight": float(lm_loss_weight),
        "seed": int(seed),
        "output_dir": shared["output_dir"],
        "logger": shared["logger"],
        "reusable_artifacts": {
            "probe_teacher_manifest_path": shared["reusable_artifacts"]["probe_teacher_manifest_path"],
            "phate_teacher_target_manifest_path": shared["reusable_artifacts"]["phate_teacher_target_manifest_path"],
        },
        "data": {
            "token_budget": token_budget,
            "max_length": max_length,
            "train_example_offset": train_example_offset,
            "resolved_total_steps": None if total_steps is None else int(total_steps),
        },
        "probe": {
            "size": probe_size,
            "size_multiplier": (
                None if shared["probe"].get("size_multiplier") is None else int(shared["probe"]["size_multiplier"])
            ),
            "train_fraction": float(shared["probe"]["train_fraction"]),
            "eval_fraction": float(shared["probe"]["eval_fraction"]),
            "sampling": shared["probe"]["sampling"],
            "save_ids": bool(shared["probe"]["save_ids"]),
        },
        "diffusion": dict(shared["diffusion"]),
        "phate": {
            **dict(shared["phate"]),
            "n_components": int(student_cfg["penultimate_dim"]),
        },
        "training": dict(shared["training"]),
        "optimizer": dict(shared["optimizer"]),
        "lr_scheduler": dict(shared["lr_scheduler"]),
        "regularization": dict(shared["regularization"]),
        "alignment": {
            "loss": shared["alignment"]["loss"],
            "mse_reduction": shared["alignment"]["mse_reduction"],
            "batch_size": int(shared["alignment"]["batch_size"]),
            "eval_batch_size": int(shared["alignment"]["eval_batch_size"]),
            "sample_every_n_steps": int(shared["alignment"]["sample_every_n_steps"]),
        },
        "seeds": dict(shared["seeds"]),
    }

    return StudyRun(
        study_name=study_name,
        run_name=run_name,
        family=family_name,
        family_config=family_cfg["family_config"],
        teacher_model=family_cfg["teacher"]["model_name_or_path"],
        teacher_revision=family_cfg["teacher"]["model_revision"],
        teacher_trust_remote_code=bool(family_cfg["teacher"]["trust_remote_code"]),
        student_key=student_cfg["key"],
        student_model=student_cfg["model_name_or_path"],
        student_revision=student_cfg["model_revision"],
        student_trust_remote_code=bool(student_cfg["trust_remote_code"]),
        student_init_from_scratch=bool(student_cfg["init_from_scratch"]),
        student_penultimate_dim=int(student_cfg["penultimate_dim"]),
        student_model_config_name_or_path=model_config_name_or_path,
        student_model_config_overrides=dict(student_cfg.get("model_config_overrides") or {}) or None,
        tokenizer_name=student_cfg["tokenizer_name"],
        layer_scheme=layer_scheme,
        training_regime=training_regime,
        staged_training_enabled=staged_training_enabled,
        lm_loss_weight=lm_loss_weight,
        seed=int(seed),
        token_budget=token_budget,
        max_length=max_length,
        train_example_offset=train_example_offset,
        probe_size=probe_size,
        reproducibility=reproducibility,
        hydra_overrides=hydra_overrides,
    )


def materialize_study(cfg: DictConfig) -> list[StudyRun]:
    study = _to_plain(cfg.study)
    training_regimes = list(study.get("training_regimes") or ["staged"])
    runs: list[StudyRun] = []
    for family_name in study["family_order"]:
        family_cfg = study["families"][family_name]
        for student_cfg in family_cfg["students"]:
            for layer_scheme in study["layer_schemes"]:
                for training_regime in training_regimes:
                    for seed in study["seeds"]:
                        runs.append(
                            _build_run(
                                study_name=study["name"],
                                family_name=family_name,
                                family_cfg=family_cfg,
                                shared=study["shared"],
                                student_cfg=student_cfg,
                                layer_scheme=layer_scheme,
                                training_regime=str(training_regime),
                                seed=seed,
                            )
                        )
    return runs


def expected_run_count(cfg: DictConfig) -> int:
    training_regimes = list(cfg.study.get("training_regimes") or ["staged"])
    total_students = sum(len(cfg.study.families[family].students) for family in cfg.study.family_order)
    return (
        total_students
        * len(cfg.study.layer_schemes)
        * len(training_regimes)
        * len(cfg.study.seeds)
    )


def write_outputs(runs: list[StudyRun], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    plain_runs = [run.as_dict() for run in runs]
    output_json.write_text(json.dumps(plain_runs, indent=2, sort_keys=True), encoding="utf-8")

    fieldnames = [
        "study_name",
        "run_name",
        "family",
        "family_config",
        "teacher_model",
        "student_key",
        "student_model",
        "student_penultimate_dim",
        "student_model_config_name_or_path",
        "student_model_config_overrides",
        "tokenizer_name",
        "layer_scheme",
        "training_regime",
        "staged_training_enabled",
        "lm_loss_weight",
        "seed",
        "token_budget",
        "max_length",
        "train_example_offset",
        "probe_size",
        "reproducibility",
        "hydra_overrides",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in plain_runs:
            row = {key: run[key] for key in fieldnames}
            row["reproducibility"] = json.dumps(row["reproducibility"], separators=(",", ":"))
            row["hydra_overrides"] = json.dumps(row["hydra_overrides"], separators=(",", ":"))
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-config",
        type=Path,
        default=Path("downstream/distill_family_study/configs/study/within_family_publication.yaml"),
        help="Path to the declarative study YAML.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/study_manifests/within_family_publication_runs.json"),
        help="Where to write the materialized run manifest JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/study_manifests/within_family_publication_runs.csv"),
        help="Where to write the materialized run manifest CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.study_config)
    runs = materialize_study(cfg)
    expected_runs = expected_run_count(cfg)
    if len(runs) != expected_runs:
        raise ValueError(f"Expected {expected_runs} runs, materialized {len(runs)}")
    write_outputs(runs, output_json=args.output_json, output_csv=args.output_csv)
    print(f"Materialized {len(runs)} runs")
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
