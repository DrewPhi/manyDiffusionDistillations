# architecture.md

## Purpose

This repo contains the reusable `manylatents` framework plus a downstream experiment package for within-family PHATE-target distillation across Pythia, Qwen2.5, and T5.

The downstream experiment surface lives under:

- `downstream/distill_family_study/`

## High-Level Flow

The study is a four-stage pipeline:

1. `probe_teacher`
2. `phate_teacher_target`
3. `distill_sweep_grid`
4. `sweep_results_sheet`

That flow is executed by the stage-pipeline runner in core `manylatents`.

## Core vs Downstream Boundary

Core `manylatents` owns:

- stage-pipeline execution
- artifact manifests and resume behavior
- text datamodule and HF training code
- family/layer resolution helpers
- reusable pipeline stages

Downstream `distill_family_study` owns:

- concrete study configs
- family registry
- layer-scheme registry
- study materialization and submission scripts
- cluster sbatch wrappers
- aggregation and plotting scripts

## Key Directories

Core pipeline implementation:

- `manylatents/pipeline/`
- `manylatents/data/text.py`
- `manylatents/lightning/hf_trainer.py`
- `manylatents/lightning/callbacks/activation_tracker.py`

Downstream study surface:

- `downstream/distill_family_study/configs/`
- `downstream/distill_family_study/scripts/`
- `downstream/distill_family_study/verification_gates/`

## Main Config Layers

Family registry:

- `downstream/distill_family_study/configs/family/pythia.yaml`
- `downstream/distill_family_study/configs/family/qwen.yaml`
- `downstream/distill_family_study/configs/family/t5.yaml`

Layer schemes:

- `downstream/distill_family_study/configs/layer_scheme/penultimate_only.yaml`
- `downstream/distill_family_study/configs/layer_scheme/second_plus_penultimate.yaml`

Single-run experiment template:

- `downstream/distill_family_study/configs/experiment/distill_family_study_template.yaml`
- `downstream/distill_family_study/configs/stage_pipeline/distill_family_study_template.yaml`

Study-level configs:

- bounded study: `downstream/distill_family_study/configs/study/within_family_publication.yaml`
- full-dataset study: `downstream/distill_family_study/configs/study/within_family_full_pile.yaml`

## Launch Surfaces

Materialize a study into canonical run specs:

- `downstream/distill_family_study/scripts/materialize_distill_study.py`

Submit a materialized study to SLURM:

- `downstream/distill_family_study/scripts/submit_distill_study.py`

Run one materialized spec:

- `downstream/distill_family_study/scripts/run_materialized_distill_run.py`
- `downstream/distill_family_study/scripts/run_distill_family_study_single.sbatch`

Mini launcher validation:

- `downstream/distill_family_study/scripts/run_submit_distill_study_mini_validation.py`
- `downstream/distill_family_study/scripts/run_submit_distill_study_mini_validation.sbatch`

## Verification Surface

The intended verification entrypoints are:

- `downstream/distill_family_study/verification_gates/`
- `FINAL_TESTS.md`
- `downstream/distill_family_study/scripts/consolidate_distill_handoff.py`

The consolidation script is the sanity-check layer that reads actual result JSON files and emits a compact status summary. It exists so a reviewer does not need to trust narrative notes.

## What Not To Use As Canonical Context

Do not treat scattered local handoff notes as the primary source of truth.

Examples:

- `memory.md`
- `notesformorning.md`
- `README_CODEX_MANYLATENTS_MIGRATION.md`

Those may contain useful history, but the canonical experiment surface should be the configs, scripts, verification gates, and result-consolidation outputs.
