# Final Tests

This file is the short pre-launch checklist for the current within-family PHATE-target distillation study.

Scope:

- four families: `pythia`, `qwen`, `bert`, `deberta_v3`
- two regimes: `staged`, `control_task_only`
- two layer schemes: `penultimate_only`, `second_plus_penultimate`
- one-GPU smoke: `downstream/distill_family_study/configs/study/staged_smoke_a100_1gpu.yaml`
- real full run: `downstream/distill_family_study/configs/study/within_family_full_pile.yaml`

The active study is not the old `lambda_align` sweep. The current experiment is:

- `staged`
  - Phase 1: alignment only
  - Phase 2: task only with aligned layers frozen
  - Phase 3: task only with those layers unfrozen
- `control_task_only`
  - one continuous task-only run over the full staged budget
  - save a matched-budget checkpoint at `phase2 + phase3`
  - save the full-budget endpoint at `phase1 + phase2 + phase3`

## Current Validation Bar

Before launching the real study, all of the following should be true:

1. The smoke launcher materializes both regimes.
2. The smoke run reaches real training for at least one staged run and one control run.
3. `distill_sweep_grid` writes phase-aware checkpoints and queue entries.
4. Aggregation and plotting succeed on real outputs.
5. The full-study manifest matches the expected run counts.

## Required Checks

### 1. Targeted Pytest

Run:

```bash
module load miniconda
source "$CONDA_ACTIVATE"
conda activate manylatents
UV_CACHE_DIR=/tmp/uv-cache uv run --active --no-sync pytest \
  tests/pipeline/test_distill_study_config.py \
  tests/pipeline/test_submit_distill_study_filters.py \
  tests/pipeline/test_distillation_sweep_stage.py -q
```

Pass condition:

- all tests pass
- no staged-regime regression
- no control-regime regression

### 2. Smoke Materialization

Run:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/staged_smoke_a100_1gpu.yaml
```

Pass condition:

- smoke manifest is created
- exactly two run specs are materialized
- one run is `staged`
- one run is `control_task_only`

### 3. Smoke Launch

Run:

```bash
sbatch downstream/distill_family_study/scripts/run_staged_smoke_a100_1gpu.sbatch
```

Pass condition:

- the wrapper starts
- both smoke run specs are submitted
- at least one staged run reaches Phase 1 training
- at least one control run reaches task-only training
- `analysis_queue.jsonl` is emitted

### 4. Aggregation And Plotting

Run:

```bash
python downstream/distill_family_study/scripts/aggregate_distill_study.py
python downstream/distill_family_study/scripts/plot_distill_study.py
```

Pass condition:

- aggregation completes without schema errors
- plotting completes without missing-column errors
- outputs land under the expected study results directory

### 5. Full Manifest Gate

Run:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/within_family_full_pile.yaml
```

Pass condition:

- manifest dry run succeeds without submission
- `48` run specs are materialized
- `24` are `staged`
- `24` are `control_task_only`
- family counts are balanced across the four families

## Launch Rule

Do not launch the full study until the smoke run has exercised both regimes and the manifest gate is clean.

Use these docs together:

- [downstream/distill_family_study/README.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/README.md)
- [architecture.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/architecture.md)
- [downstream/distill_family_study/verification_gates/README.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/verification_gates/README.md)
