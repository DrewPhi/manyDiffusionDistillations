# Push List For Big-Cluster Distillation Run

This file is the single push-prep checklist for getting the repo into a clean state for running the larger distillation study on another cluster.

The goal is simple:

- push the code, configs, tests, and minimal docs needed to run the experiment
- do not push local agent handoff notes, scratch outputs, or generated result artifacts
- reduce the number of overlapping markdown notes that describe the same operational history

## What Must Be Pushed

Push the code and config changes that are required for the publication-scale within-family study to run.

Core code and config areas that matter:

- `manylatents/pipeline/`
- `manylatents/data/text.py`
- `manylatents/lightning/hf_trainer.py`
- `manylatents/lightning/callbacks/activation_tracker.py`
- `manylatents/lightning/hooks.py`
- `manylatents/experiment.py`
- `manylatents/main.py`
- `manylatents/api.py`
- `manylatents/configs/config.py`
- `manylatents/configs/config.yaml`
- `manylatents/configs/logger/wandb.yaml`
- `manylatents/configs/trainer/logger/wandb.yaml`
- `manylatents/configs/data/pile_uncopyrighted.yaml`
- any stage-pipeline or experiment configs under `manylatents/configs/stage_pipeline/`
- any required experiment presets under `manylatents/configs/experiment/`

Downstream study assets that matter:

- `downstream/distill_family_study/configs/`
- `downstream/distill_family_study/scripts/`
- `downstream/distill_family_study/README.md`

Tests that should be pushed with the implementation:

- `tests/pipeline/`
- `tests/test_text_datamodule_pile_controls.py`
- `manylatents/lightning/tests/test_hf_trainer.py`
- `manylatents/lightning/tests/test_hf_trainer_from_config.py`
- `manylatents/lightning/tests/test_hooks.py`

Dependency updates that should be pushed if required by the new pipeline:

- `pyproject.toml`
- `uv.lock`

## What Should Not Be Pushed

These are local, generated, or agent-facing artifacts and should not be part of the repo state used for the large run.

- `.codex`
- `results/`
- `representationvisualizations/`
- `scripts/outputs/`
- SLURM logs under `outputs/` if they are untracked or machine-specific
- ad hoc analysis outputs and cached artifacts produced by local experiments

These files are useful for local debugging/history, but should not be pushed as primary repo docs for the cluster handoff:

- `memory.md`
- `notesformorning.md`
- `README_CODEX_MANYLATENTS_MIGRATION.md`

## What Should Be Consolidated

Right now the operational story is spread across too many notes. Before pushing, keep one canonical handoff doc and drop the rest from the push.

Recommended doc structure:

1. Keep `new_distilation_expiriment_README.md` only if it is rewritten into a concise operator-facing runbook.
2. Keep `FINAL_TESTS.md` only if it stays as the short launch-gate checklist.
3. Remove the need for `memory.md`, `notesformorning.md`, and `README_CODEX_MANYLATENTS_MIGRATION.md` in the pushed branch.

Recommended replacement set:

- `PUSH_LIST.md`
  - what to push
  - what not to push
  - what still needs cleanup
- `FINAL_TESTS.md`
  - launch gates
  - pass/fail conditions
  - exact final validation steps
- `downstream/distill_family_study/README.md`
  - how to run the study
  - required env vars
  - expected configs and scripts

## Minimum Repo State For The Other Cluster

The pushed branch should let someone else do all of the following without reading agent notes:

- materialize the 54-run study manifest
- run family smoke jobs
- run the mini launcher validation
- launch family batches through `submit_distill_study.py`
- aggregate outputs
- plot outputs

That means the branch needs:

- working family configs
- working layer-scheme configs
- working stage-pipeline configs
- the downstream launch scripts
- the current tests
- one clear runbook

It does not need:

- historical failure notes
- job-id diaries
- local path archaeology
- generated reports from this machine

## Pre-Push Cleanup Checklist

- decide which markdown file is the single canonical runbook
- delete or stop tracking local-only handoff notes from the pushed branch
- make sure `results/`, `representationvisualizations/`, and other generated artifacts are ignored
- confirm no local cluster-specific absolute paths are hardcoded in the pushed configs unless intentionally parameterized
- make sure the downstream README explains the required environment variables for HF cache, offline mode, and W&B
- make sure the launch scripts reference the current stable commands, not old smoke-only commands
- make sure tests still reflect the current launcher and stage-pipeline behavior

## Recommended Next Step

Use this file as the pruning checklist, then do one cleanup pass with this rule:

- keep code, configs, tests, and one operator-facing runbook
- drop agent memory files, generated outputs, and duplicated narrative notes
