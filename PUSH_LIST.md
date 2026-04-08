# Push List For Big-Cluster Distillation Run

This is the short push-prep checklist for the current staged-plus-control study.

## Push

Push the code, configs, tests, and operator docs required to run:

- the staged regime
- the matched-budget `control_task_only` regime
- the smoke wrapper
- the full-Pile launch path

Core areas:

- `manylatents/pipeline/`
- `manylatents/data/`
- `manylatents/lightning/`
- `manylatents/configs/`
- `downstream/distill_family_study/configs/`
- `downstream/distill_family_study/scripts/`
- `tests/pipeline/`
- `downstream/distill_family_study/README.md`
- `FINAL_TESTS.md`
- `architecture.md`

## Do Not Push

Do not push generated or machine-local artifacts:

- `results/`
- `outputs/` logs and job outputs
- `representationvisualizations/`
- local scratch notes
- agent memory files

## Repo State Required Before Push

The pushed branch should let another operator do all of the following without reading old handoff notes:

- run the smoke wrapper
- confirm both regimes materialize
- confirm the smoke reaches real training
- materialize the full-Pile manifest
- submit the full experiment
- aggregate and plot results

## Current Expected Study Shapes

- smoke study:
  - `2` runs
  - `1` staged
  - `1` control
- full-Pile study:
  - `48` runs
  - `24` staged
  - `24` control

## Final Check Before Push

- docs point to the current four-family staged-plus-control study
- no `lambda_align` launch instructions remain in the active runbooks
- the full-Pile config and README agree on the hyperparameters
- smoke instructions point at `run_staged_smoke_a100_1gpu.sbatch`
