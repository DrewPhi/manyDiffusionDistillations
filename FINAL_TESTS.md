# Final Tests

This file is the final pre-launch checklist for the within-family publication study.

Validation claims in this file should always be interpreted as scoped to:

- the current git commit being handed off
- the `manylatents` conda environment used for launch
- the exact commands recorded in the handoff notes

If any of those change, rerun the relevant gate instead of inheriting confidence from older runs.

Current validated status:

- Pythia publication smoke passed: `9145958`
- Qwen publication smoke passed: `9145956`
- T5 publication smoke passed: `9146136`
- W&B online logging worked for all three families
- the generalized family-study pipeline now works for both decoder-only and encoder-decoder families
- the cheap final validation suite passed on job `9146930`
- targeted study/config tests passed: `27 passed`
  - command:
    `pytest tests/pipeline/test_distill_study_config.py tests/pipeline/test_submit_distill_study_filters.py tests/pipeline/test_distillation_sweep_stage.py -q`
  - environment:
    `module load miniconda && source "$CONDA_ACTIVATE" && conda activate manylatents`
- study manifest dry run passed with:
  - `54` run specs
  - family counts `18/18/18`
  - layer schemes `penultimate_only` and `second_plus_penultimate`
  - all runs marked `submitted=false`

What the passing family smokes prove:

- Hydra composition works on the generalized experiment path
- the stage chain runs end-to-end:
  - `probe_teacher`
  - `phate_teacher_target`
  - `distill_sweep_grid`
  - `sweep_results_sheet`
- `bf16` execution is compatible with the current activation-capture path
- T5 encoder-side layer alignment works with the generalized distillation stage
- the current representation contract is consistent across teacher and student:
  - probe extraction uses mean-pooled per-example layer vectors
  - PHATE targets are aligned coordinates derived from teacher diffusion geometry
  - training compares student mean-pooled vectors to those aligned PHATE targets

What they do not prove:

- aggregation and plotting succeed on partial and full publication-study outputs
- resume behavior is safe after partial failures
- artifact reuse is collision-free across a larger shared-HPC run
- actual submitted runs from `submit_distill_study.py` succeed on the cluster end-to-end

Experiment scope reminder:

- this is a within-family study, not a cross-family transfer study
- alignment is performed on mean-pooled per-example layer representations
- the student matches aligned PHATE coordinates derived from teacher diffusion geometry
- the deployed study path uses an adaptive Gaussian kernel with row normalization; it does not apply a separate `alpha` normalization term
- probe size may be adaptive by student width or fixed per study; if fixed, the configured size should cover the largest student alignment width in the study
- the full-Pile study now retains fixed analysis checkpoints at 10%, 25%, 50%, and 75% of training as `student_analysis_step{step}.pt`, in addition to `student_last.pt` and the usual best-checkpoint retention

## Completed Validation

These checks are now complete and green:

- family smoke wrappers passed end-to-end:
  - Pythia `9145958`
  - Qwen `9145956`
  - T5 `9146136`
- final validation wrapper passed:
  - job `9146930`
  - report `results/final_validation/20260402T192755Z/validation_report.md`
- targeted pytest checks passed:
  - `tests/pipeline/test_distill_study_config.py`
  - `tests/pipeline/test_submit_distill_study_filters.py`
  - `tests/pipeline/test_distill_study_aggregator.py`
  - `tests/pipeline/test_distillation_sweep_stage.py`
- real launcher dry run passed through `submit_distill_study.py`
- manifest shape checks passed:
  - `54` total runs
  - `18` runs per family
  - unique run names
  - correct layer schemes
  - correct student keys

## Final Pre-Launch Tests

### 1. Study Manifest Dry Run

Status: complete

Run:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py
```

Pass condition:

- `results/study_manifests/within_family_publication/run_specs/` exists
- `submission_manifest.json` exists
- the expected number of run specs is produced
- family, student, layer-scheme, and lambda names are readable and stable

Why this matters:

- this is the first test of the real launcher path, not just the family smoke wrappers

### 2. One Mini Study Launch

Status: remaining

Run a minimal real study batch instead of jumping straight to the full matrix.

Suggested scope:

- one family
- one student
- one layer scheme
- one or two `lambda_align` values

Pass condition:

- submitted runs materialize outputs in the expected directories
- W&B names and job names are stable
- the run set matches the manifest you expected

Why this matters:

- this validates orchestration, naming, and path contracts under the real study launcher

### 3. Aggregation And Plotting On Real Outputs

Status: remaining

Run:

```bash
python downstream/distill_family_study/scripts/aggregate_distill_study.py
python downstream/distill_family_study/scripts/plot_distill_study.py
```

Pass condition:

- aggregation completes without missing-field or schema errors
- plotting completes without missing-column or missing-file errors
- expected study outputs are generated under `results/publication_within_family/`

Why this matters:

- many pipelines pass training and fail only when rows are aggregated across runs

### 4. Resume / Partial-Failure Test

Status: remaining

Test:

- leave one run incomplete or intentionally fail one submission
- rerun the submission and aggregation path

Pass condition:

- completed runs are not needlessly recomputed
- incomplete runs are resubmitted cleanly
- aggregation tolerates partial completion without corrupting the study summary

Why this matters:

- this is exactly the kind of failure mode that appears on larger HPC launches

### 5. Artifact Collision Check

Status: remaining

Verify:

- probe IDs are not silently reused across incompatible runs
- PHATE target artifacts are scoped by compatible student target dimension
- output paths are namespaced by family, student, and run identity

Pass condition:

- no run can overwrite another run's probe, PHATE, or sweep-summary artifacts without being obviously intentional

Why this matters:

- shared storage and reruns make silent overwrites much more likely than code bugs

### 6. T5 Included In The Mini Study

Status: remaining

Requirement:

- do not treat the T5 family smoke as enough on its own
- include at least one T5 run in the mini study launched through `submit_distill_study.py`

Why this matters:

- T5 was the family that exposed the encoder-decoder mismatch
- it should be included in the final launcher validation path, not just the smoke wrapper path

### 7. Reproducibility Handoff

Status: remaining

Record before pushing or handing off:

- git commit hash
- Python environment details
- CUDA / driver details
- required environment variables
- expected cache locations
- exact submission commands used for the smoke and mini-study checks

Pass condition:

- another engineer can rerun the launch procedure without guessing hidden environment assumptions

Suggested handoff block to record verbatim before large submission:

- git commit hash
- output of `which python`
- output of `python --version`
- output of `pytest --version`
- output of `nvidia-smi | head`
- exact env activation commands
- exact dry-run command
- exact mini-launch command
- exact aggregation command
- exact plotting command

## What Is Left

What remains before a full launch:

1. Launch one minimal real batch through `submit_distill_study.py`.
2. Aggregate and plot those real outputs.
3. Verify rerun behavior after a partial or failed submission.
4. Confirm artifact paths are collision-safe on shared storage.
5. Record the exact handoff environment and commands.

## Minimal Viable `submit_distill_study.py` Stage

If the goal is the smallest real launcher test that gives high confidence the full launch will work, use this stage:

1. Dry run the manifest.
2. Submit one family and one student only.
3. Include one T5 student in that launcher path before scaling further.
4. Run aggregation and plotting on the resulting real outputs.
5. Only then expand to more students or families.

The minimal viable working slice should be:

- one student
- one family
- one layer scheme
- one or two `lambda_align` values
- launched through `submit_distill_study.py`, not the smoke wrappers

The highest-signal choice is:

- T5
- `t5_small`
- `penultimate_only`
- `lambda_align` values `0.0` and `0.5`

Why this is the best minimum:

- T5 was the architecture that exposed the nontrivial encoder-decoder bug
- if the real launcher works for T5, the simpler decoder-only families are lower risk
- one student and one layer scheme keeps cost bounded
- two lambda values prove the sweep dimension is working without paying for the full grid

## Step-By-Step Remaining Launch Path

1. Confirm the current dry-run manifest is still clean.

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py
```

2. Launch the smallest real study slice through the real launcher.

Suggested form:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --submit \
  --family pythia \
  --family qwen \
  --family t5 \
  --student-key pythia_410m \
  --student-key qwen2_5_0_5b \
  --student-key t5_small \
  --layer-scheme penultimate_only \
  --lambda-align 0.0 \
  --lambda-align 0.5
```

This is the intended minimal viable launcher slice:

- `3` families
- `1` student per family
- `1` layer scheme
- `2` lambda values
- total `6` launcher-submitted runs

3. Wait for those submitted runs to finish and inspect their outputs and W&B names.

4. Run aggregation and plotting on those real outputs.

```bash
python downstream/distill_family_study/scripts/aggregate_distill_study.py
python downstream/distill_family_study/scripts/plot_distill_study.py
```

5. Intentionally rerun the submission command or leave one run incomplete and rerun it.

6. Confirm no artifact overwrite or reuse hazard appears in probe IDs, PHATE targets, or sweep outputs.

7. Record commit, env, cache paths, and exact commands.

8. Only then expand to more students, then more families, then the full matrix.

## Recommended Launch Gate

Do not submit the full publication matrix until all of the following are true:

- all three family smokes pass
- the study manifest dry run passes
- one mini study passes
- aggregation and plotting pass on real mini-study outputs
- resume behavior is acceptable
- artifact collision checks are clean
- the environment and launch procedure are documented for handoff

## Short Version

The family smokes show that the engine starts.

The cheap validation gates are now green.

The only major risks left are the real launcher submission path, late aggregation/plotting behavior on real study outputs, recovery after partial failure, and artifact hygiene on shared storage.
