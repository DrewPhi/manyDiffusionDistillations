# Downstream Distillation Study

This directory contains the downstream configs, launch scripts, detached analysis worker, and verification surface for the within-family PHATE-target distillation study.

## Read These Files First

- [architecture.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/architecture.md)
- [Revamp.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/Revamp.md)
- [verification_gates/README.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/verification_gates/README.md)
- [FINAL_TESTS.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/FINAL_TESTS.md)

## Main Study Configs

Bounded publication study:

- `downstream/distill_family_study/configs/study/within_family_publication.yaml`
- uses a fixed `5e9` token budget
- materializes both `staged` and `control_task_only` regimes
- active matrix: `4 families x 3 students x 2 layer schemes x 2 regimes x 1 seed = 48 runs`

Full-dataset study:

- `downstream/distill_family_study/configs/study/within_family_full_pile.yaml`
- uses `token_budget: null`
- uses an explicit one-epoch-equivalent stopping rule via `training.max_steps`
- materializes both `staged` and `control_task_only` regimes
- the staged regime splits the total budget across the three training phases
- the control regime runs task-only continuously for the full staged budget and saves matched-budget checkpoints at:
  - `phase2 + phase3`
  - `phase1 + phase2 + phase3`
- retains rich detached analysis checkpoints for PHATE-trajectory monitoring

One-GPU smoke study:

- `downstream/distill_family_study/configs/study/staged_smoke_a100_1gpu.yaml`
- two `bert_11m` runs:
  - one `staged`
  - one `control_task_only`
- intended to fit on one `a100l` within a day
- exercises training, queue emission, detached analysis, aggregation, and plotting

Remaining-families one-GPU smoke study:

- `downstream/distill_family_study/configs/study/staged_smoke_remaining_families_a100_1gpu.yaml`
- six runs across the remaining non-BERT families:
  - `pythia_410m`
  - `deberta_v3_xsmall`
  - `qwen2_5_0_5b`
- each student materializes:
  - one `staged`
  - one `control_task_only`
- family order is `pythia`, `deberta_v3`, then `qwen`
- intended as a family-coverage smoke before the full 48-run launch

## Current Experiment Design

The active study is no longer a broad `lambda_align` sweep.

Each study now materializes two training regimes:

1. `staged`
2. `control_task_only`

The `staged` regime runs:

1. Phase 1: alignment-only training
2. Phase 2: freeze aligned student layers and train on task loss only
3. Phase 3: unfreeze aligned layers and continue task-only fine-tuning

The `control_task_only` regime runs:

1. task-only training for the same full optimizer-step budget as `phase1 + phase2 + phase3`
2. matched analysis checkpoints at the `phase2 + phase3` budget boundary
3. a final checkpoint at the full `phase1 + phase2 + phase3` budget boundary

The study still sweeps:

- family
- student size
- layer scheme
- training regime
- seed

It no longer sweeps:

- `lambda_align`

## What The Distillation Target Actually Is

The current study is more specific than "match teacher activations."

This is a within-family study:

- Pythia teacher -> Pythia student
- Qwen teacher -> Qwen student
- BERT teacher -> BERT student
- DeBERTa-v3 teacher -> DeBERTa-v3 student

It is not a cross-family transfer study.

For each configured teacher layer:

- extract activations on a fixed probe set
- reduce sequence outputs with `reduce: mean`, producing one vector per example
- build a diffusion operator from those per-example vectors using the configured adaptive Gaussian kernel followed by row normalization
- run PHATE to obtain a low-dimensional target with dimensionality matching the student penultimate size
- Procrustes-align that PHATE target to the teacher activation basis
- in the staged setup, optimize the student against those aligned PHATE coordinates during Phase 1

So the student alignment loss compares:

- student mean-pooled layer representations
- aligned PHATE coordinates derived from teacher geometry

It does not compare:

- full tokenwise hidden-state tensors
- raw teacher activations directly
- the diffusion operators directly

If teacher activation width differs from the PHATE target width, teacher activations may be PCA-reduced inside the Procrustes alignment step only. That PCA reduction is an alignment helper, not the final target consumed by the student.

## Probe Size Policy

The study materializer supports two probe-size modes.

Adaptive probe size:

- set `study.shared.probe.size_multiplier`
- each run uses `probe_size = student_penultimate_dim * size_multiplier`

Fixed probe size:

- set `study.shared.probe.size`
- the fixed `size` takes precedence over `size_multiplier`
- every run in the study uses that same probe size

Example fixed-size config:

```yaml
study:
  shared:
    probe:
      size: 4096
      size_multiplier: 2
```

In fixed-size mode, choose a probe size that is at least as large as the largest student alignment width you intend to run in that study. In practice, it should comfortably cover the largest student penultimate dimension, because PHATE is used to produce a target coordinate system whose dimensionality matches the student alignment width.

## Full-Pile Hyperparameters

The full-dataset experiment hyperparameters are set explicitly in:

- `downstream/distill_family_study/configs/study/within_family_full_pile.yaml`

The current full-Pile launch uses:

- training regimes:
  - `staged`
  - `control_task_only`
- families:
  - `pythia`
  - `qwen`
  - `bert`
  - `deberta_v3`
- layer schemes:
  - `penultimate_only`
  - `second_plus_penultimate`
- seeds:
  - `42`
- total runs:
  - `48`
- precision:
  - `bf16`
- global batch size:
  - `512`
- micro batch size:
  - `8`
- grad accumulation steps:
  - `64`
- max length:
  - `1024`
- stopping rule:
  - `training.max_steps: 345711`
- optimizer:
  - `adamw`
  - `learning_rate: 3e-4`
  - `betas: [0.9, 0.95]`
  - `eps: 1e-8`
  - `weight_decay: 0.1`
- LR schedule:
  - `cosine`
  - `warmup_steps: 2000`
  - `min_lr: 3e-5`
- regularization:
  - `dropout: 0.0`
  - `label_smoothing: 0.0`
- probe policy:
  - `size_multiplier: 2`
  - `train_fraction: 0.8`
  - `eval_fraction: 0.2`
- alignment loss:
  - `mse`
  - `batch_size: 16`
  - `eval_batch_size: 8`
- staged budgets:
  - `phase1.max_steps: 34571`
  - `phase2.max_steps: 259283`
  - `phase3.max_steps: 51857`
- fixed detached analysis checkpoints:
  - `34571`
  - `86428`
  - `172856`
  - `259283`

The config is the source of truth. If any of these values change, update the YAML first and then update this README.

## Core Launch Commands

Before any of the commands below, activate the expected environment on cluster shells:

```bash
module load miniconda
source "$CONDA_ACTIVATE"
conda activate manylatents
```

Materialize the bounded study manifest directly:

```bash
python downstream/distill_family_study/scripts/materialize_distill_study.py
```

Dry-run the bounded study manifest through the submit surface:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py
```

Dry-run the full-dataset study manifest:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/within_family_full_pile.yaml \
  --manifest-dir results/study_manifests/within_family_full_pile
```

Submit one filtered family batch:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --submit \
  --family pythia \
  --student-key pythia_410m
```

Submit from the full-dataset study config:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/within_family_full_pile.yaml \
  --manifest-dir results/study_manifests/within_family_full_pile \
  --submit
```

Submit the full experiment in two explicit steps:

1. Dry-run and inspect the manifest:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/within_family_full_pile.yaml \
  --manifest-dir results/study_manifests/within_family_full_pile
```

2. Submit the full `48`-run matrix:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/within_family_full_pile.yaml \
  --manifest-dir results/study_manifests/within_family_full_pile \
  --submit
```

Optional: submit one filtered slice first:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/within_family_full_pile.yaml \
  --manifest-dir results/study_manifests/within_family_full_pile \
  --family bert \
  --student-key bert_11m \
  --layer-scheme penultimate_only \
  --submit
```

Run the mini launcher validation wrapper:

```bash
sbatch downstream/distill_family_study/scripts/run_submit_distill_study_mini_validation.sbatch
```

Run the one-GPU staged smoke end to end:

```bash
sbatch downstream/distill_family_study/scripts/run_staged_smoke_a100_1gpu.sbatch
```

Run the remaining-families one-GPU staged smoke end to end:

```bash
sbatch downstream/distill_family_study/scripts/run_staged_smoke_remaining_families_a100_1gpu.sbatch
```

Aggregate completed study outputs:

```bash
python downstream/distill_family_study/scripts/aggregate_distill_study.py
python downstream/distill_family_study/scripts/plot_distill_study.py
```

Process detached checkpoint-analysis jobs for one run:

```bash
python downstream/distill_family_study/scripts/process_analysis_queue.py \
  --queue-path outputs/pipelines/<run_name>/distill_sweep_grid/analysis_queue.jsonl
```

Generate a handoff summary from actual result files:

```bash
python downstream/distill_family_study/scripts/consolidate_distill_handoff.py
```

## Operator Handoff

Use this section as the shortest reproducible launch recipe for another operator.

Environment:

- load `miniconda`
- activate the `manylatents` conda environment
- confirm `python`, `pytest`, and the Hugging Face model dependencies resolve inside that env

Required paths and variables:

- `HF_HOME`
- `HF_DATASETS_CACHE`
- `TRANSFORMERS_CACHE`
- `TMPDIR`
- access to `/network/datasets/pile.var/pile_uncopyrighted/pile-uncopyrighted`

Optional but expected for online tracking:

- `WANDB_ENTITY`
- `WANDB_PROJECT`

Recommended W&B policy:

- the main training process logs scalar metrics to the training run
- the detached analysis worker logs PHATE geometry outputs to a linked companion analysis run
- both runs should share enough metadata to join them later

Recommended pre-launch sequence:

1. Dry-run the bounded publication manifest.
2. Launch the smoke wrapper that now exercises both `staged` and `control_task_only`.
3. Launch a small filtered slice through `submit_distill_study.py`.
4. Run aggregation and plotting on the resulting real outputs.
5. Process at least one real `analysis_queue.jsonl` with the detached worker.
6. Re-run once to check partial-failure and reuse behavior.
7. Record commit hash, env details, and exact commands before scaling up.

Expected output roots:

- manifests under `results/study_manifests/`
- pipeline outputs under `outputs/pipelines/`
- aggregated publication outputs under `results/publication_within_family/`

For the remaining-families smoke specifically:

- manifest dir:
  - `results/study_manifests/staged_smoke_remaining_families_a100_1gpu`
- SLURM wrapper logs:
  - `outputs/slurm/staged_smoke_remaining_families_a100-<jobid>.out`
  - `outputs/slurm/staged_smoke_remaining_families_a100-<jobid>.err`
- pipeline outputs:
  - `outputs/pipelines/staged_smoke_remaining_families_a100_1gpu_*`
- aggregated outputs:
  - `results/publication_within_family/staged_smoke_remaining_families_a100_1gpu`

Known caveat:

- continuation should be described as resume from saved student weights
- it should not be described as exact optimizer-state resume

## Required Environment

Expected cluster environment:

- working `manylatents` Python environment
- Hugging Face cache paths configured on shared scratch
- W&B credentials configured if online logging is desired
- access to the uncopyrighted Pile mirror

Important env vars used by the current launch scripts:

- `HF_HOME`
- `HF_DATASETS_CACHE`
- `TRANSFORMERS_CACHE`
- `HF_HUB_OFFLINE`
- `TRANSFORMERS_OFFLINE`
- `WANDB_ENTITY`
- `WANDB_PROJECT`
- `TMPDIR`

## Continuation Policy

Current intended policy for the large run:

- run one epoch-equivalent first
- inspect results
- continue only if justified

Important caveat:

- the current pipeline should be treated as supporting continuation from saved student weights
- it should not be described as exact optimizer-state resume unless that support is added explicitly

## Checkpoint And Analysis Policy

Periodic training checkpoints and detached analysis checkpoints are not the same thing.

Best-checkpoint retention:

- periodic `student_step*.pt` checkpoints are still ranked by validation loss
- the usual `save_top_k` pruning still applies to those periodic checkpoints
- every run also writes `student_last.pt`

Detached analysis checkpoints:

- every run may write `student_analysis_step{step}.pt`
- those checkpoints are not pruned by `save_top_k`
- they are queued into `analysis_queue.jsonl`
- the detached analysis worker converts them into:
  - student activations on the fixed probe set
  - diffusion operators
  - PHATE 2D coordinates
  - W&B geometry outputs

The full-Pile study requests rich geometry monitoring:

- Phase 1: 5 snapshots including phase end
- Phase 2: 10 snapshots including phase end
- Phase 3: 5 snapshots including phase end

Important detail:

- actual phase-end checkpoints are always preserved, even if Phase 1 stops early
- this is necessary so the geometry trajectory respects real phase boundaries rather than only planned ones

## Notes

This README is intended to replace scattered local handoff notes. The source of truth for gates should be the verification docs, the configs, and the consolidation-script output, not agent-memory markdown files.
