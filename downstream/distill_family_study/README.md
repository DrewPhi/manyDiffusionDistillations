# Downstream Distillation Study

This directory contains the downstream configs, launch scripts, and verification surface for the within-family PHATE-target distillation study.

## Read These Files First

- [architecture.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/architecture.md)
- [verification_gates/README.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/verification_gates/README.md)
- [FINAL_TESTS.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/FINAL_TESTS.md)

## Main Study Configs

Bounded publication study:

- `downstream/distill_family_study/configs/study/within_family_publication.yaml`
- uses a fixed `5e9` token budget

Full-dataset study:

- `downstream/distill_family_study/configs/study/within_family_full_pile.yaml`
- uses `token_budget: null`
- uses an explicit one-epoch-equivalent stopping rule via `training.max_steps`
- retains fixed analysis checkpoints at 10%, 25%, 50%, and 75% of training

## What The Distillation Target Actually Is

The current study is more specific than "match teacher activations."

This is a within-family study:

- Pythia teacher -> Pythia student
- Qwen teacher -> Qwen student
- T5 teacher -> T5 student

It is not a cross-family transfer study.

For each configured teacher layer:

- extract activations on a fixed probe set
- reduce sequence outputs with `reduce: mean`, producing one vector per example
- build a diffusion operator from those per-example vectors using the configured adaptive Gaussian kernel followed by row normalization
- run PHATE to obtain a low-dimensional target with dimensionality matching the student penultimate size
- Procrustes-align that PHATE target to the teacher activation basis
- train the student to match the aligned PHATE coordinates on probe minibatches during LM training

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

In fixed-size mode, choose a probe size that is at least as large as the largest student alignment width you intend to run in that study. In practice, it should be chosen to comfortably cover the largest student penultimate dimension, because PHATE is used to produce a target coordinate system whose dimensionality matches the student alignment width.

The code does not infer that fixed size automatically. If you choose fixed mode, you are responsible for setting a large enough value.

## Core Launch Commands

Before any of the commands below, activate the expected environment on cluster shells:

```bash
module load miniconda
source "$CONDA_ACTIVATE"
conda activate manylatents
```

Dry-run the bounded study manifest:

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

Run the mini launcher validation wrapper:

```bash
sbatch downstream/distill_family_study/scripts/run_submit_distill_study_mini_validation.sbatch
```

Aggregate completed study outputs:

```bash
python downstream/distill_family_study/scripts/aggregate_distill_study.py
python downstream/distill_family_study/scripts/plot_distill_study.py
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

Recommended pre-launch sequence:

1. Dry-run the bounded publication manifest.
2. Launch the 6-run mini slice through `submit_distill_study.py`.
3. Run aggregation and plotting on the resulting real outputs.
4. Re-run once to check partial-failure / reuse behavior.
5. Record commit hash, env details, and exact commands before scaling up.

Expected output roots:

- manifests under `results/study_manifests/`
- pipeline outputs under `outputs/pipelines/`
- aggregated publication outputs under `results/publication_within_family/`

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

## Checkpoint Policy

Periodic training checkpoints and analysis checkpoints are not the same thing.

Best-checkpoint retention:

- periodic `student_step*.pt` checkpoints are still ranked by validation loss
- the usual `save_top_k` pruning still applies to those periodic checkpoints
- every run also writes `student_last.pt`

Full-dataset trajectory checkpoints:

- the full-Pile study additionally retains fixed analysis checkpoints at:
  - `34571` steps (10%)
  - `86428` steps (25%)
  - `172856` steps (50%)
  - `259283` steps (75%)
- those are written as `student_analysis_step{step}.pt`
- they are not pruned by `save_top_k`

This exists so the large run can be probed after training to study how student diffusion geometry evolves over time without weakening the usual best-checkpoint retention policy.

## Notes

This README is intended to replace scattered local handoff notes. The source of truth for gates should be the verification docs and the consolidation script output, not agent-memory markdown files.
