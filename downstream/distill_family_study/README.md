# Downstream Distillation Study

This directory contains the downstream configs, launch scripts, and verification surface for the within-family PHATE-target distillation study.

## Read These Files First

- [architecture.md](/home/mila/d/drewd/newManylatents/manylatents/architecture.md)
- [verification_gates/README.md](/home/mila/d/drewd/newManylatents/manylatents/downstream/distill_family_study/verification_gates/README.md)
- [FINAL_TESTS.md](/home/mila/d/drewd/newManylatents/manylatents/FINAL_TESTS.md)

## Main Study Configs

Bounded publication study:

- `downstream/distill_family_study/configs/study/within_family_publication.yaml`
- uses a fixed `5e9` token budget

Full-dataset study:

- `downstream/distill_family_study/configs/study/within_family_full_pile.yaml`
- uses `token_budget: null`
- uses an explicit one-epoch-equivalent stopping rule via `training.max_steps`

## Core Launch Commands

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

## Notes

This README is intended to replace scattered local handoff notes. The source of truth for gates should be the verification docs and the consolidation script output, not agent-memory markdown files.
