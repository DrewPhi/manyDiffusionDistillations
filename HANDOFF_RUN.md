# Handoff Run Note

This repo contains the `manylatents` framework plus the active downstream within-family PHATE-target distillation study.

Read these first:

- [downstream/distill_family_study/README.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/README.md)
- [architecture.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/architecture.md)
- [FINAL_TESTS.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/FINAL_TESTS.md)

Main study configs:

- publication-scale config:
  [within_family_publication.yaml](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/configs/study/within_family_publication.yaml)
- full-Pile config:
  [within_family_full_pile.yaml](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/configs/study/within_family_full_pile.yaml)
- one-GPU smoke:
  [staged_smoke_a100_1gpu.yaml](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/configs/study/staged_smoke_a100_1gpu.yaml)
- remaining-families one-GPU smoke:
  [staged_smoke_remaining_families_a100_1gpu.yaml](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/configs/study/staged_smoke_remaining_families_a100_1gpu.yaml)

Environment:

```bash
module load miniconda
source "$CONDA_ACTIVATE"
conda activate manylatents
```

Current experiment surface:

- families: `pythia`, `qwen`, `bert`, `deberta_v3`
- layer schemes: `penultimate_only`, `second_plus_penultimate`
- regimes:
  - `staged`
  - `control_task_only`

Quick operator flow:

1. Run the targeted pytest slice from [FINAL_TESTS.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/FINAL_TESTS.md).
2. Materialize the smoke manifest.
3. Launch the smoke wrapper.
4. Launch the remaining-families smoke wrapper.
5. Confirm the staged and control runs reach training across `pythia`, `deberta_v3`, and `qwen`.
6. Only then materialize or submit the full-Pile study.

Useful commands:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/staged_smoke_a100_1gpu.yaml

sbatch downstream/distill_family_study/scripts/run_staged_smoke_a100_1gpu.sbatch

sbatch downstream/distill_family_study/scripts/run_staged_smoke_remaining_families_a100_1gpu.sbatch

python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/within_family_full_pile.yaml
```

If resuming later to check the remaining-families smoke, inspect these first:

- manifest:
  [submission_manifest.json](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/results/study_manifests/staged_smoke_remaining_families_a100_1gpu/submission_manifest.json)
- pipeline outputs:
  `outputs/pipelines/staged_smoke_remaining_families_a100_1gpu_*`
- aggregate outputs:
  `results/publication_within_family/staged_smoke_remaining_families_a100_1gpu/`
- wrapper logs:
  `outputs/slurm/staged_smoke_remaining_families_a100-<jobid>.out`
  `outputs/slurm/staged_smoke_remaining_families_a100-<jobid>.err`
- expected run count:
  `6`
- family order:
  `pythia`, `deberta_v3`, `qwen`

Use the verification gates for go/no-go decisions:

- [downstream/distill_family_study/verification_gates/README.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/verification_gates/README.md)
