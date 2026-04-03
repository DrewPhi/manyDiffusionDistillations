# Handoff Run Note

This repo contains the `manylatents` framework plus the downstream within-family PHATE-target distillation study.

Read these first:

- [downstream/distill_family_study/README.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/README.md)
- [FINAL_TESTS.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/FINAL_TESTS.md)
- [architecture.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/architecture.md)

Main experiment drivers:

- publication study config:
  [`downstream/distill_family_study/configs/study/within_family_publication.yaml`](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/configs/study/within_family_publication.yaml)
- full-dataset study config:
  [`downstream/distill_family_study/configs/study/within_family_full_pile.yaml`](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/configs/study/within_family_full_pile.yaml)

Activate environment:

```bash
module load miniconda
source "$CONDA_ACTIVATE"
conda activate manylatents
```

Recommended first steps:

1. Dry-run the publication manifest:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py
```

2. Launch the recommended mini slice through the real launcher:

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

3. After those runs finish, aggregate and plot:

```bash
python downstream/distill_family_study/scripts/aggregate_distill_study.py
python downstream/distill_family_study/scripts/plot_distill_study.py
```

Important experiment facts:

- this is a within-family study, not cross-family transfer
- alignment is on mean-pooled per-example layer representations
- the student matches aligned PHATE coordinates derived from teacher diffusion geometry
- the deployed study path uses an adaptive Gaussian kernel followed by row normalization
- continuation should be described as resume from saved student weights, not exact optimizer-state resume

Probe size:

- adaptive mode is controlled by `study.shared.probe.size_multiplier`
- fixed mode is controlled by `study.shared.probe.size`
- if fixed mode is used, choose a size that covers the largest student alignment width in the study

Before scaling to the full run, check the remaining launch gates in:

- [FINAL_TESTS.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/FINAL_TESTS.md)
- [downstream/distill_family_study/verification_gates/README.md](/home/mila/d/drewd/codeReview/manyDiffusionDistillations/downstream/distill_family_study/verification_gates/README.md)
