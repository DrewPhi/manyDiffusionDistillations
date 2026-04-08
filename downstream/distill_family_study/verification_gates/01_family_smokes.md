# 01 Family Smokes

The current smoke should cover the active experiment surface, not the old per-family T5-era preflights.

Required smoke coverage:

- one `staged` run
- one `control_task_only` run
- both under `downstream/distill_family_study/configs/study/staged_smoke_a100_1gpu.yaml`
- plus one smallest-student smoke for each remaining non-BERT family under:
  - `downstream/distill_family_study/configs/study/staged_smoke_remaining_families_a100_1gpu.yaml`
- expected remaining-family students:
  - `pythia_410m`
  - `deberta_v3_xsmall`
  - `qwen2_5_0_5b`

Pass condition:

- the wrapper starts on GPU hardware
- `probe_teacher` completes
- `phate_teacher_target` completes
- `distill_sweep_grid` reaches real training
- phase-aware checkpoints are written
- `analysis_queue.jsonl` is emitted
- the remaining-families smoke materializes `6` runs:
  - `3` families x `2` regimes
- Qwen is scheduled last in family order

Evidence sources:

- SLURM logs for the smoke wrapper
- pipeline artifacts under `outputs/pipelines/staged_smoke_a100_1gpu_*`
- pipeline artifacts under `outputs/pipelines/staged_smoke_remaining_families_a100_1gpu_*`
- W&B runs for the smoke jobs
