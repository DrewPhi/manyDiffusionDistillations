# 03 Mini Launcher Gate

Before a large launch, the real launcher path should be exercised on the one-GPU smoke wrapper.

Current mini gate:

- study config: `downstream/distill_family_study/configs/study/staged_smoke_a100_1gpu.yaml`
- expected run count: `2`
- regimes:
  - `staged`
  - `control_task_only`

Pass condition:

- the wrapper submits both smoke runs
- the staged run reaches Phase 1 training
- the control run reaches task-only training
- at least one analysis checkpoint is queued for each regime
- the launcher path works without manual patching between runs

Evidence sources:

- `results/study_manifests/staged_smoke_a100_1gpu/`
- actual pipeline outputs under `outputs/pipelines/staged_smoke_a100_1gpu_*`
- SLURM logs for the smoke wrapper
