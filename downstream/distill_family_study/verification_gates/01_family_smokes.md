# 01 Family Smokes

All three family preflights should pass before trusting the generalized study path.

Required families:

- Pythia
- Qwen
- T5

Pass condition:

- the job starts on GPU hardware
- W&B initialization succeeds if enabled
- `probe_teacher` completes
- `phate_teacher_target` completes
- `distill_sweep_grid` completes
- `sweep_results_sheet` completes

Evidence sources:

- final validation JSON under `results/final_validation/*/validation_report.json`
- SLURM logs for the family preflight jobs
