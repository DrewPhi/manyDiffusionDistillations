# 03 Mini Launcher Gate

Before a large launch, the real launcher path should be exercised on a small study slice.

Current intended mini slice:

- families:
  - `pythia`
  - `qwen`
  - `t5`
- one student per family
- one layer scheme
- two lambda values
- `6` runs total

Pass condition:

- the top-level mini-launch wrapper runs
- child jobs are submitted from the real launcher path
- at least one child run reaches real training, not just setup
- ideally the full 6-run batch finishes and writes the mini validation report
- aggregation and plotting either pass or any failure is clearly downstream of runtime scale rather than config wiring

Evidence sources:

- `results/mini_launch_validation/*/report/mini_validation_report.json`
- `results/mini_launch_validation/*/manifest/submission_manifest.json`
- actual pipeline outputs under `outputs/pipelines/within_family_publication_*`
