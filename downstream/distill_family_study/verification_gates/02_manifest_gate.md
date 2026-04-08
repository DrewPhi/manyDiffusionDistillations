# 02 Manifest Gate

The materializer must produce the expected run matrix before any large submission.

Expected shapes:

- smoke study:
  - `2` total runs
  - `1` staged
  - `1` control
- full-Pile study:
  - `48` total runs
  - `24` staged
  - `24` control
  - balanced across `pythia`, `qwen`, `bert`, and `deberta_v3`

Pass condition:

- manifest dry run completes without submission
- `submission_manifest.json` exists
- `run_specs/` exists
- run count matches expectation
- run names are unique
- regime counts match expectation

Evidence sources:

- materialized manifests under `results/study_manifests/`
- dry-run output from `submit_distill_study.py`
