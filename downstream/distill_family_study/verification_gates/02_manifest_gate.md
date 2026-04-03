# 02 Manifest Gate

The study materializer must produce the expected run matrix before submission.

Bounded publication study expected shape:

- `54` total runs
- `18` per family
- layer schemes:
  - `penultimate_only`
  - `second_plus_penultimate`

Pass condition:

- manifest dry run completes without submission
- `submission_manifest.json` exists
- `run_specs/` exists
- run count matches expectation
- family counts match expectation
- run names are unique

Evidence sources:

- `results/final_validation/*/study_manifest/submission_manifest.json`
- `results/final_validation/*/validation_report.json`
