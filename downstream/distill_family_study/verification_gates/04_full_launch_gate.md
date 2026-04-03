# 04 Full Launch Gate

The large-cluster launch is acceptable when the following are true:

- family smokes passed
- manifest dry run passed
- mini launcher validation shows real end-to-end progress
- no new family-specific logic bug appears
- the intended study config is explicit

For the full-dataset run, also require:

- the stopping rule is explicit
- the continuation policy is explicit
- the operator knows whether the study is:
  - bounded-token
  - full-dataset

Current full-dataset policy:

- study config: `downstream/distill_family_study/configs/study/within_family_full_pile.yaml`
- stopping rule: one epoch-equivalent via explicit `training.max_steps`
- continuation: resume from saved student weights, not exact optimizer-state resume
