# Push Plan

This is the concrete repo-cleanup and handoff plan for pushing the distillation study to a remote repo and then running the larger job on another cluster.

## Goal

End up with a branch that contains:

- the working stage-pipeline code
- the downstream study configs and launch scripts
- the tests that defend the launcher/config behavior
- one clear operator-facing runbook

And does not contain:

- agent memory files
- generated results
- local-only scratch outputs
- overlapping markdown histories that say the same thing in different ways

## Keep

These should stay in the pushed branch.

Code and configs:

- `manylatents/pipeline/`
- `manylatents/data/text.py`
- `manylatents/lightning/hf_trainer.py`
- `manylatents/lightning/callbacks/activation_tracker.py`
- `manylatents/lightning/hooks.py`
- `manylatents/experiment.py`
- `manylatents/main.py`
- `manylatents/api.py`
- `manylatents/configs/config.py`
- `manylatents/configs/config.yaml`
- `manylatents/configs/logger/wandb.yaml`
- `manylatents/configs/trainer/logger/wandb.yaml`
- `manylatents/configs/data/pile_uncopyrighted.yaml`
- `manylatents/configs/stage_pipeline/`
- the experiment presets in `manylatents/configs/experiment/` that are still relevant to the current pipeline

Downstream study assets:

- `downstream/distill_family_study/configs/`
- `downstream/distill_family_study/scripts/`
- `downstream/distill_family_study/README.md`

Tests:

- `tests/pipeline/`
- `tests/test_text_datamodule_pile_controls.py`
- `manylatents/lightning/tests/test_hf_trainer.py`
- `manylatents/lightning/tests/test_hf_trainer_from_config.py`
- `manylatents/lightning/tests/test_hooks.py`

Project metadata:

- `pyproject.toml`
- `uv.lock`
- `README.md`
- `FINAL_TESTS.md`
- `PUSH_LIST.md`
- `PUSH_PLAN.md`

## Drop From The Push

These should not be part of the pushed branch used for the big-cluster run.

Generated artifacts:

- `results/`
- `representationvisualizations/`
- `scripts/outputs/`
- local pipeline output directories
- local SLURM output files unless there is a deliberate reason to version a tiny sample

Agent-only or temporary notes:

- `memory.md`
- `notesformorning.md`
- `README_CODEX_MANYLATENTS_MIGRATION.md`

Anything under `.codex/`

## Consolidate

Right now there are too many overlapping markdown notes. Reduce them to this structure:

1. `downstream/distill_family_study/README.md`
   - how to run the study
   - required environment variables
   - expected dataset/cache/W&B requirements
   - the main launcher commands

2. `FINAL_TESTS.md`
   - final launch gates
   - what must pass before a large submission

3. `PUSH_LIST.md`
   - what to push vs not push

4. `PUSH_PLAN.md`
   - exact cleanup steps before push

Everything else in the current note pile should either be deleted from the push or folded into one of the files above.

## Cleanup Sequence

### Phase 1: Freeze The Real Experiment Surface

- confirm which downstream study config is the real one to hand off
- confirm which launcher scripts are current
- confirm which tests still match the current launcher behavior

### Phase 2: Remove Generated Noise

- stop tracking local result artifacts
- stop tracking local visualizations and scratch outputs
- make sure `.gitignore` covers any generated directories that are currently polluting status

### Phase 3: Collapse Documentation

- move any still-useful handoff details out of `memory.md`
- move any still-useful “morning check” logic into the downstream README or `FINAL_TESTS.md`
- remove `README_CODEX_MANYLATENTS_MIGRATION.md` from the pushed branch unless it is rewritten into a short migration note

### Phase 4: Prepare The Big-Cluster Config Surface

- keep the publication study config for the current bounded matrix
- add the full-Pile study config for the actual large launch
- document the difference between:
  - bounded-token study
  - full-dataset study

### Phase 5: Verify Push Readiness

- run the targeted tests that validate the study config and materializer
- run one dry-run manifest generation for the chosen study config
- check that the downstream README gives a new operator enough information to launch without reading local notes

## Open Decisions Before Push

These need to be explicit in the pushed docs/configs.

1. Which study config is the primary handoff target?
   - bounded publication matrix
   - full-Pile matrix

2. For full-Pile runs, what is the stopping rule?
   - one epoch over train split
   - explicit `max_steps`
   - family-specific stop budgets

3. Which docs are canonical?
   - there should be one answer, not five

## Continuation Policy

For the big-cluster launch, the intended training policy is:

- run one epoch-equivalent first
- stop and inspect results
- only continue further if the first pass justifies the extra compute

Important implementation note:

- the current pipeline supports stage-level resume and model-weight continuation
- it does not currently guarantee exact optimizer-state and scheduler-state resume for the distillation sweep

That means a later continuation should be described as:

- resume from saved student weights

It should not be described as:

- exact uninterrupted training-state resume

This is acceptable for the current experimental goal, but it should be stated clearly in the pushed docs so nobody assumes the second pass is mathematically identical to one uninterrupted multi-epoch run.

## Immediate Next Actions

1. Keep `PUSH_LIST.md` and `PUSH_PLAN.md`.
2. Add the large-run study config.
3. Update the downstream README to point at the real study configs and launcher commands.
4. Remove local notes and generated artifacts from the pushed branch.
