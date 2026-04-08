# 04 Full Launch Gate

The full launch is acceptable when all of the following are true:

- targeted pytest passed
- the manifest gate passed
- the smoke wrapper exercised both regimes
- no new regime-specific or family-specific bug appeared in the smoke
- the intended study config is explicit and documented

For the full-Pile run, also require:

- study config:
  - `downstream/distill_family_study/configs/study/within_family_full_pile.yaml`
- stopping rule:
  - one epoch-equivalent via explicit `training.max_steps`
- continuation policy:
  - resume from saved student weights if needed, not exact optimizer-state resume

Full-Pile expected shape:

- `48` runs total
- `24` staged
- `24` control
- four families
- two layer schemes

Submission command:

```bash
python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/within_family_full_pile.yaml \
  --submit
```
