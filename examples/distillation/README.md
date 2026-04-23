# Distillation algo module — usage

Minimal end-to-end example of the distillation primitives in this library.
All the surface area lives in:

- `manylatents.lightning.activation_snapshot.ActivationSnapshot` (data container + producer)
- `manylatents.algorithms.lightning.distillation.Distillation` (LightningModule)
- `manylatents.algorithms.lightning.phase1_align.align_on_snapshot` (phase1 helper)
- `manylatents.callbacks.staged_training.StagedTrainingCallback` (phase2→phase3 transition)

## The three-line idea

```python
from manylatents.lightning.activation_snapshot import ActivationSnapshot
from manylatents.algorithms.lightning.distillation import Distillation
from manylatents.algorithms.lightning.phase1_align import align_on_snapshot
from manylatents.callbacks.staged_training import StagedTrainingCallback
from lightning.pytorch import Trainer

# 1. Materialize a frozen reference of teacher activations at named layers.
snap = ActivationSnapshot.from_model(
    teacher, input_ids, attention_mask, sample_ids=probe_ids,
    layer_paths=["bert.encoder.layer.22"], reduction="mean",
)

# 2. Phase 1: align the student's chosen layer against the snapshot.
#    Imperative, NOT a trainer.fit - phase1 is a small closed-form pass.
phase1_losses = align_on_snapshot(
    student, snap,
    layer_pairs=[{"student": "bert.encoder.layer.10", "teacher": "bert.encoder.layer.22", "weight": 1.0}],
    n_steps=1000,
    optimizer_cfg={"learning_rate": 3e-4},
    device="cuda",
)

# 3. Phase 2 (frozen aligned layers) + Phase 3 (unfrozen) in one Lightning fit.
#    StagedTrainingCallback does the freeze/unfreeze; Distillation adds the
#    MSE regularizer sampled from `snap` at every training_step.
trainer = Trainer(
    max_steps=10000,
    callbacks=[StagedTrainingCallback(
        phase3_start_step=5000,
        # IMPORTANT: prefixes match the LightningModule's param names, not
        # the student's. Distillation wraps the student as self.student, so
        # the full param paths begin with "student.". See the callback
        # docstring for the full gotcha explanation.
        frozen_prefixes_phase2=["student.bert.encoder.layer.10"],
    )],
)
trainer.fit(Distillation(
    datamodule=task_dm,
    student=student,
    activation_snapshot=snap,
    layer_pairs=[{"student": "bert.encoder.layer.10", "teacher": "bert.encoder.layer.22", "weight": 1.0}],
    optimizer={"learning_rate": 1e-4},
    alignment_weight=1.0,
), datamodule=task_dm)
```

## Why phase1 is outside `trainer.fit`

Phase1 is a short pre-training pass over a fixed probe buffer with
pre-computed targets. Lightning's dataloader-per-step model doesn't fit —
there's nothing to stream. Making phase1 a free function taking any
`nn.Module` keeps it reusable for probing warmups, CKA warmups, and other
non-Lightning scenarios. The prior refactor tried to unify phase1 with
phase2/3 under a single `trainer.fit` and that structural choice caused
every friction we debugged.

## Consumer contracts worth internalizing

- **`snapshot.sample_ids` is an ID space.** Build your snapshot against
  the exact same probe IDs your datamodule emits (`batch["probe_ids"]` or
  equivalent). Mismatched ID spaces cause silent target-lookup errors.
- **One reduction per snapshot.** If you want `mean` at layer 11 and `cls`
  at layer 3, materialize two snapshots.
- **`frozen_prefixes_phase2` match the LightningModule's parameter names.**
  That includes the wrapping-attribute prefix - `"student.bert.encoder..."`,
  not `"bert.encoder..."`. The callback refuses to guess which attribute
  holds "the real model".
- **`Distillation.configure_optimizers` filters by `requires_grad`.** That
  is what makes the callback's freeze semantic actually freeze (a frozen
  param is simply absent from the optimizer).

## On-disk snapshot format

```python
snap.save("/scratch/you/snap.pt")       # single .pt dict, versioned
snap2 = ActivationSnapshot.load("/scratch/you/snap.pt")  # validates on read
```

Use this to decouple teacher-target materialization (run once, internet-enabled
or pre-cached) from student training (run many times, may be offline).

## Tamia smoke

`run_smoke.py` + `run_smoke.sbatch` exercise the full pipeline on real
`bert-large-uncased` / `bert-base-uncased` models against `pile_mini`. A
full-node H100 allocation, 1-hour budget, emits per-step losses to
`smoke_out/<job_id>/smoke_results.json` for post-hoc inspection.
