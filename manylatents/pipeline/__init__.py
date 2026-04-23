"""Stage-based pipeline utilities (DEPRECATED).

This subpackage is deprecated as of the distillation-algo-module work. Experiment
orchestration and distillation-specific stages do not belong inside manylatents
core (see CLAUDE.md: library scope excludes sweep definitions and project-specific
pipelines). Imports still work for backward compatibility but will be removed in
a follow-up release.

New code should use:
    - manylatents/lightning/activation_snapshot.py  (ActivationSnapshot + from_model)
    - manylatents/algorithms/lightning/distillation.py  (Distillation LightningModule)
    - manylatents/algorithms/lightning/phase1_align.py  (align_on_snapshot helper)
    - manylatents/callbacks/staged_training.py  (StagedTrainingCallback)
"""
import warnings

warnings.warn(
    "manylatents.pipeline is deprecated and will be removed in a future release. "
    "Use manylatents.algorithms.lightning.distillation + "
    "manylatents.lightning.activation_snapshot instead. "
    "See CLAUDE.md for the migration path.",
    DeprecationWarning,
    # stacklevel=1 attributes the warning to this module (manylatents.pipeline),
    # so pyproject.toml's filterwarnings ["ignore::DeprecationWarning:manylatents.pipeline"]
    # matches and suppresses it cleanly under strict-warning CI.
    stacklevel=1,
)

from manylatents.pipeline.runner import run_stage_pipeline

__all__ = ["run_stage_pipeline"]
