"""Stage implementations for config-native pipelines."""

from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult
from manylatents.pipeline.stages.diffusion_merge import DiffusionMergeStage
from manylatents.pipeline.stages.distillation_sweep import DistillationSweepStage
from manylatents.pipeline.stages.final_evaluation import FinalEvaluationStage
from manylatents.pipeline.stages.noop import NoOpStage
from manylatents.pipeline.stages.phate_aligned_target import PHATEAlignedTargetStage
from manylatents.pipeline.stages.phate_target import PHATETargetStage
from manylatents.pipeline.stages.projection_alignment import ProjectionAlignmentStage
from manylatents.pipeline.stages.projection_controls import ProjectionControlsStage
from manylatents.pipeline.stages.probe_extraction import ProbeExtractionStage
from manylatents.pipeline.stages.report_combine import ReportCombineStage
from manylatents.pipeline.stages.sweep_spreadsheet import SweepSpreadsheetStage

__all__ = [
    "PipelineStage",
    "StageContext",
    "StageResult",
    "DiffusionMergeStage",
    "DistillationSweepStage",
    "FinalEvaluationStage",
    "NoOpStage",
    "PHATEAlignedTargetStage",
    "PHATETargetStage",
    "ProjectionAlignmentStage",
    "ProjectionControlsStage",
    "ProbeExtractionStage",
    "ReportCombineStage",
    "SweepSpreadsheetStage",
]
