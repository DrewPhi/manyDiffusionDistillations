"""Base stage abstractions for stage-based pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class StageContext:
    """Execution context shared across pipeline stages."""

    run_id: str
    run_dir: Path
    cfg: Any
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class StageResult:
    """Standardized stage result payload."""

    outputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineStage:
    """Base class for all stage implementations."""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        raise NotImplementedError
