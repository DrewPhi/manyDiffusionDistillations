"""No-op stage used for pipeline scaffolding and tests."""
from __future__ import annotations

import json
from pathlib import Path

from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult


class NoOpStage(PipelineStage):
    def __init__(self, stage_name: str, message: str = "ok"):
        super().__init__(stage_name=stage_name)
        self.message = message

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        stage_dir.mkdir(parents=True, exist_ok=True)
        output_path = stage_dir / "noop.json"
        output_path.write_text(json.dumps({"message": self.message}, indent=2), encoding="utf-8")
        return StageResult(
            outputs={"noop_json": str(output_path)},
            metadata={"message": self.message},
        )
