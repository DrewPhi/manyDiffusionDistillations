"""Combine multiple JSON report sheets into one aggregate report."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult


class ReportCombineStage(PipelineStage):
    """Combine row-based JSON sheets from upstream stages."""

    def __init__(
        self,
        stage_name: str,
        source_stages: List[str],
        source_key: str = "projection_experiment_sheet_json",
        output_subdir: str | None = None,
        output_prefix: str = "projection_experiment_sheet",
    ):
        super().__init__(stage_name=stage_name)
        self.source_stages = source_stages
        self.source_key = source_key
        self.output_subdir = output_subdir or stage_name
        self.output_prefix = output_prefix

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        stage_output_dir = stage_dir / self.output_subdir
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        combined: List[Dict[str, Any]] = []
        for source_stage in self.source_stages:
            artifacts = context.artifacts.get(source_stage)
            if artifacts is None:
                raise KeyError(f"No artifacts found for source stage '{source_stage}'")
            path = artifacts.get(self.source_key)
            if path is None:
                raise KeyError(f"Missing key '{self.source_key}' in stage '{source_stage}' artifacts")

            rows = json.loads(Path(path).read_text(encoding="utf-8"))
            for row in rows:
                row = dict(row)
                row["source_report_stage"] = source_stage
                combined.append(row)

        json_path = stage_output_dir / f"{self.output_prefix}.json"
        json_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")

        csv_path = stage_output_dir / f"{self.output_prefix}.csv"
        if combined:
            fieldnames = sorted({k for row in combined for k in row.keys()})
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(combined)
        else:
            csv_path.write_text("", encoding="utf-8")

        return StageResult(
            outputs={
                "stage_output_dir": str(stage_output_dir),
                "combined_sheet_json": str(json_path),
                "combined_sheet_csv": str(csv_path),
            },
            metadata={
                "num_rows": len(combined),
                "num_sources": len(self.source_stages),
            },
        )
