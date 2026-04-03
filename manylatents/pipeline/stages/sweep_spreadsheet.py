"""Spreadsheet aggregation stage for distillation sweep outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult


class SweepSpreadsheetStage(PipelineStage):
    """Build final spreadsheet artifacts from sweep result rows."""

    def __init__(
        self,
        stage_name: str,
        source_stage: str,
        source_key: str = "sweep_results_json",
        output_subdir: str | None = None,
        output_prefix: str = "sweep_results",
        include_columns: List[str] | None = None,
    ):
        super().__init__(stage_name=stage_name)
        self.source_stage = source_stage
        self.source_key = source_key
        self.output_subdir = output_subdir or stage_name
        self.output_prefix = output_prefix
        self.include_columns = include_columns

    @staticmethod
    def _load_rows(path: Path) -> List[Dict[str, Any]]:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError("Sweep rows payload must be a JSON list")
        return [dict(r) for r in rows]

    def _select_columns(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.include_columns:
            return rows
        selected: List[Dict[str, Any]] = []
        for row in rows:
            selected.append({k: row.get(k) for k in self.include_columns})
        return selected

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        source_artifacts = context.artifacts.get(self.source_stage)
        if source_artifacts is None:
            raise KeyError(f"No artifacts found for source stage '{self.source_stage}'")

        source_path = source_artifacts.get(self.source_key)
        if source_path is None:
            raise KeyError(
                f"Missing key '{self.source_key}' in stage '{self.source_stage}' artifacts"
            )

        rows = self._load_rows(Path(source_path))
        rows = self._select_columns(rows)

        stage_output_dir = stage_dir / self.output_subdir
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        json_path = stage_output_dir / f"{self.output_prefix}.json"
        json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

        csv_path = stage_output_dir / f"{self.output_prefix}.csv"
        if rows:
            fieldnames = self.include_columns or sorted({k for r in rows for k in r.keys()})
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            csv_path.write_text("", encoding="utf-8")

        summary = {
            "num_rows": len(rows),
            "num_columns": len((self.include_columns or []))
            if self.include_columns
            else len({k for r in rows for k in r.keys()}),
            "source_stage": self.source_stage,
            "source_key": self.source_key,
        }
        summary_path = stage_output_dir / f"{self.output_prefix}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        return StageResult(
            outputs={
                "stage_output_dir": str(stage_output_dir),
                "spreadsheet_json": str(json_path),
                "spreadsheet_csv": str(csv_path),
                "spreadsheet_summary": str(summary_path),
                # compatibility aliases for downstream reporting expectations
                "combined_sheet_json": str(json_path),
                "combined_sheet_csv": str(csv_path),
            },
            metadata=summary,
        )
