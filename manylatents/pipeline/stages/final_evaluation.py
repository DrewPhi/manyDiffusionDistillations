"""Final report stage that enriches projection sheets with model metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult


class FinalEvaluationStage(PipelineStage):
    """Build a final spreadsheet with projection and probe-model metrics."""

    def __init__(
        self,
        stage_name: str,
        sheet_stage: str,
        sheet_key: str = "combined_sheet_json",
        probe_scores_key: str = "scores",
        output_subdir: str | None = None,
        output_prefix: str = "projection_final_evaluation",
    ):
        super().__init__(stage_name=stage_name)
        self.sheet_stage = sheet_stage
        self.sheet_key = sheet_key
        self.probe_scores_key = probe_scores_key
        self.output_subdir = output_subdir or stage_name
        self.output_prefix = output_prefix

    @staticmethod
    def _flatten_scalars(payload: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(FinalEvaluationStage._flatten_scalars(value, prefix=name))
                continue
            if isinstance(value, bool):
                out[name] = float(value)
                continue
            if isinstance(value, (int, float)):
                out[name] = float(value)
        return out

    @staticmethod
    def _pick_metric(flat_scores: Dict[str, float], candidates: List[str]) -> Tuple[str, float] | Tuple[None, None]:
        lowered = {k.lower(): k for k in flat_scores.keys()}

        for candidate in candidates:
            if candidate in lowered:
                key = lowered[candidate]
                return key, flat_scores[key]

        for lower_key, original_key in lowered.items():
            for candidate in candidates:
                if lower_key.endswith(candidate) or f".{candidate}" in lower_key:
                    return original_key, flat_scores[original_key]

        return None, None

    def _load_sheet_rows(self, context: StageContext) -> List[Dict[str, Any]]:
        sheet_artifacts = context.artifacts.get(self.sheet_stage)
        if sheet_artifacts is None:
            raise KeyError(f"No artifacts found for sheet stage '{self.sheet_stage}'")

        path = sheet_artifacts.get(self.sheet_key)
        if path is None:
            raise KeyError(f"Missing sheet key '{self.sheet_key}' in stage '{self.sheet_stage}' artifacts")

        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError("Combined sheet must be a JSON list of rows")
        return [dict(r) for r in rows]

    def _load_probe_scores(self, context: StageContext, source_stage: str) -> Dict[str, Any]:
        artifacts = context.artifacts.get(source_stage)
        if artifacts is None:
            return {}
        score_path = artifacts.get(self.probe_scores_key)
        if score_path is None:
            return {}
        try:
            payload = json.loads(Path(score_path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        rows = self._load_sheet_rows(context)
        stage_output_dir = stage_dir / self.output_subdir
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        source_stage_scores: Dict[str, Dict[str, float]] = {}
        source_stage_primary: Dict[str, Dict[str, Any]] = {}

        for row in rows:
            source_stage = row.get("source_stage")
            if not source_stage or source_stage in source_stage_scores:
                continue

            payload = self._load_probe_scores(context, str(source_stage))
            flat = self._flatten_scalars(payload)
            source_stage_scores[str(source_stage)] = flat

            acc_key, acc_val = self._pick_metric(
                flat,
                candidates=["test_accuracy", "eval_accuracy", "accuracy", "acc"],
            )
            ppl_key, ppl_val = self._pick_metric(
                flat,
                candidates=["test_perplexity", "eval_perplexity", "perplexity", "ppl"],
            )
            source_stage_primary[str(source_stage)] = {
                "model_accuracy_key": acc_key,
                "model_accuracy": acc_val,
                "model_perplexity_key": ppl_key,
                "model_perplexity": ppl_val,
            }

        enriched_rows: List[Dict[str, Any]] = []
        for row in rows:
            source_stage = str(row.get("source_stage", ""))
            out = dict(row)
            out.update(source_stage_primary.get(source_stage, {}))
            flat = source_stage_scores.get(source_stage, {})
            for key, value in flat.items():
                safe_key = key.replace(" ", "_").replace("/", "_")
                out[f"probe_score__{safe_key}"] = value
            enriched_rows.append(out)

        json_path = stage_output_dir / f"{self.output_prefix}.json"
        json_path.write_text(json.dumps(enriched_rows, indent=2, sort_keys=True), encoding="utf-8")

        csv_path = stage_output_dir / f"{self.output_prefix}.csv"
        if enriched_rows:
            fieldnames = sorted({k for row in enriched_rows for k in row.keys()})
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(enriched_rows)
        else:
            csv_path.write_text("", encoding="utf-8")

        summary = {
            "num_rows": len(enriched_rows),
            "num_source_stages_with_scores": sum(1 for _, v in source_stage_scores.items() if v),
            "source_stages": sorted(source_stage_scores.keys()),
        }
        summary_path = stage_output_dir / f"{self.output_prefix}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        return StageResult(
            outputs={
                "stage_output_dir": str(stage_output_dir),
                "final_evaluation_json": str(json_path),
                "final_evaluation_csv": str(csv_path),
                "final_evaluation_summary": str(summary_path),
            },
            metadata=summary,
        )
