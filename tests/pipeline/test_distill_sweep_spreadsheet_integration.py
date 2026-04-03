import csv
import json
from pathlib import Path

import numpy as np

from manylatents.pipeline.stages.base import StageContext
from manylatents.pipeline.stages.distillation_sweep import DistillationSweepStage
from manylatents.pipeline.stages.sweep_spreadsheet import SweepSpreadsheetStage


def test_distill_sweep_to_spreadsheet_smoke_2x2(monkeypatch, tmp_path):
    aligned_target_path = tmp_path / "aligned_target.npy"
    np.save(aligned_target_path, np.ones((8, 4), dtype=np.float32))

    probe_ids_path = tmp_path / "probe_ids.json"
    probe_ids_path.write_text(json.dumps(list(range(8))), encoding="utf-8")

    context = StageContext(
        run_id="r_integration",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "probe_teacher": {
                "probe_ids_path": str(probe_ids_path),
            },
            "phate_teacher_target": {
                "aligned_target_path": str(aligned_target_path),
                "aligned_probe_ids_path": str(probe_ids_path),
            },
        },
    )

    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        teacher_stage="probe_teacher",
        teacher_target_stage="phate_teacher_target",
        sweep={
            "seed": [42, 314],
            "lambda_align": [0.0, 0.3],
            "teacher_model": ["teacher-A"],
            "student_model": ["student-A"],
            "learning_rate": [3e-4],
            "max_length": [32],
            "token_budget": [1024],
        },
    )

    def _fake_run_single(self, combo, aligned_targets, expected_probe_ids, run_output_dir):
        return {
            "teacher_model": combo["teacher_model_name_or_path"],
            "student_model": combo["student_model_name_or_path"],
            "seed": combo["seed"],
            "lambda_align": combo["lambda_align"],
            "val_loss": 1.0,
            "val_perplexity": 2.0,
            "align_mse": 0.5,
            "run_dir": str(run_output_dir),
            "ckpt_path": str(run_output_dir / "student_last.pt"),
            "metrics_path": str(run_output_dir / "metrics.json"),
        }

    monkeypatch.setattr(DistillationSweepStage, "_run_single_combo", _fake_run_single)

    distill_result = stage.run(context=context, stage_dir=tmp_path / "distill")
    context.artifacts["distill_sweep_grid"] = distill_result.outputs

    sheet_stage = SweepSpreadsheetStage(
        stage_name="sweep_results_sheet",
        source_stage="distill_sweep_grid",
        source_key="sweep_results_json",
        output_prefix="distill_sheet",
        include_columns=["seed", "lambda_align", "val_loss", "align_mse"],
    )
    sheet_result = sheet_stage.run(context=context, stage_dir=tmp_path / "sheet")

    csv_path = Path(sheet_result.outputs["spreadsheet_csv"])
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 5  # header + 4 combinations
