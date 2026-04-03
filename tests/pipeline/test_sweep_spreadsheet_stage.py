import json
from pathlib import Path

from manylatents.pipeline.stages.base import StageContext
from manylatents.pipeline.stages.sweep_spreadsheet import SweepSpreadsheetStage


def test_sweep_spreadsheet_stage_writes_selected_columns(tmp_path):
    rows = [
        {
            "teacher_model": "teacher-A",
            "student_model": "student-A",
            "seed": 42,
            "lambda_align": 0.3,
            "val_loss": 1.1,
            "align_mse": 0.2,
            "run_dir": "/tmp/run0",
        },
        {
            "teacher_model": "teacher-A",
            "student_model": "student-A",
            "seed": 314,
            "lambda_align": 0.7,
            "val_loss": 1.0,
            "align_mse": 0.18,
            "run_dir": "/tmp/run1",
        },
    ]
    source_json = tmp_path / "sweep_results.json"
    source_json.write_text(json.dumps(rows), encoding="utf-8")

    context = StageContext(
        run_id="r_sheet",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "distill_sweep_grid": {
                "sweep_results_json": str(source_json),
            }
        },
    )

    stage = SweepSpreadsheetStage(
        stage_name="sweep_results_sheet",
        source_stage="distill_sweep_grid",
        source_key="sweep_results_json",
        output_prefix="final_sheet",
        include_columns=["teacher_model", "student_model", "seed", "lambda_align", "val_loss"],
    )
    result = stage.run(context=context, stage_dir=tmp_path / "sheet")

    outputs = result.outputs
    assert Path(outputs["spreadsheet_json"]).exists()
    assert Path(outputs["spreadsheet_csv"]).exists()
    assert Path(outputs["spreadsheet_summary"]).exists()

    payload = json.loads(Path(outputs["spreadsheet_json"]).read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert set(payload[0].keys()) == {
        "teacher_model",
        "student_model",
        "seed",
        "lambda_align",
        "val_loss",
    }
