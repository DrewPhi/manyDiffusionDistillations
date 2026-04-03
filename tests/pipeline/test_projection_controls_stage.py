from pathlib import Path
import json

import numpy as np

from manylatents.pipeline.stages.base import StageContext
from manylatents.pipeline.stages.projection_controls import ProjectionControlsStage


def test_projection_controls_stage_builds_combined_sheet(tmp_path):
    rng = np.random.default_rng(0)

    x = rng.normal(size=(100, 5))
    y = x[:, :2] * 0.5

    raw_dir = tmp_path / "probe_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / "acts.npy", x)

    target_path = tmp_path / "target.npy"
    np.save(target_path, y)

    primary_report = [
        {
            "source_stage": "probe_a",
            "target_path": str(target_path),
            "n_test": 20,
            "test_mse": 0.01,
            "test_r2": 0.95,
            "test_cosine_mean": 0.9,
        }
    ]
    primary_path = tmp_path / "primary_report.json"
    primary_path.write_text(json.dumps(primary_report), encoding="utf-8")

    context = StageContext(
        run_id="controls",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "projection_alignment": {
                "projection_report_json": str(primary_path),
            },
            "probe_a": {
                "raw_activations_dir": str(raw_dir),
            },
            "phate_targets": {
                "phate_targets": [str(target_path)],
            },
        },
    )

    stage = ProjectionControlsStage(
        stage_name="projection_controls",
        projection_stage="projection_alignment",
        source_stages=["probe_a"],
        target_stage="phate_targets",
        target_key="phate_targets",
        random_state=42,
    )

    result = stage.run(context=context, stage_dir=tmp_path / "projection_controls")
    sheet_path = Path(result.outputs["projection_experiment_sheet_json"])
    assert sheet_path.exists()

    rows = json.loads(sheet_path.read_text(encoding="utf-8"))
    conditions = {row["condition"] for row in rows}
    assert "matched" in conditions
    assert "no_match" in conditions
    assert "random_target" in conditions
    assert "mean_baseline" in conditions

    csv_path = Path(result.outputs["projection_experiment_sheet_csv"])
    assert csv_path.exists()
