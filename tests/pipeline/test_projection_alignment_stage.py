from pathlib import Path

import json
import numpy as np

from manylatents.pipeline.stages.base import StageContext
from manylatents.pipeline.stages.projection_alignment import ProjectionAlignmentStage


def test_projection_alignment_stage_fits_linear_map(tmp_path):
    rng = np.random.default_rng(7)

    x = rng.normal(size=(200, 6))
    w_true = rng.normal(size=(6, 2))
    b_true = np.array([0.2, -0.1])
    y = x @ w_true + b_true

    raw_dir = tmp_path / "probe_a_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / "acts.npy", x)

    target_path = tmp_path / "target.npy"
    np.save(target_path, y)

    context = StageContext(
        run_id="align1",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "probe_a": {"raw_activations_dir": str(raw_dir)},
            "phate_targets": {"phate_targets": [str(target_path)]},
        },
    )

    stage = ProjectionAlignmentStage(
        stage_name="projection_alignment",
        source_stages=["probe_a"],
        target_stage="phate_targets",
        target_key="phate_targets",
        target_selection="single",
        target_index=0,
        test_fraction=0.2,
        random_state=42,
        ridge_alpha=1e-6,
    )

    result = stage.run(context=context, stage_dir=tmp_path / "projection_alignment")

    report_path = Path(result.outputs["projection_report_json"])
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report) == 1
    assert report[0]["test_r2"] > 0.98


def test_projection_alignment_stage_uses_fallback_embeddings(tmp_path):
    rng = np.random.default_rng(13)

    x = rng.normal(size=(120, 4))
    y = x[:, :2]

    emb_path = tmp_path / "emb.npy"
    target_path = tmp_path / "target.npy"
    np.save(emb_path, x)
    np.save(target_path, y)

    context = StageContext(
        run_id="align2",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "probe_b": {"embeddings": str(emb_path)},
            "phate_targets": {"phate_targets": [str(target_path)]},
        },
    )

    stage = ProjectionAlignmentStage(
        stage_name="projection_alignment",
        source_stages=["probe_b"],
        target_stage="phate_targets",
        ridge_alpha=1e-6,
    )

    result = stage.run(context=context, stage_dir=tmp_path / "projection_alignment")
    csv_path = Path(result.outputs["projection_report_csv"])
    assert csv_path.exists()
