from pathlib import Path

import numpy as np

from manylatents.pipeline.stages.base import StageContext
from manylatents.pipeline.stages.diffusion_merge import DiffusionMergeStage


def _write_op(path: Path, value: float) -> str:
    op = np.array(
        [
            [value, 1.0 - value],
            [1.0 - value, value],
        ]
    )
    np.save(path, op)
    return str(path)


def test_diffusion_merge_stage_all(tmp_path):
    op_a = _write_op(tmp_path / "a.npy", 0.8)
    op_b = _write_op(tmp_path / "b.npy", 0.6)

    context = StageContext(
        run_id="r1",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "probe_a": {"diffusion_operators": [op_a]},
            "probe_b": {"diffusion_operators": [op_b]},
        },
    )

    stage = DiffusionMergeStage(
        stage_name="merge",
        source_stages=["probe_a", "probe_b"],
        strategy="frobenius_mean",
        merge_mode="all",
    )
    result = stage.run(context=context, stage_dir=tmp_path / "merge")

    merged_paths = result.outputs["merged_diffusion_operators"]
    assert len(merged_paths) == 1

    merged = np.load(merged_paths[0])
    expected = np.array([[0.7, 0.3], [0.3, 0.7]])
    np.testing.assert_allclose(merged, expected, rtol=1e-6, atol=1e-6)


def test_diffusion_merge_stage_adjacent_pairs(tmp_path):
    op_a = _write_op(tmp_path / "a.npy", 0.9)
    op_b = _write_op(tmp_path / "b.npy", 0.7)
    op_c = _write_op(tmp_path / "c.npy", 0.5)

    context = StageContext(
        run_id="r2",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "probe_a": {"diffusion_operators": [op_a]},
            "probe_b": {"diffusion_operators": [op_b]},
            "probe_c": {"diffusion_operators": [op_c]},
        },
    )

    stage = DiffusionMergeStage(
        stage_name="merge",
        source_stages=["probe_a", "probe_b", "probe_c"],
        strategy="frobenius_mean",
        merge_mode="adjacent_pairs",
    )
    result = stage.run(context=context, stage_dir=tmp_path / "merge_pairs")

    assert len(result.outputs["merged_diffusion_operators"]) == 2
