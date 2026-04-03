from pathlib import Path

import numpy as np
import torch

from manylatents.pipeline.stages.base import StageContext
from manylatents.pipeline.stages.phate_target import PHATETargetStage


class _DummyPHATE:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, x: torch.Tensor):
        # Deterministic 2D embedding from first two columns
        arr = x.detach().cpu().numpy()
        if arr.shape[1] < 2:
            arr = np.pad(arr, ((0, 0), (0, 2 - arr.shape[1])))
        return torch.from_numpy(arr[:, :2].astype(np.float32))


def test_phate_target_stage_builds_targets(monkeypatch, tmp_path):
    import manylatents.pipeline.stages.phate_target as phate_stage_module

    monkeypatch.setattr(phate_stage_module, "PHATEModule", _DummyPHATE)

    merged_op = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    source_path = tmp_path / "merged.npy"
    np.save(source_path, merged_op)

    context = StageContext(
        run_id="r3",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "diffusion_merge": {
                "merged_diffusion_operators": [str(source_path)],
            }
        },
    )

    stage = PHATETargetStage(
        stage_name="phate_targets",
        source_stage="diffusion_merge",
        source_key="merged_diffusion_operators",
        n_components=2,
    )

    result = stage.run(context=context, stage_dir=tmp_path / "phate")
    outputs = result.outputs

    assert len(outputs["phate_targets"]) == 1
    target = np.load(Path(outputs["phate_targets"][0]))
    assert target.shape == (3, 2)
    assert Path(outputs["phate_targets_index"]).exists()
