from pathlib import Path

import numpy as np
import torch

from manylatents.pipeline.stages.base import StageContext
from manylatents.pipeline.stages.phate_aligned_target import PHATEAlignedTargetStage


class _DummyPHATE:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, x: torch.Tensor):
        # Deterministic output with requested n_components.
        arr = x.detach().cpu().numpy()
        n_components = int(self.kwargs.get("n_components", 2))
        cols = min(arr.shape[1], n_components)
        out = arr[:, :cols]
        if cols < n_components:
            out = np.pad(out, ((0, 0), (0, n_components - cols)))
        return torch.from_numpy(out.astype(np.float32))


def test_phate_aligned_target_stage_builds_aligned_targets(monkeypatch, tmp_path):
    import manylatents.pipeline.stages.phate_aligned_target as aligned_stage_module

    monkeypatch.setattr(aligned_stage_module, "PHATEModule", _DummyPHATE)

    n = 6
    diffusion = np.eye(n, dtype=np.float32)
    diff_path = tmp_path / "teacher_diffop.npy"
    np.save(diff_path, diffusion)

    # Simulate teacher raw activations with larger hidden dimension than n_components
    teacher_acts = np.random.default_rng(42).normal(size=(n, 8)).astype(np.float32)
    raw_dir = tmp_path / "raw_activations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "transformer_h_m1_step0.npy"
    np.save(raw_path, teacher_acts)

    context = StageContext(
        run_id="r_phate_align",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "probe_teacher": {
                "diffusion_operators": [str(diff_path)],
                "raw_activations_dir": str(raw_dir),
            }
        },
    )

    stage = PHATEAlignedTargetStage(
        stage_name="phate_teacher_target",
        source_stage="probe_teacher",
        diffusion_key="diffusion_operators",
        n_components=4,
    )

    result = stage.run(context=context, stage_dir=tmp_path / "aligned")
    outputs = result.outputs

    assert "aligned_targets" in outputs
    assert "phate_targets" in outputs  # compatibility alias
    assert len(outputs["aligned_targets"]) == 1
    assert Path(outputs["aligned_targets_index"]).exists()

    aligned = np.load(outputs["aligned_targets"][0])
    assert aligned.shape == (n, 4)


def test_phate_aligned_target_stage_builds_per_layer_targets(monkeypatch, tmp_path):
    import manylatents.pipeline.stages.phate_aligned_target as aligned_stage_module

    monkeypatch.setattr(aligned_stage_module, "PHATEModule", _DummyPHATE)

    n = 5
    raw_dir = tmp_path / "raw_activations"
    raw_dir.mkdir(parents=True, exist_ok=True)

    layer_names = ["transformer_h_0", "transformer_h_1"]
    diffusion_paths = []
    for idx, layer_name in enumerate(layer_names):
        diff_path = tmp_path / f"{layer_name}_step0__diffop.npy"
        np.save(diff_path, np.eye(n, dtype=np.float32) * float(idx + 1))
        diffusion_paths.append(str(diff_path))

        raw_path = raw_dir / f"{layer_name}_step0.npy"
        acts = np.random.default_rng(idx).normal(size=(n, 6)).astype(np.float32)
        np.save(raw_path, acts)

    context = StageContext(
        run_id="r_phate_align_multi",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "probe_teacher": {
                "diffusion_operators": diffusion_paths,
                "raw_activations_dir": str(raw_dir),
            }
        },
    )

    stage = PHATEAlignedTargetStage(
        stage_name="phate_teacher_target",
        source_stage="probe_teacher",
        diffusion_key="diffusion_operators",
        n_components=3,
    )

    result = stage.run(context=context, stage_dir=tmp_path / "aligned")
    outputs = result.outputs

    assert outputs["aligned_target_layers"] == layer_names
    assert set(outputs["aligned_target_paths_by_layer"].keys()) == set(layer_names)
    for layer_name in layer_names:
        aligned = np.load(outputs["aligned_target_paths_by_layer"][layer_name])
        assert aligned.shape == (n, 3)
