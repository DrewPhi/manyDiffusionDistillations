import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from manylatents.pipeline.stages.base import StageContext
from manylatents.pipeline.stages.probe_extraction import ProbeExtractionStage


def test_probe_extraction_stage_emits_contract_keys(monkeypatch, tmp_path):
    import manylatents.pipeline.stages.probe_extraction as probe_stage_module

    def fake_run_algorithm(run_cfg):
        raw_path = Path(
            OmegaConf.select(run_cfg, "callbacks.trainer.probe.save_path")
        )
        raw_path.mkdir(parents=True, exist_ok=True)
        np.save(raw_path / "teacher_layer.npy", np.random.randn(5, 4).astype(np.float32))
        return {
            "embeddings": np.random.randn(5, 4).astype(np.float32),
            "scores": {"dummy": 1.0},
        }

    class _DummyProbeDataset:
        indices = [100, 101, 102, 103, 104]

    class _DummyDM:
        probe_dataset = _DummyProbeDataset()

        def setup(self):
            return None

    def fake_instantiate_datamodule(run_cfg):
        return _DummyDM()

    monkeypatch.setattr(probe_stage_module, "run_algorithm", fake_run_algorithm)
    monkeypatch.setattr(probe_stage_module, "instantiate_datamodule", fake_instantiate_datamodule)

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "callbacks": {
                "trainer": {
                    "probe": {
                        "save_raw": True,
                        "save_path": "",
                    }
                }
            },
        }
    )
    context = StageContext(run_id="r_probe_contract", run_dir=tmp_path, cfg=cfg, artifacts={})
    stage = ProbeExtractionStage(stage_name="probe_teacher")

    result = stage.run(context=context, stage_dir=tmp_path / "probe")
    out = result.outputs

    assert Path(out["probe_ids_path"]).exists()
    probe_ids = json.loads(Path(out["probe_ids_path"]).read_text(encoding="utf-8"))
    assert probe_ids == [100, 101, 102, 103, 104]

    assert "teacher_activations_path" in out
    assert "teacher_diffop_path" in out
    assert "teacher_activations_paths" in out
    assert "teacher_diffop_paths" in out
    assert Path(out["teacher_activations_path"]).exists()
    assert Path(out["teacher_diffop_path"]).exists()
