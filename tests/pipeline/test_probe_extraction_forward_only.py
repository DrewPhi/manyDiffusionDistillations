from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from manylatents.pipeline.stages.base import StageContext
from manylatents.pipeline.stages.probe_extraction import ProbeExtractionStage


class _DummyProbeDataset:
    indices = [10, 11, 12]


class _DummyDM:
    def __init__(self):
        self.probe_dataset = _DummyProbeDataset()
        self.setup_stage = None

    def setup(self, stage=None):
        self.setup_stage = stage
        return None

    def probe_dataloader(self):
        batch = {
            "input_ids": torch.randint(low=0, high=8, size=(3, 4)),
            "attention_mask": torch.ones(3, 4, dtype=torch.long),
            "labels": torch.zeros(3, 4, dtype=torch.long),
        }
        return [batch]


class _ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, input_ids, attention_mask=None):
        x = input_ids.float()
        x = self.transformer.h[0](x)
        x = self.transformer.h[1](x)
        x = self.proj(x)
        return x


class _DummyLightningAlgo:
    def __init__(self):
        self.network = _ToyNet()

    def configure_model(self):
        return None


def test_probe_extraction_forward_only_uses_probe_loader(monkeypatch, tmp_path):
    import manylatents.pipeline.stages.probe_extraction as probe_stage_module

    dummy_dm = _DummyDM()
    monkeypatch.setattr(probe_stage_module, "instantiate_datamodule", lambda cfg: dummy_dm)
    monkeypatch.setattr(probe_stage_module, "instantiate_algorithm", lambda cfg, datamodule=None: _DummyLightningAlgo())

    def _should_not_run_algorithm(_cfg):
        raise AssertionError("run_algorithm should not be called in forward-only path")

    monkeypatch.setattr(probe_stage_module, "run_algorithm", _should_not_run_algorithm)

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "algorithms": {"lightning": {"_target_": "ignored.for.test"}},
            "callbacks": {
                "trainer": {
                    "probe": {
                        "save_raw": True,
                        "save_path": "",
                        "layer_specs": [
                            {
                                "_target_": "manylatents.lightning.hooks.LayerSpec",
                                "path": "transformer.h[-2]",
                                "extraction_point": "output",
                                "reduce": "mean",
                            }
                        ],
                    }
                }
            },
        }
    )

    context = StageContext(run_id="r_probe_forward", run_dir=tmp_path, cfg=cfg, artifacts={})
    stage = ProbeExtractionStage(stage_name="probe_teacher", forward_only=True)

    result = stage.run(context=context, stage_dir=tmp_path / "probe")
    out = result.outputs

    assert Path(out["teacher_activations_path"]).exists()
    assert Path(out["teacher_diffop_path"]).exists()
    assert Path(out["probe_ids_path"]).exists()

    emb = np.load(out["embeddings"])
    assert emb.shape[0] == 3
    assert dummy_dm.setup_stage == "probe"


def test_probe_extraction_forward_only_accepts_plain_layer_spec_mapping(monkeypatch, tmp_path):
    import manylatents.pipeline.stages.probe_extraction as probe_stage_module

    monkeypatch.setattr(probe_stage_module, "instantiate_datamodule", lambda cfg: _DummyDM())
    monkeypatch.setattr(probe_stage_module, "instantiate_algorithm", lambda cfg, datamodule=None: _DummyLightningAlgo())
    monkeypatch.setattr(
        probe_stage_module,
        "run_algorithm",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("run_algorithm should not be called")),
    )

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "algorithms": {"lightning": {"_target_": "ignored.for.test"}},
            "callbacks": {
                "trainer": {
                    "probe": {
                        "save_raw": True,
                        "save_path": "",
                        "layer_specs": [
                            {
                                "path": "transformer.h[-2]",
                                "extraction_point": "output",
                                "reduce": "mean",
                            }
                        ],
                    }
                }
            },
        }
    )

    context = StageContext(run_id="r_probe_forward_plain", run_dir=tmp_path, cfg=cfg, artifacts={})
    stage = ProbeExtractionStage(stage_name="probe_teacher", forward_only=True)
    result = stage.run(context=context, stage_dir=tmp_path / "probe")
    assert Path(result.outputs["teacher_activations_path"]).exists()
