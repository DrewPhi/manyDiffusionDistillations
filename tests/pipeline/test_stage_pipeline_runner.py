from pathlib import Path

from omegaconf import OmegaConf

from manylatents.pipeline.runner import run_stage_pipeline


def _cfg(tmp_path):
    return OmegaConf.create(
        {
            "name": "test_run",
            "output_dir": str(tmp_path / "outputs"),
            "stage_pipeline": {
                "run_id": "pipeline_test",
                "output_root": str(tmp_path / "pipeline_outputs"),
                "resume": True,
                "stages": [
                    {
                        "name": "first",
                        "target": "manylatents.pipeline.stages.noop.NoOpStage",
                        "depends_on": [],
                        "params": {"message": "a"},
                    },
                    {
                        "name": "second",
                        "target": "manylatents.pipeline.stages.noop.NoOpStage",
                        "depends_on": ["first"],
                        "params": {"message": "b"},
                    },
                ],
            },
        }
    )


def test_stage_pipeline_executes_and_writes_manifests(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_stage_pipeline(cfg)

    assert result["stage_order"] == ["first", "second"]
    assert result["executed_stages"] == ["first", "second"]
    assert result["skipped_stages"] == []

    for stage_name in ["first", "second"]:
        manifest_path = result["manifests"][stage_name]
        assert Path(manifest_path).exists()


def test_stage_pipeline_resume_skips_existing_stages(tmp_path):
    cfg = _cfg(tmp_path)
    run_stage_pipeline(cfg)
    second_run = run_stage_pipeline(cfg)

    assert second_run["executed_stages"] == []
    assert second_run["skipped_stages"] == ["first", "second"]
