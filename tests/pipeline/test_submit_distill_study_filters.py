from pathlib import Path

from omegaconf import OmegaConf

from downstream.distill_family_study.scripts.materialize_distill_study import materialize_study
from downstream.distill_family_study.scripts.submit_distill_study import _filter_runs


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "downstream" / "distill_family_study" / "configs"


def _materialized_runs() -> list[dict]:
    cfg = OmegaConf.load(CONFIG_ROOT / "study" / "within_family_publication.yaml")
    return [run.as_dict() for run in materialize_study(cfg)]


def test_filter_runs_supports_layer_scheme_and_lambda_align():
    runs = _materialized_runs()

    filtered = _filter_runs(
        runs,
        families=["t5"],
        student_keys=["t5_small"],
        layer_schemes=["penultimate_only"],
        lambda_align_values=[0.0, 0.5],
    )

    assert len(filtered) == 2
    assert {run["family"] for run in filtered} == {"t5"}
    assert {run["student_key"] for run in filtered} == {"t5_small"}
    assert {run["layer_scheme"] for run in filtered} == {"penultimate_only"}
    assert {float(run["lambda_align"]) for run in filtered} == {0.0, 0.5}


def test_filter_runs_can_build_one_student_per_family_slice():
    runs = _materialized_runs()

    filtered = _filter_runs(
        runs,
        families=["pythia", "qwen", "t5"],
        student_keys=["pythia_410m", "qwen2_5_0_5b", "t5_small"],
        layer_schemes=["penultimate_only"],
        lambda_align_values=[0.0, 0.5],
    )

    assert len(filtered) == 6
    assert {run["family"] for run in filtered} == {"pythia", "qwen", "t5"}
    assert {run["student_key"] for run in filtered} == {"pythia_410m", "qwen2_5_0_5b", "t5_small"}
    assert {run["layer_scheme"] for run in filtered} == {"penultimate_only"}
    assert {float(run["lambda_align"]) for run in filtered} == {0.0, 0.5}


def test_filter_runs_applies_limit_last():
    runs = _materialized_runs()

    filtered = _filter_runs(
        runs,
        families=["t5"],
        student_keys=["t5_small"],
        layer_schemes=["penultimate_only"],
        lambda_align_values=[0.0, 0.5],
        limit=1,
    )

    assert len(filtered) == 1
    assert filtered[0]["family"] == "t5"
    assert filtered[0]["student_key"] == "t5_small"
