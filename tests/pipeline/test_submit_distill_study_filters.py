from pathlib import Path

from omegaconf import OmegaConf

from downstream.distill_family_study.scripts.materialize_distill_study import materialize_study
from downstream.distill_family_study.scripts.submit_distill_study import _filter_runs


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "downstream" / "distill_family_study" / "configs"


def _materialized_runs() -> list[dict]:
    cfg = OmegaConf.load(CONFIG_ROOT / "study" / "within_family_publication.yaml")
    return [run.as_dict() for run in materialize_study(cfg)]


def test_filter_runs_supports_layer_scheme():
    runs = _materialized_runs()

    filtered = _filter_runs(
        runs,
        families=["bert"],
        student_keys=["bert_11m"],
        layer_schemes=["penultimate_only"],
    )

    assert len(filtered) == 2
    assert {run["family"] for run in filtered} == {"bert"}
    assert {run["student_key"] for run in filtered} == {"bert_11m"}
    assert {run["layer_scheme"] for run in filtered} == {"penultimate_only"}
    assert {run["training_regime"] for run in filtered} == {"staged", "control_task_only"}


def test_filter_runs_can_build_one_student_per_family_slice():
    runs = _materialized_runs()

    filtered = _filter_runs(
        runs,
        families=["pythia", "qwen", "bert", "deberta_v3"],
        student_keys=["pythia_410m", "qwen2_5_0_5b", "bert_11m", "deberta_v3_xsmall"],
        layer_schemes=["penultimate_only"],
    )

    assert len(filtered) == 8
    assert {run["family"] for run in filtered} == {"pythia", "qwen", "bert", "deberta_v3"}
    assert {run["student_key"] for run in filtered} == {"pythia_410m", "qwen2_5_0_5b", "bert_11m", "deberta_v3_xsmall"}
    assert {run["layer_scheme"] for run in filtered} == {"penultimate_only"}
    assert {run["training_regime"] for run in filtered} == {"staged", "control_task_only"}


def test_filter_runs_applies_limit_last():
    runs = _materialized_runs()

    filtered = _filter_runs(
        runs,
        families=["bert"],
        student_keys=["bert_11m"],
        layer_schemes=["penultimate_only"],
        limit=1,
    )

    assert len(filtered) == 1
    assert filtered[0]["family"] == "bert"
    assert filtered[0]["student_key"] == "bert_11m"
