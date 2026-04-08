import json
from pathlib import Path

from downstream.distill_family_study.scripts.aggregate_distill_study import (
    _best_per_student,
    _group_mean,
    _load_result_rows,
)


def test_group_mean_aggregates_family_metrics():
    rows = [
        {"family": "pythia", "layer_scheme": "penultimate_only", "val_loss": 2.0, "align_mse": 0.4},
        {"family": "pythia", "layer_scheme": "penultimate_only", "val_loss": 4.0, "align_mse": 0.6},
    ]

    summary = _group_mean(rows, group_keys=["family", "layer_scheme"], metric_keys=["val_loss", "align_mse"])

    assert summary == [
        {
            "family": "pythia",
            "layer_scheme": "penultimate_only",
            "num_runs": 2,
            "val_loss_mean": 3.0,
            "align_mse_mean": 0.5,
        }
    ]


def test_best_per_student_picks_lowest_val_loss():
    rows = [
        {"family": "pythia", "student_model": "student-a", "val_loss": 3.0, "align_mse": 0.2},
        {"family": "pythia", "student_model": "student-a", "val_loss": 2.0, "align_mse": 0.9},
        {"family": "qwen", "student_model": "student-b", "val_loss": 5.0, "align_mse": 0.1},
    ]

    winners = _best_per_student(rows)

    assert len(winners) == 2
    assert winners[0]["val_loss"] == 2.0
    assert winners[1]["student_model"] == "student-b"


def test_load_result_rows_reads_expected_study_run_locations(tmp_path):
    repo_root = tmp_path
    manifest_dir = repo_root / "results" / "study_manifests" / "demo" / "run_specs"
    manifest_dir.mkdir(parents=True)

    run_spec = {
        "study_name": "demo_study",
        "run_name": "demo_run",
        "family": "pythia",
        "student_key": "pythia_410m",
        "layer_scheme": "penultimate_only",
        "staged_training_enabled": True,
        "seed": 42,
        "probe_size": 1024,
        "hydra_overrides": ["output_dir=./outputs"],
    }

    run_dir = repo_root / "outputs" / "pipelines" / "demo_run" / "distill_sweep_grid" / "distill_sweep_grid"
    run_dir.mkdir(parents=True)
    (run_dir / "sweep_results.json").write_text(
        json.dumps([{"family": "pythia", "student_model": "student-a", "val_loss": 1.23}]),
        encoding="utf-8",
    )

    rows, missing = _load_result_rows(repo_root=repo_root, run_specs=[run_spec])

    assert not missing
    assert len(rows) == 1
    assert rows[0]["study_run_name"] == "demo_run"
    assert rows[0]["study_staged_training_enabled"] is True
    assert rows[0]["study_expected_run_dir"] == str(repo_root / "outputs" / "pipelines" / "demo_run")
