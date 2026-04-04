from pathlib import Path

from omegaconf import OmegaConf

from downstream.distill_family_study.scripts.materialize_distill_study import materialize_study
from manylatents.pipeline.family_resolution import get_family_layer_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "downstream" / "distill_family_study" / "configs"


def _load_yaml(*parts: str):
    return OmegaConf.load(CONFIG_ROOT.joinpath(*parts))


def test_family_registry_has_publication_families():
    for family_name in ("pythia", "qwen", "t5"):
        cfg = _load_yaml("family", f"{family_name}.yaml")
        resolved = get_family_layer_spec(family_name)
        assert cfg.name == family_name
        assert resolved is not None
        assert resolved.family_name == family_name
        assert resolved.second_layer == cfg.layer_paths.second
        assert resolved.penultimate_layer == cfg.layer_paths.penultimate
        assert resolved.architecture == cfg.architecture
        assert resolved.alignment_side == cfg.alignment_side
        assert cfg.layer_paths.second
        assert cfg.layer_paths.penultimate
        assert len(cfg.candidate_students) >= 3


def test_layer_scheme_registry_expands_expected_layer_counts():
    penultimate = _load_yaml("layer_scheme", "penultimate_only.yaml")
    second_plus = _load_yaml("layer_scheme", "second_plus_penultimate.yaml")
    penultimate_plain = OmegaConf.to_container(penultimate, resolve=False)
    second_plus_plain = OmegaConf.to_container(second_plus, resolve=False)

    assert penultimate_plain["teacher_layer_specs"] == ["${family.layer_paths.penultimate}"]
    assert penultimate_plain["student_layer_specs"] == ["${family.layer_paths.penultimate}"]
    assert penultimate_plain["layer_loss_weights"] == [1.0]

    assert second_plus_plain["teacher_layer_specs"] == [
        "${family.layer_paths.second}",
        "${family.layer_paths.penultimate}",
    ]
    assert second_plus_plain["student_layer_specs"] == [
        "${family.layer_paths.second}",
        "${family.layer_paths.penultimate}",
    ]
    assert second_plus_plain["layer_loss_weights"] == [1.0, 1.0]


def test_publication_study_materializes_expected_54_runs():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    runs = materialize_study(cfg)

    assert len(runs) == 54
    run_names = [run.run_name for run in runs]
    assert len(run_names) == len(set(run_names))


def test_publication_study_has_expected_family_student_counts():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    runs = materialize_study(cfg)

    family_counts = {}
    for run in runs:
        family_counts[run.family] = family_counts.get(run.family, 0) + 1

    assert family_counts == {"pythia": 18, "qwen": 18, "t5": 18}


def test_publication_study_probe_size_tracks_student_penultimate_dim():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    runs = materialize_study(cfg)

    for run in runs:
        assert run.probe_size == 2 * run.student_penultimate_dim


def test_publication_study_can_use_fixed_probe_size():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    cfg.study.shared.probe.size = 4096
    runs = materialize_study(cfg)

    assert runs
    assert all(run.probe_size == 4096 for run in runs)
    assert all(run.reproducibility["probe"]["size"] == 4096 for run in runs)


def test_phate_target_reuse_boundary_matches_student_dimension():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    runs = materialize_study(cfg)

    dims_by_student_key = {}
    for run in runs:
        dims_by_student_key.setdefault(run.student_key, set()).add(run.student_penultimate_dim)

    assert all(len(dims) == 1 for dims in dims_by_student_key.values())
    assert {run.student_penultimate_dim for run in runs if run.family == "pythia"} == {1024, 2048, 2560}


def test_materialized_run_overrides_match_run_identity():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    run = materialize_study(cfg)[0]

    override_blob = " ".join(run.hydra_overrides)
    assert f"name={run.run_name}" in override_blob
    assert f"family={run.family_config}" in override_blob
    assert f"layer_scheme={run.layer_scheme}" in override_blob
    assert f"stage_pipeline.params.probe.size={run.probe_size}" in override_blob
    assert f"stage_pipeline.params.student.penultimate_dim={run.student_penultimate_dim}" in override_blob


def test_materialized_run_freezes_reproducibility_metadata():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    run = materialize_study(cfg)[0]

    repro = run.reproducibility
    assert repro["experiment"] == "distill_family_study_template"
    assert repro["teacher"]["model_revision"] == run.teacher_revision
    assert repro["student"]["model_revision"] == run.student_revision
    assert repro["student"]["penultimate_dim"] == run.student_penultimate_dim
    assert repro["probe"]["size"] == run.probe_size
    assert repro["data"]["token_budget"] == run.token_budget


def test_publication_study_uses_ramp_schedule_for_lambda_1():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    runs = materialize_study(cfg)

    lambda_one_runs = [run for run in runs if run.lambda_align == 1.0]
    assert lambda_one_runs
    assert all(run.reproducibility["alignment"]["lambda_schedule"] == "ramp" for run in lambda_one_runs)
    assert all(run.reproducibility["alignment"]["lambda_ramp_fraction"] == 0.2 for run in lambda_one_runs)
    assert all(run.reproducibility["alignment"]["lambda_ramp_steps"] == 1907 for run in lambda_one_runs)

    lambda_half_runs = [run for run in runs if run.lambda_align == 0.5]
    assert lambda_half_runs
    assert all(run.reproducibility["alignment"]["lambda_schedule"] == "constant" for run in lambda_half_runs)


def test_full_pile_study_includes_fixed_analysis_checkpoints():
    cfg = _load_yaml("study", "within_family_full_pile.yaml")
    runs = materialize_study(cfg)

    assert runs
    expected_steps = [34571, 86428, 172856, 259283]
    assert cfg.study.shared.training.analysis_checkpoint_steps == expected_steps
    assert all(
        "stage_pipeline.params.training.analysis_checkpoint_steps=[34571,86428,172856,259283]"
        in " ".join(run.hydra_overrides)
        for run in runs
    )
    assert all(run.reproducibility["training"]["analysis_checkpoint_steps"] == expected_steps for run in runs)
