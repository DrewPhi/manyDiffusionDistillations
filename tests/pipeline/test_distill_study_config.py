from pathlib import Path

from omegaconf import OmegaConf

from downstream.distill_family_study.scripts.materialize_distill_study import (
    expected_run_count,
    materialize_study,
)
from manylatents.pipeline.family_resolution import get_family_layer_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "downstream" / "distill_family_study" / "configs"


def _load_yaml(*parts: str):
    return OmegaConf.load(CONFIG_ROOT.joinpath(*parts))


def test_family_registry_has_publication_families():
    for family_name in ("pythia", "qwen", "bert", "deberta_v3"):
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


def test_encoder_only_family_registry_supports_bert_and_deberta_v3():
    for family_name in ("bert", "deberta_v3"):
        cfg = _load_yaml("family", f"{family_name}.yaml")
        resolved = get_family_layer_spec(family_name)
        assert cfg.name == family_name
        assert cfg.model_family == "masked_lm"
        assert cfg.architecture == "encoder_only"
        assert resolved is not None
        assert resolved.second_layer == cfg.layer_paths.second
        assert resolved.penultimate_layer == cfg.layer_paths.penultimate
        assert cfg.custom_students
        assert all("model_config_overrides" in student for student in cfg.custom_students)


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


def test_publication_study_materializes_expected_24_runs():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    runs = materialize_study(cfg)

    assert len(runs) == 48
    run_names = [run.run_name for run in runs]
    assert len(run_names) == len(set(run_names))


def test_publication_study_has_expected_family_student_counts():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    runs = materialize_study(cfg)

    family_counts = {}
    for run in runs:
        family_counts[run.family] = family_counts.get(run.family, 0) + 1

    assert family_counts == {"pythia": 12, "qwen": 12, "bert": 12, "deberta_v3": 12}


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
    assert repro["training_regime"] in {"staged", "control_task_only"}
    assert repro["staged_training_enabled"] == (run.training_regime == "staged")


def test_materialized_run_carries_custom_student_architecture_overrides():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    cfg.study.family_order = ["pythia"]
    cfg.study.layer_schemes = ["penultimate_only"]
    cfg.study.families.pythia.students = [
        {
            "key": "pythia_custom_probe",
            "model_name_or_path": "EleutherAI/pythia-410m",
            "model_config_name_or_path": "EleutherAI/pythia-410m",
            "model_revision": "main",
            "trust_remote_code": False,
            "init_from_scratch": True,
            "penultimate_dim": 1024,
            "tokenizer_name": "EleutherAI/pythia-410m",
            "model_config_overrides": {
                "hidden_size": 1024,
                "num_hidden_layers": 10,
            },
        }
    ]

    run = materialize_study(cfg)[0]
    override_blob = " ".join(run.hydra_overrides)

    assert run.student_model_config_name_or_path == "EleutherAI/pythia-410m"
    assert run.student_model_config_overrides == {"hidden_size": 1024, "num_hidden_layers": 10}
    assert "stage_pipeline.params.student.model_config_name_or_path=EleutherAI/pythia-410m" in override_blob
    assert "stage_pipeline.params.student.model_config_overrides.hidden_size=1024" in override_blob
    assert "stage_pipeline.params.student.model_config_overrides.num_hidden_layers=10" in override_blob

def test_materialized_run_carries_staged_training_overrides():
    cfg = _load_yaml("study", "within_family_publication.yaml")
    runs = materialize_study(cfg)
    run = next(run for run in runs if run.training_regime == "staged")

    override_blob = " ".join(run.hydra_overrides)
    assert run.staged_training_enabled is True
    assert run.training_regime == "staged"
    assert "stage_pipeline.params.training.training_regime=staged" in override_blob
    assert "stage_pipeline.params.training.staged_training.enabled=true" in override_blob
    assert "stage_pipeline.params.training.staged_training.phase1.objective=alignment_only" in override_blob
    assert "stage_pipeline.params.training.staged_training.phase2.objective=task_only_frozen" in override_blob
    assert "stage_pipeline.params.training.staged_training.phase3.objective=task_only_unfrozen" in override_blob
    assert run.reproducibility["training"]["staged_training"]["checkpoint_analysis"]["phase2_snapshots"] == 10


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


def test_staged_smoke_a100_materializes_single_run():
    cfg = _load_yaml("study", "staged_smoke_a100_1gpu.yaml")
    runs = materialize_study(cfg)

    assert len(runs) == 2
    regimes = {run.training_regime for run in runs}
    assert regimes == {"staged", "control_task_only"}
    assert all(run.family == "bert" for run in runs)
    assert all(run.student_key == "bert_11m" for run in runs)
    assert all(run.layer_scheme == "penultimate_only" for run in runs)
    staged = next(run for run in runs if run.training_regime == "staged")
    control = next(run for run in runs if run.training_regime == "control_task_only")
    assert staged.staged_training_enabled is True
    assert control.staged_training_enabled is False
    assert staged.reproducibility["training"]["staged_training"]["phase2"]["max_steps"] == 1250
    assert control.reproducibility["training"]["staged_training"]["phase2"]["max_steps"] == 1250
    assert "stage_pipeline.params.training.training_regime=control_task_only" in " ".join(control.hydra_overrides)


def test_expected_run_count_includes_training_regimes():
    smoke_cfg = _load_yaml("study", "staged_smoke_a100_1gpu.yaml")
    remaining_cfg = _load_yaml("study", "staged_smoke_remaining_families_a100_1gpu.yaml")
    full_cfg = _load_yaml("study", "within_family_full_pile.yaml")

    assert expected_run_count(smoke_cfg) == 2
    assert expected_run_count(remaining_cfg) == 6
    assert expected_run_count(full_cfg) == 48


def test_remaining_families_smoke_materializes_expected_runs():
    cfg = _load_yaml("study", "staged_smoke_remaining_families_a100_1gpu.yaml")
    runs = materialize_study(cfg)

    assert len(runs) == 6
    assert cfg.study.family_order == ["pythia", "deberta_v3", "qwen"]
    assert {run.training_regime for run in runs} == {"staged", "control_task_only"}
    assert {run.layer_scheme for run in runs} == {"penultimate_only"}

    students_by_family = {}
    for run in runs:
        students_by_family.setdefault(run.family, set()).add(run.student_key)

    assert students_by_family == {
        "pythia": {"pythia_410m"},
        "deberta_v3": {"deberta_v3_xsmall"},
        "qwen": {"qwen2_5_0_5b"},
    }
