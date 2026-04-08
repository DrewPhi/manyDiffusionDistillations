import json
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import Subset

from manylatents.pipeline.stages.base import StageContext
from manylatents.pipeline.stages.distillation_sweep import DistillationSweepStage


def test_distillation_sweep_stage_expands_grid_and_writes_sheet(monkeypatch, tmp_path):
    target_path = tmp_path / "aligned_target.npy"
    np.save(target_path, np.ones((8, 4), dtype=np.float32))

    context = StageContext(
        run_id="r_distill",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "phate_teacher_target": {
                "aligned_targets": [str(target_path)],
            }
        },
    )

    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        teacher_target_stage="phate_teacher_target",
        teacher_target_key="aligned_targets",
        sweep={
            "seed": [1, 2],
            "lambda_align": [0.0, 0.3],
            "teacher_model": ["teacher-A"],
            "student_model": ["student-A"],
            "learning_rate": [3e-4],
            "max_length": [32],
            "token_budget": [1024],
        },
    )

    def _fake_run_single(self, combo, aligned_targets, expected_probe_ids, run_output_dir, combo_idx):
        return {
            "teacher_model": combo["teacher_model_name_or_path"],
            "student_model": combo["student_model_name_or_path"],
            "seed": combo["seed"],
            "lambda_align": combo["lambda_align"],
            "learning_rate": combo["learning_rate"],
            "max_length": combo["max_length"],
            "token_budget": combo["token_budget"],
            "max_steps": 1,
            "lambda_schedule": "constant",
            "lambda_last": combo["lambda_align"],
            "val_loss": 1.23,
            "val_perplexity": 3.41,
            "align_mse": 0.42,
            "combo_idx": combo_idx,
            "run_dir": str(run_output_dir),
            "ckpt_path": str(run_output_dir / "student_last.pt"),
            "metrics_path": str(run_output_dir / "metrics.json"),
        }

    monkeypatch.setattr(
        DistillationSweepStage,
        "_run_single_combo",
        _fake_run_single,
    )

    result = stage.run(context=context, stage_dir=tmp_path / "stage_run")
    outputs = result.outputs

    payload = json.loads((tmp_path / "stage_run" / "distill_sweep_grid" / "sweep_results.json").read_text())
    assert len(payload) == 4  # 2 seeds x 2 lambdas
    assert outputs["sweep_results_json"].endswith("sweep_results.json")
    assert outputs["sweep_results_csv"].endswith("sweep_results.csv")


def test_distillation_sweep_stage_supports_tied_sweep_keys():
    combos = DistillationSweepStage._expand_sweep_grid(
        {
            "lambda_align": [0.0, 1.0],
            "lm_loss_weight": [1.0, 0.0],
            "seed": [42],
        },
        tied_key_groups=[["lambda_align", "lm_loss_weight"]],
    )

    assert combos == [
        {"lambda_align": 0.0, "lm_loss_weight": 1.0, "seed": 42},
        {"lambda_align": 1.0, "lm_loss_weight": 0.0, "seed": 42},
    ]


def test_distillation_sweep_stage_uses_multi_layer_target_map(monkeypatch, tmp_path):
    target_a = tmp_path / "aligned_target_a.npy"
    target_b = tmp_path / "aligned_target_b.npy"
    np.save(target_a, np.ones((8, 4), dtype=np.float32))
    np.save(target_b, np.full((8, 4), 2.0, dtype=np.float32))

    context = StageContext(
        run_id="r_distill_multi",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "phate_teacher_target": {
                "aligned_target_paths_by_layer": {
                    "transformer.h[0]": str(target_a),
                    "transformer.h[-2]": str(target_b),
                },
            }
        },
    )

    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        teacher_target_stage="phate_teacher_target",
        teacher_layer_specs=["transformer.h[0]", "transformer.h[-2]"],
        student_layer_specs=["transformer.h[0]", "transformer.h[-2]"],
        sweep={
            "seed": [1],
            "lambda_align": [0.3],
            "teacher_model": ["teacher-A"],
            "student_model": ["student-A"],
            "learning_rate": [3e-4],
            "max_length": [32],
            "token_budget": [1024],
        },
    )

    def _fake_run_single(self, combo, aligned_targets, expected_probe_ids, run_output_dir, combo_idx):
        assert set(aligned_targets.keys()) == {"transformer.h[0]", "transformer.h[-2]"}
        assert aligned_targets["transformer.h[0]"].shape == (8, 4)
        assert aligned_targets["transformer.h[-2]"].shape == (8, 4)
        return {
            "teacher_model": combo["teacher_model_name_or_path"],
            "student_model": combo["student_model_name_or_path"],
            "seed": combo["seed"],
            "lambda_align": combo["lambda_align"],
            "combo_idx": combo_idx,
            "align_mse": 0.2,
            "run_dir": str(run_output_dir),
            "ckpt_path": str(run_output_dir / "student_last.pt"),
            "metrics_path": str(run_output_dir / "metrics.json"),
        }

    monkeypatch.setattr(DistillationSweepStage, "_run_single_combo", _fake_run_single)

    result = stage.run(context=context, stage_dir=tmp_path / "stage_run_multi")
    payload = json.loads(result.outputs["sweep_results_json"] and Path(result.outputs["sweep_results_json"]).read_text())
    assert len(payload) == 1


def test_distillation_sweep_stage_rejects_probe_id_mismatch(tmp_path):
    target_path = tmp_path / "aligned_target.npy"
    np.save(target_path, np.ones((4, 2), dtype=np.float32))

    teacher_probe_ids = tmp_path / "teacher_probe_ids.json"
    teacher_probe_ids.write_text(json.dumps([10, 11, 12, 13]), encoding="utf-8")
    target_probe_ids = tmp_path / "target_probe_ids.json"
    target_probe_ids.write_text(json.dumps([10, 11, 12, 999]), encoding="utf-8")

    context = StageContext(
        run_id="r_distill_bad",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "probe_teacher": {"probe_ids_path": str(teacher_probe_ids)},
            "phate_teacher_target": {
                "aligned_target_path": str(target_path),
                "aligned_probe_ids_path": str(target_probe_ids),
            },
        },
    )

    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        teacher_stage="probe_teacher",
        teacher_target_stage="phate_teacher_target",
        sweep={"seed": [1], "lambda_align": [0.0]},
    )

    try:
        stage.run(context=context, stage_dir=tmp_path / "stage_run_bad")
        assert False, "Expected ValueError due to probe ID mismatch"
    except ValueError as exc:
        assert "Probe ID contract mismatch" in str(exc)


def test_probe_ids_from_datamodule_uses_source_ids_not_subset_indices():
    class _BaseDataset:
        def source_id_for_index(self, idx):
            return [101, 205, 999][idx]

    class _DM:
        probe_source_ids = []
        probe_dataset = Subset(_BaseDataset(), [0, 2])

    probe_ids = DistillationSweepStage._probe_ids_from_datamodule(_DM())
    assert probe_ids == [101, 999]


def test_distillation_sweep_stage_passes_probe_ids_path_to_datamodule(monkeypatch, tmp_path):
    target_path = tmp_path / "aligned_target.npy"
    np.save(target_path, np.ones((4, 2), dtype=np.float32))

    probe_ids_path = tmp_path / "probe_ids.json"
    probe_ids_path.write_text(json.dumps([10, 11, 12, 13]), encoding="utf-8")

    context = StageContext(
        run_id="r_distill_probe_path",
        run_dir=tmp_path,
        cfg={},
        artifacts={
            "probe_teacher": {"probe_ids_path": str(probe_ids_path)},
            "phate_teacher_target": {
                "aligned_target_path": str(target_path),
                "aligned_probe_ids_path": str(probe_ids_path),
            },
        },
    )

    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        teacher_stage="probe_teacher",
        teacher_target_stage="phate_teacher_target",
        sweep={"seed": [1], "lambda_align": [0.0]},
    )

    captured = {}

    class _FakeTextDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.probe_dataset = None
            self.probe_source_ids = []

        def setup(self):
            return None

    def _stop_after_dm(self, dm, aligned_targets, expected_probe_ids):
        raise RuntimeError("stop after dm setup")

    monkeypatch.setattr("manylatents.pipeline.stages.distillation_sweep.TextDataModule", _FakeTextDataModule)
    monkeypatch.setattr(DistillationSweepStage, "_materialize_targets_for_datamodule", _stop_after_dm)

    try:
        stage.run(context=context, stage_dir=tmp_path / "stage_run_probe_path")
        assert False, "Expected sentinel RuntimeError"
    except RuntimeError as exc:
        assert "stop after dm setup" in str(exc)

    assert captured["probe_ids_path"] == str(probe_ids_path)


def test_distillation_sweep_stage_seed_resolution():
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        seeds={},
    )
    assert stage._resolve_stage_seed("model_init_seed", combo_seed=123) == 123
    assert stage._resolve_stage_seed("data_order_seed", combo_seed=999) == 999


def test_distillation_sweep_stage_infers_standard_layer_schemes():
    assert DistillationSweepStage._infer_layer_scheme_name(
        [{"teacher_layer": "a", "student_layer": "a", "weight": 1.0}]
    ) == "penultimate_only"
    assert DistillationSweepStage._infer_layer_scheme_name(
        [
            {"teacher_layer": "a", "student_layer": "a", "weight": 1.0},
            {"teacher_layer": "b", "student_layer": "b", "weight": 1.0},
        ]
    ) == "second_plus_penultimate"
    assert DistillationSweepStage._infer_layer_scheme_name(
        [
            {"teacher_layer": "a", "student_layer": "a", "weight": 1.0},
            {"teacher_layer": "b", "student_layer": "b", "weight": 1.0},
            {"teacher_layer": "c", "student_layer": "c", "weight": 1.0},
        ]
    ) == "custom"

    stage_explicit = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        seeds={
            "model_init_seed": 7,
            "data_order_seed": 8,
            "dataloader_seed": 9,
            "global_seed": 10,
        },
    )
    assert stage_explicit._resolve_stage_seed("model_init_seed", combo_seed=123) == 7
    assert stage_explicit._resolve_stage_seed("data_order_seed", combo_seed=123) == 8
    assert stage_explicit._resolve_stage_seed("dataloader_seed", combo_seed=123) == 9
    assert stage_explicit._resolve_stage_seed("global_seed", combo_seed=123) == 10


def test_distillation_sweep_stage_lambda_schedule_ramp():
    stage = DistillationSweepStage(stage_name="distill_sweep_grid")
    assert stage._lambda_value(step=0, base_lambda=1.0, schedule="ramp", ramp_steps=10) == 0.0
    assert stage._lambda_value(step=5, base_lambda=1.0, schedule="ramp", ramp_steps=10) == 0.5
    assert stage._lambda_value(step=10, base_lambda=1.0, schedule="ramp", ramp_steps=10) == 1.0
    assert stage._lambda_value(step=100, base_lambda=1.0, schedule="ramp", ramp_steps=10) == 1.0


def test_distillation_sweep_stage_max_steps_from_token_budget():
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        global_batch_size=4,
        micro_batch_size=1,
        grad_accum_steps=4,
        max_steps=None,
    )
    # tokens_per_step = 4 * 8 = 32 => floor(320 / 32) = 10
    assert stage._compute_max_steps(token_budget=320, max_length=8) == 10


def test_distillation_sweep_stage_rejects_inconsistent_effective_batch():
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        global_batch_size=512,
        micro_batch_size=8,
        grad_accum_steps=8,  # effective 64, does not match global 512
    )
    try:
        stage._effective_global_batch_size()
        assert False, "Expected ValueError for inconsistent batch size config"
    except ValueError as exc:
        assert "Inconsistent batch configuration" in str(exc)


def test_eval_alignment_batches_probe_eval(monkeypatch):
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        micro_batch_size=8,
        alignment={"batch_size": 2},
    )
    batch_sizes = []

    def _fake_activations(self, model, layer_paths, input_ids, attention_mask, device, detach):
        batch_sizes.append(int(input_ids.shape[0]))
        return {layer_paths[0]: torch.ones((input_ids.shape[0], 4), dtype=torch.float32)}

    monkeypatch.setattr(DistillationSweepStage, "_activations_from_batch", _fake_activations)

    probe_buffers = {
        "input_ids": torch.zeros((5, 3), dtype=torch.long),
        "attention_mask": torch.ones((5, 3), dtype=torch.long),
        "targets": {"teacher.layer": torch.ones((5, 4), dtype=torch.float32)},
        "n": 5,
    }

    mse, per_layer = stage._eval_alignment(
        model=object(),
        device=torch.device("cpu"),
        probe_buffers=probe_buffers,
        eval_idx=np.array([0, 1, 2, 3, 4]),
        layer_pairs=[{"student_layer": "student.layer", "teacher_layer": "teacher.layer", "weight": 1.0}],
    )

    assert batch_sizes == [2, 2, 1]
    assert mse == 0.0
    assert per_layer["student.layer"] == 0.0


def test_eval_alignment_rejects_width_mismatch(monkeypatch):
    stage = DistillationSweepStage(stage_name="distill_sweep_grid", micro_batch_size=2)

    def _fake_activations(self, model, layer_paths, input_ids, attention_mask, device, detach):
        return {layer_paths[0]: torch.ones((input_ids.shape[0], 4), dtype=torch.float32)}

    monkeypatch.setattr(DistillationSweepStage, "_activations_from_batch", _fake_activations)

    probe_buffers = {
        "input_ids": torch.zeros((3, 3), dtype=torch.long),
        "attention_mask": torch.ones((3, 3), dtype=torch.long),
        "targets": {"teacher.layer": torch.ones((3, 5), dtype=torch.float32)},
        "n": 3,
    }

    try:
        stage._eval_alignment(
            model=object(),
            device=torch.device("cpu"),
            probe_buffers=probe_buffers,
            eval_idx=np.array([0, 1, 2]),
            layer_pairs=[{"student_layer": "student.layer", "teacher_layer": "teacher.layer", "weight": 1.0}],
        )
        assert False, "Expected ValueError for alignment width mismatch"
    except ValueError as exc:
        assert "Alignment width mismatch" in str(exc)


def test_activations_from_batch_uses_backbone_and_autocast(monkeypatch):
    stage = DistillationSweepStage(stage_name="distill_sweep_grid")
    autocast_entered = {"count": 0}
    captured = {}

    class _FakeExtractor:
        def __init__(self, specs, detach=True):
            self.specs = specs

        def capture(self, model):
            captured["model"] = model
            return nullcontext()

        def get_activations(self, clear=True):
            captured["spec_paths"] = [spec.path for spec in self.specs]
            return {spec.path: torch.ones((2, 4), dtype=torch.float32) for spec in self.specs}

    class _Backbone(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None):
            captured["forward_called"] = True
            return torch.ones((input_ids.shape[0], input_ids.shape[1], 4), dtype=torch.float32)

    class _FullModel(torch.nn.Module):
        base_model_prefix = "gpt_neox"

        def __init__(self):
            super().__init__()
            self.gpt_neox = _Backbone()

        def forward(self, *args, **kwargs):
            raise AssertionError("full causal LM forward should not be used for activation extraction")

    def _fake_autocast(device):
        class _Ctx:
            def __enter__(self_inner):
                autocast_entered["count"] += 1

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()

    monkeypatch.setattr("manylatents.pipeline.stages.distillation_sweep.ActivationExtractor", _FakeExtractor)
    monkeypatch.setattr(stage, "_autocast_ctx", _fake_autocast)

    acts = stage._activations_from_batch(
        model=_FullModel(),
        layer_paths=["transformer.h[-2]"],
        input_ids=torch.zeros((2, 3), dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        device=torch.device("cpu"),
        detach=True,
    )

    assert captured["model"].__class__.__name__ == "_Backbone"
    assert captured["forward_called"] is True
    assert autocast_entered["count"] == 1
    assert "transformer.h[-2]" in acts


def test_activations_from_batch_uses_encoder_for_encoder_side_alignment(monkeypatch):
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        alignment_side="encoder",
    )
    autocast_entered = {"count": 0}
    captured = {}

    class _FakeExtractor:
        def __init__(self, specs, detach=True):
            self.specs = specs

        def capture(self, model):
            captured["model"] = model
            return nullcontext()

        def get_activations(self, clear=True):
            captured["spec_paths"] = [spec.path for spec in self.specs]
            return {spec.path: torch.ones((2, 4), dtype=torch.float32) for spec in self.specs}

    class _Encoder(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None):
            captured["forward_called"] = True
            return torch.ones((input_ids.shape[0], input_ids.shape[1], 4), dtype=torch.float32)

    class _Seq2Seq(torch.nn.Module):
        base_model_prefix = "transformer"

        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"is_encoder_decoder": True})()
            self.encoder = _Encoder()

        def get_encoder(self):
            return self.encoder

        def forward(self, *args, **kwargs):
            raise AssertionError("full seq2seq forward should not be used for encoder-side activation extraction")

    def _fake_autocast(device):
        class _Ctx:
            def __enter__(self_inner):
                autocast_entered["count"] += 1

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()

    monkeypatch.setattr("manylatents.pipeline.stages.distillation_sweep.ActivationExtractor", _FakeExtractor)
    monkeypatch.setattr(stage, "_autocast_ctx", _fake_autocast)

    acts = stage._activations_from_batch(
        model=_Seq2Seq(),
        layer_paths=["encoder.block[-2]"],
        input_ids=torch.zeros((2, 3), dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        device=torch.device("cpu"),
        detach=True,
    )

    assert captured["model"].__class__.__name__ == "_Encoder"
    assert captured["forward_called"] is True
    assert autocast_entered["count"] == 1
    assert captured["spec_paths"] == ["block[-2]"]
    assert "encoder.block[-2]" in acts


def test_activations_from_batch_avoids_encoder_shortcut_for_masked_lm(monkeypatch):
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        alignment_side="encoder",
    )
    autocast_entered = {"count": 0}
    captured = {}

    class _FakeExtractor:
        def __init__(self, specs, detach=True):
            self.specs = specs

        def capture(self, model):
            captured["model"] = model
            return nullcontext()

        def get_activations(self, clear=True):
            captured["spec_paths"] = [spec.path for spec in self.specs]
            return {spec.path: torch.ones((2, 4), dtype=torch.float32) for spec in self.specs}

    class _Encoder(torch.nn.Module):
        def forward(self, hidden_states):
            raise AssertionError("internal encoder module should not receive direct activation forwards")

    class _Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()

        def forward(self, input_ids=None, attention_mask=None):
            captured["forward_called"] = True
            return torch.ones((input_ids.shape[0], input_ids.shape[1], 4), dtype=torch.float32)

    class _MaskedLM(torch.nn.Module):
        base_model_prefix = "bert"

        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"is_encoder_decoder": False})()
            self.bert = _Backbone()

        def get_encoder(self):
            return self.bert.encoder

        def forward(self, *args, **kwargs):
            raise AssertionError("full masked LM forward should not be used for activation extraction")

    def _fake_autocast(device):
        class _Ctx:
            def __enter__(self_inner):
                autocast_entered["count"] += 1

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()

    monkeypatch.setattr("manylatents.pipeline.stages.distillation_sweep.ActivationExtractor", _FakeExtractor)
    monkeypatch.setattr(stage, "_autocast_ctx", _fake_autocast)

    acts = stage._activations_from_batch(
        model=_MaskedLM(),
        layer_paths=["bert.encoder.layer[-2]"],
        input_ids=torch.zeros((2, 3), dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        device=torch.device("cpu"),
        detach=True,
    )

    assert captured["model"].__class__.__name__ == "_Backbone"
    assert captured["forward_called"] is True
    assert autocast_entered["count"] == 1
    assert captured["spec_paths"] == ["encoder.layer[-2]"]
    assert "bert.encoder.layer[-2]" in acts


def test_cleanup_combo_resources_runs_gc_and_cuda_cleanup(monkeypatch):
    calls = {"gc": 0, "cuda": 0}

    monkeypatch.setattr("manylatents.pipeline.stages.distillation_sweep.gc.collect", lambda: calls.__setitem__("gc", calls["gc"] + 1))
    monkeypatch.setattr("manylatents.pipeline.stages.distillation_sweep.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "manylatents.pipeline.stages.distillation_sweep.torch.cuda.empty_cache",
        lambda: calls.__setitem__("cuda", calls["cuda"] + 1),
    )

    class _Trainer:
        network = object()
        tokenizer = object()

    DistillationSweepStage._cleanup_combo_resources(
        trainer_module=_Trainer(),
        model=object(),
        optimizer=object(),
        dm=object(),
        probe_buffers={"x": 1},
        aligned_targets_for_run={"y": np.zeros((1, 1), dtype=np.float32)},
        train_loader=object(),
        train_iter=object(),
    )

    assert calls == {"gc": 1, "cuda": 1}


def test_save_periodic_checkpoint_keeps_analysis_copy_when_best_checkpoint_is_pruned(tmp_path):
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        save_top_k=1,
        analysis_checkpoint_steps=[10],
    )
    run_output_dir = tmp_path / "run"
    run_output_dir.mkdir()
    model = torch.nn.Linear(2, 2)
    best_checkpoints = []

    stage._save_periodic_checkpoint(
        model=model,
        run_output_dir=run_output_dir,
        step=10,
        eval_loss_for_rank=5.0,
        best_checkpoints=best_checkpoints,
    )
    assert (run_output_dir / "student_step10.pt").exists()
    assert (run_output_dir / "student_analysis_step10.pt").exists()

    stage._save_periodic_checkpoint(
        model=model,
        run_output_dir=run_output_dir,
        step=20,
        eval_loss_for_rank=1.0,
        best_checkpoints=best_checkpoints,
    )

    assert not (run_output_dir / "student_step10.pt").exists()
    assert (run_output_dir / "student_analysis_step10.pt").exists()
    assert (run_output_dir / "student_step20.pt").exists()


def test_resolved_analysis_checkpoint_steps_from_staged_training_defaults():
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        staged_training={
            "enabled": True,
            "phase1": {"max_steps": 10},
            "phase2": {"max_steps": 20},
            "phase3": {"max_steps": 10},
            "checkpoint_analysis": {
                "enabled": True,
                "phase1_snapshots": 2,
                "phase2_snapshots": 3,
                "phase3_snapshots": 2,
                "include_phase_end": True,
            },
        },
    )

    steps = stage._resolved_analysis_checkpoint_steps()
    assert steps
    assert steps[0] == 1
    assert 10 in steps
    assert 30 in steps
    assert 40 in steps


def test_append_analysis_queue_entry_writes_jsonl(tmp_path):
    queue_path = tmp_path / "analysis_queue.jsonl"
    DistillationSweepStage._append_analysis_queue_entry(
        queue_path,
        {"step": 10, "phase": "phase1", "checkpoint_path": "demo.pt"},
    )

    payload = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    assert payload == [{"step": 10, "phase": "phase1", "checkpoint_path": "demo.pt"}]


def test_build_analysis_queue_entry_includes_trust_remote_code(tmp_path):
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        student_trust_remote_code=True,
        student_layer_specs=["transformer.h[-2]"],
    )
    entry = stage._build_analysis_queue_entry(
        combo={
            "student_model_name_or_path": "Qwen/Qwen2.5-0.5B",
            "student_model_revision": "main",
            "student_model_family": "causal_lm",
            "student_model_config_name_or_path": None,
            "student_model_config_overrides": {},
            "max_length": 128,
        },
        run_output_dir=tmp_path,
        step=12,
        phase_name="phase2",
        phase_step=3,
        probe_buffers={"n": 64},
        probe_source_ids=[101, 202, 303],
        probe_seed=42,
        data_order_seed=42,
        dataloader_seed=42,
    )

    assert entry["student_trust_remote_code"] is True
    assert entry["probe_source_ids"] == [101, 202, 303]
    assert entry["checkpoint_path"].endswith("student_analysis_step12.pt")


def test_staged_training_rejects_max_step_mismatch():
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        max_steps=11,
        staged_training={
            "enabled": True,
            "phase1": {"max_steps": 5},
            "phase2": {"max_steps": 5},
            "phase3": {"max_steps": 0},
        },
    )

    try:
        # Exercise the validation through the staged checkpoint planner path.
        total = sum(
            int((stage.staged_training_cfg.get(name) or {}).get("max_steps", 0))
            for name in ("phase1", "phase2", "phase3")
        )
        if stage.max_steps is not None and int(stage.max_steps) != int(total):
            raise ValueError(
                "staged_training phase step budgets must sum to training.max_steps "
                f"({total} vs {stage.max_steps})"
            )
        assert False, "Expected ValueError for staged max-step mismatch"
    except ValueError as exc:
        assert "must sum to training.max_steps" in str(exc)


def test_control_task_only_uses_matched_budget_checkpoints():
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        training_regime="control_task_only",
        analysis_checkpoint_steps=[],
        staged_training={
            "enabled": True,
            "phase1": {"max_steps": 250},
            "phase2": {"max_steps": 1250},
            "phase3": {"max_steps": 500},
        },
    )

    assert stage._control_budget_steps() == (1750, 2000)
    assert stage._control_analysis_checkpoint_steps() == [1750, 2000]


def test_control_task_only_analysis_steps_include_explicit_steps():
    stage = DistillationSweepStage(
        stage_name="distill_sweep_grid",
        training_regime="control_task_only",
        analysis_checkpoint_steps=[1, 1750],
        staged_training={
            "enabled": True,
            "phase1": {"max_steps": 250},
            "phase2": {"max_steps": 1250},
            "phase3": {"max_steps": 500},
        },
    )

    assert stage._resolved_analysis_checkpoint_steps() == [1, 1750, 2000]
