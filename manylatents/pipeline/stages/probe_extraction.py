"""Probe extraction stage for representation experiments."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from manylatents.callbacks.diffusion_operator import build_diffusion_operator
from manylatents.experiment import instantiate_algorithm, instantiate_datamodule, run_algorithm
from manylatents.lightning.hooks import ActivationExtractor, LayerSpec
from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult

logger = logging.getLogger(__name__)


class ProbeExtractionStage(PipelineStage):
    """Run probe extraction and persist diffusion artifacts.

    This stage delegates model/data/training logic to existing manylatents configs,
    then computes diffusion operators from saved raw activations when available.
    """

    def __init__(
        self,
        stage_name: str,
        config_overrides: Dict[str, Any] | None = None,
        output_subdir: str | None = None,
        probe_callback_path: str = "callbacks.trainer.probe",
        diffusion: Dict[str, Any] | None = None,
        source_file_index: int = -1,
        forward_only: bool = True,
        external_manifest_path: str | None = None,
    ):
        super().__init__(stage_name=stage_name)
        self.config_overrides = config_overrides or {}
        self.output_subdir = output_subdir or stage_name
        self.probe_callback_path = probe_callback_path
        self.diffusion = diffusion or {
            "knn": 35,
            "alpha": 1.0,
            "symmetric": False,
            "metric": "euclidean",
        }
        self.source_file_index = source_file_index
        self.forward_only = forward_only
        self.external_manifest_path = external_manifest_path

    def _load_external_outputs(self) -> StageResult:
        if not self.external_manifest_path:
            raise ValueError("external_manifest_path is not configured")
        manifest_path = Path(self.external_manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = payload.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError(f"Manifest at {manifest_path} does not contain stage outputs")
        metadata = payload.get("metadata") or {}
        metadata = {
            **metadata,
            "imported_from_manifest": str(manifest_path),
            "import_mode": True,
        }
        logger.info("ProbeExtractionStage '%s' importing outputs from %s", self.stage_name, manifest_path)
        return StageResult(outputs=outputs, metadata=metadata)

    def _build_run_cfg(self, cfg: Any, stage_dir: Path) -> Any:
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        OmegaConf.set_struct(run_cfg, False)

        # Ensure probe callback writes raw activations to the stage directory.
        raw_dir = stage_dir / "raw_activations"
        callback_cfg = OmegaConf.select(run_cfg, self.probe_callback_path)
        if callback_cfg is not None:
            OmegaConf.update(run_cfg, f"{self.probe_callback_path}.save_raw", True, merge=False)
            OmegaConf.update(run_cfg, f"{self.probe_callback_path}.save_path", str(raw_dir), merge=False)

        if self.config_overrides:
            run_cfg = OmegaConf.merge(run_cfg, OmegaConf.create(self.config_overrides))

        return run_cfg

    def _resolve_layer_specs(self, run_cfg: Any) -> List[LayerSpec]:
        layer_specs_cfg = OmegaConf.select(run_cfg, f"{self.probe_callback_path}.layer_specs")
        if layer_specs_cfg is None:
            raise KeyError(f"Missing '{self.probe_callback_path}.layer_specs' for forward-only probe extraction")

        layer_specs: List[LayerSpec] = []
        for spec_cfg in layer_specs_cfg:
            spec = hydra.utils.instantiate(spec_cfg)
            if isinstance(spec, LayerSpec):
                layer_specs.append(spec)
                continue

            # Hydra may return plain DictConfig/dict when _target_ metadata was
            # resolved away upstream. Support that shape directly.
            if isinstance(spec, (DictConfig, dict)):
                path = spec.get("path")
                if path is None:
                    raise TypeError(f"Layer spec mapping missing required 'path': {spec}")
                spec = LayerSpec(
                    path=str(path),
                    extraction_point=str(spec.get("extraction_point", "output")),
                    reduce=str(spec.get("reduce", "mean")),
                )
                layer_specs.append(spec)
                continue

            raise TypeError(f"Expected LayerSpec from callback layer_specs, got {type(spec)}")

        if not layer_specs:
            raise ValueError("No layer_specs configured for probe extraction")
        return layer_specs

    @staticmethod
    def _safe_layer_name(layer: str) -> str:
        return layer.replace(".", "_").replace("[", "_").replace("]", "_").replace("-", "m")

    @staticmethod
    def _select_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _forward_inputs_from_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        keys = ("input_ids", "attention_mask", "labels")
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            v = batch.get(k)
            if v is not None:
                out[k] = v.to(device)
        if not out:
            raise KeyError("Probe batch missing required keys (input_ids/attention_mask)")
        return out

    @staticmethod
    def _precision_to_dtype(precision: Any) -> torch.dtype | None:
        p = str(precision).lower()
        if p in {"bf16", "bf16-mixed", "bfloat16"}:
            return torch.bfloat16
        if p in {"16", "16-mixed", "fp16", "float16"}:
            return torch.float16
        return None

    @staticmethod
    def _select_forward_module(
        network: torch.nn.Module,
        layer_specs: List[LayerSpec],
    ) -> tuple[torch.nn.Module, List[LayerSpec]]:
        layer_paths = [spec.path for spec in layer_specs]
        if layer_paths and all(path.startswith("encoder.") for path in layer_paths):
            encoder = getattr(network, "encoder", None)
            if isinstance(encoder, torch.nn.Module):
                normalized = [
                    LayerSpec(
                        path=spec.path[len("encoder."):],
                        extraction_point=spec.extraction_point,
                        reduce=spec.reduce,
                    )
                    for spec in layer_specs
                ]
                return encoder, normalized
        return network, layer_specs

    def _run_forward_only_probe(self, run_cfg: Any, stage_output_dir: Path) -> Dict[str, Any]:
        logger.info("Starting forward-only probe extraction")
        dm = instantiate_datamodule(run_cfg)
        try:
            dm.setup(stage="probe")
        except TypeError:
            dm.setup()
        logger.info("Datamodule probe setup complete")
        probe_loader = dm.probe_dataloader() if hasattr(dm, "probe_dataloader") else dm.val_dataloader()
        logger.info("Probe dataloader ready with %d batches", len(probe_loader))

        algo_cfg = OmegaConf.select(run_cfg, "algorithms.lightning")
        if algo_cfg is None:
            raise ValueError("Forward-only probe extraction currently requires algorithms.lightning config")
        algorithm = instantiate_algorithm(algo_cfg, datamodule=dm)
        target_dtype = self._precision_to_dtype(OmegaConf.select(run_cfg, "trainer.precision"))
        if target_dtype is not None and hasattr(algorithm, "config"):
            algo_config = getattr(algorithm, "config", None)
            if algo_config is not None and getattr(algo_config, "torch_dtype", None) is None:
                setattr(algo_config, "torch_dtype", target_dtype)

        if hasattr(algorithm, "configure_model"):
            logger.info("Configuring teacher model for probe extraction")
            algorithm.configure_model()
        network = getattr(algorithm, "network", None) or algorithm
        device = self._select_device()
        network = network.to(device)
        network.eval()
        logger.info("Teacher model moved to %s and set to eval mode", device)

        layer_specs = self._resolve_layer_specs(run_cfg)
        forward_network, forward_layer_specs = self._select_forward_module(network, layer_specs)
        logger.info("Capturing activations for %d layer spec(s)", len(layer_specs))
        extractor = ActivationExtractor(forward_layer_specs)
        with torch.no_grad():
            with extractor.capture(forward_network):
                batch_count = 0
                for batch in probe_loader:
                    inputs = self._forward_inputs_from_batch(batch, device)
                    if forward_network is not network:
                        inputs.pop("labels", None)
                    forward_network(**inputs)
                    batch_count += 1
        logger.info("Completed teacher forward pass over %d probe batches", batch_count)
        acts = extractor.get_activations()

        raw_dir = stage_output_dir / "raw_activations"
        raw_dir.mkdir(parents=True, exist_ok=True)

        first_path: Path | None = None
        for original_spec, forward_spec in zip(layer_specs, forward_layer_specs):
            layer_acts = acts.get(forward_spec.path)
            if layer_acts is None:
                raise KeyError(f"Could not extract activations for layer '{original_spec.path}'")
            safe_name = self._safe_layer_name(original_spec.path)
            raw_path = raw_dir / f"{safe_name}_step0.npy"
            np.save(raw_path, layer_acts.detach().float().cpu().numpy())
            if first_path is None:
                first_path = raw_path

        if first_path is None:
            raise RuntimeError("Forward-only probe extraction produced no activation files")
        embeddings = np.load(first_path)
        logger.info("Forward-only probe extraction wrote raw activations to %s", raw_dir)
        return {
            "embeddings": embeddings,
            "scores": {
                "forward_only": True,
                "num_probe_batches": int(len(probe_loader)),
                "num_layers": int(len(layer_specs)),
            },
            "datamodule": dm,
        }

    def _save_scores(self, scores: Dict[str, Any], output_path: Path) -> Path:
        def _to_jsonable(val: Any) -> Any:
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (np.floating, np.integer)):
                return val.item()
            if isinstance(val, tuple):
                return [_to_jsonable(v) for v in val]
            if isinstance(val, dict):
                return {str(k): _to_jsonable(v) for k, v in val.items()}
            return val

        serializable = {str(k): _to_jsonable(v) for k, v in (scores or {}).items()}
        output_path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
        return output_path

    def _build_probe_ids(self, run_cfg: Any, n_expected: int, datamodule: Any | None = None) -> list[int]:
        # Contract source of truth: source IDs from the datamodule probe subset when available.
        if datamodule is not None:
            probe_source_ids = getattr(datamodule, "probe_source_ids", None)
            if probe_source_ids:
                return [int(i) for i in probe_source_ids[:n_expected]]
            probe_dataset = getattr(datamodule, "probe_dataset", None)
            if probe_dataset is not None and hasattr(probe_dataset, "indices"):
                base_dataset = getattr(probe_dataset, "dataset", None)
                if base_dataset is not None and hasattr(base_dataset, "source_id_for_index"):
                    probe_ids = [
                        int(base_dataset.source_id_for_index(int(i)))
                        for i in getattr(probe_dataset, "indices")
                    ]
                else:
                    probe_ids = [int(i) for i in getattr(probe_dataset, "indices")]
                if probe_ids:
                    return probe_ids[:n_expected]

        try:
            dm = instantiate_datamodule(run_cfg)
            try:
                dm.setup(stage="probe")
            except TypeError:
                dm.setup()
            probe_source_ids = getattr(dm, "probe_source_ids", None)
            if probe_source_ids:
                return [int(i) for i in probe_source_ids[:n_expected]]
            probe_dataset = getattr(dm, "probe_dataset", None)
            if probe_dataset is not None and hasattr(probe_dataset, "indices"):
                base_dataset = getattr(probe_dataset, "dataset", None)
                if base_dataset is not None and hasattr(base_dataset, "source_id_for_index"):
                    probe_ids = [
                        int(base_dataset.source_id_for_index(int(i)))
                        for i in getattr(probe_dataset, "indices")
                    ]
                else:
                    probe_ids = [int(i) for i in getattr(probe_dataset, "indices")]
                if probe_ids:
                    return probe_ids[:n_expected]
        except Exception:
            # Fallback to positional IDs to keep pipeline robust.
            pass

        return list(range(int(n_expected)))

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        if self.external_manifest_path:
            return self._load_external_outputs()

        stage_output_dir = stage_dir / self.output_subdir
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ProbeExtractionStage '%s' writing to %s", self.stage_name, stage_output_dir)
        run_cfg = self._build_run_cfg(context.cfg, stage_output_dir)

        probe_dm = None
        result: Dict[str, Any]
        if self.forward_only:
            try:
                result = self._run_forward_only_probe(run_cfg, stage_output_dir)
                probe_dm = result.get("datamodule")
            except Exception as exc:
                logger.warning(
                    "Forward-only probe extraction failed; falling back to run_algorithm path. Error: %s",
                    exc,
                )
                result = run_algorithm(run_cfg)
        else:
            result = run_algorithm(run_cfg)
        scores_path = self._save_scores(result.get("scores", {}), stage_output_dir / "scores.json")

        diffusion_dir = stage_output_dir / "diff_ops"
        diffusion_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = stage_output_dir / "raw_activations"
        raw_activation_files = sorted(raw_dir.glob("*.npy")) if raw_dir.exists() else []

        embeddings = result.get("embeddings")
        if embeddings is not None:
            if torch.is_tensor(embeddings):
                embeddings = embeddings.detach().cpu().numpy()
            else:
                embeddings = np.asarray(embeddings)
        elif raw_activation_files:
            # Lightning/HF probe-only runs may not produce embeddings; use raw activations.
            first_raw = np.load(raw_activation_files[self.source_file_index])
            embeddings = np.asarray(first_raw)
        else:
            raise RuntimeError(
                "ProbeExtractionStage expected either run_algorithm()['embeddings'] "
                "or saved raw activations, but found neither."
            )

        embeddings_path = stage_output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)

        diffusion_paths = []
        if raw_activation_files:
            logger.info("Building %d diffusion operator(s) from raw activations", len(raw_activation_files))
            for raw_file in raw_activation_files:
                activations = np.load(raw_file)
                diff_op = build_diffusion_operator(activations, method="diffusion", **self.diffusion)
                diff_path = diffusion_dir / f"{raw_file.stem}__diffop.npy"
                np.save(diff_path, diff_op)
                diffusion_paths.append(str(diff_path))
        else:
            fallback_diff = build_diffusion_operator(embeddings, method="diffusion", **self.diffusion)
            fallback_path = diffusion_dir / "embeddings__diffop.npy"
            np.save(fallback_path, fallback_diff)
            diffusion_paths.append(str(fallback_path))

        probe_ids = self._build_probe_ids(run_cfg, n_expected=int(embeddings.shape[0]), datamodule=probe_dm)
        probe_ids_path = stage_output_dir / "probe_ids.json"
        probe_ids_path.write_text(json.dumps(probe_ids, indent=2), encoding="utf-8")
        logger.info("Wrote %d probe IDs to %s", len(probe_ids), probe_ids_path)

        # Contract primary pointers (single canonical file for downstream consumption).
        if raw_activation_files:
            act_idx = min(max(self.source_file_index, -len(raw_activation_files)), len(raw_activation_files) - 1)
            teacher_activations_path = str(raw_activation_files[act_idx])
        else:
            teacher_activations_path = str(embeddings_path)

        if diffusion_paths:
            diff_idx = min(max(self.source_file_index, -len(diffusion_paths)), len(diffusion_paths) - 1)
            teacher_diffop_path = diffusion_paths[diff_idx]
        else:
            teacher_diffop_path = ""

        return StageResult(
            outputs={
                "stage_output_dir": str(stage_output_dir),
                "embeddings": str(embeddings_path),
                "scores": str(scores_path),
                "diffusion_operators": diffusion_paths,
                "raw_activations_dir": str(raw_dir),
                # Contract keys
                "probe_ids_path": str(probe_ids_path),
                "teacher_activations_path": teacher_activations_path,
                "teacher_activations_paths": [str(p) for p in raw_activation_files] if raw_activation_files else [str(embeddings_path)],
                "teacher_diffop_path": teacher_diffop_path,
                "teacher_diffop_paths": diffusion_paths,
            },
            metadata={
                "num_raw_activation_files": len(raw_activation_files),
                "num_diffusion_operators": len(diffusion_paths),
                "embedding_shape": list(embeddings.shape),
                "probe_size": int(len(probe_ids)),
                "artifact_contract": "probe_target_v1",
            },
        )
