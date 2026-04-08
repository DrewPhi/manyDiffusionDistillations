"""PHATE + Procrustes aligned teacher target construction stage."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from manylatents.algorithms.latent.phate import PHATEModule
from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult

try:
    import wandb
    wandb.init
except (ImportError, AttributeError):
    wandb = None


class PHATEAlignedTargetStage(PipelineStage):
    """Build fixed teacher targets by PHATE embedding + Procrustes alignment."""

    def __init__(
        self,
        stage_name: str,
        source_stage: str = "probe_teacher",
        diffusion_key: str = "diffusion_operators",
        teacher_diffop_key: str = "teacher_diffop_path",
        probe_ids_key: str = "probe_ids_path",
        teacher_activations_key: str = "teacher_activations_path",
        raw_activations_key: str = "raw_activations_dir",
        raw_activation_glob: str = "*.npy",
        source_file_index: int = -1,
        output_subdir: str | None = None,
        output_suffix: str = "__teacher_aligned_target.npy",
        n_components: int = 2,
        random_state: int = 42,
        knn: int = 5,
        t: int | str = 15,
        decay: int = 40,
        gamma: float = 1.0,
        n_pca: int | None = None,
        n_landmark: int | None = None,
        n_jobs: int = -1,
        verbose: bool = False,
        fit_fraction: float = 1.0,
        external_manifest_path: str | None = None,
    ):
        super().__init__(stage_name=stage_name)
        self.source_stage = source_stage
        self.diffusion_key = diffusion_key
        self.teacher_diffop_key = teacher_diffop_key
        self.probe_ids_key = probe_ids_key
        self.teacher_activations_key = teacher_activations_key
        self.raw_activations_key = raw_activations_key
        self.raw_activation_glob = raw_activation_glob
        self.source_file_index = source_file_index
        self.output_subdir = output_subdir or stage_name
        self.output_suffix = output_suffix
        self.phate_params = {
            "n_components": n_components,
            "random_state": random_state,
            "knn": knn,
            "t": t,
            "decay": decay,
            "gamma": gamma,
            "n_pca": n_pca,
            "n_landmark": n_landmark,
            "n_jobs": n_jobs,
            "verbose": verbose,
            "fit_fraction": fit_fraction,
        }
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
        return StageResult(outputs=outputs, metadata=metadata)

    @staticmethod
    def _layer_name_from_stem(stem: str) -> str:
        stem = stem.replace("__diffop", "")
        stem = re.sub(r"_step\d+$", "", stem)
        return stem

    @staticmethod
    def _to_2d(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.reshape(-1, 1)
        if x.ndim == 2:
            return x
        return x.reshape(x.shape[0], -1)

    @staticmethod
    def _center_scale(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        mean = x.mean(axis=0, keepdims=True)
        centered = x - mean
        norm = float(np.linalg.norm(centered))
        if norm < 1e-12:
            norm = 1.0
        return centered / norm, mean, norm

    @staticmethod
    def _pca_reduce(x: np.ndarray, n_components: int) -> np.ndarray:
        x_centered = x - x.mean(axis=0, keepdims=True)
        u, s, _vt = np.linalg.svd(x_centered, full_matrices=False)
        k = min(n_components, u.shape[1], s.shape[0])
        return u[:, :k] * s[:k]

    def _get_source_artifacts(self, context: StageContext) -> Dict[str, Any]:
        artifacts = context.artifacts.get(self.source_stage)
        if artifacts is None:
            raise KeyError(f"No artifacts found for source stage '{self.source_stage}'")
        return artifacts

    def _resolve_diffusion_paths(self, source_artifacts: Dict[str, Any]) -> List[Path]:
        primary = source_artifacts.get(self.teacher_diffop_key)
        if isinstance(primary, str) and primary:
            return [Path(primary)]

        value = source_artifacts.get(self.diffusion_key)
        if value is None:
            raise KeyError(
                f"Missing diffusion key '{self.diffusion_key}' in stage '{self.source_stage}' artifacts"
            )
        if isinstance(value, str):
            return [Path(value)]
        if isinstance(value, Sequence):
            return [Path(v) for v in value]
        raise TypeError(f"Unsupported diffusion value type: {type(value)}")

    def _load_teacher_activations(self, source_artifacts: Dict[str, Any]) -> np.ndarray:
        primary = source_artifacts.get(self.teacher_activations_key)
        if isinstance(primary, str) and primary:
            return self._to_2d(np.load(primary))

        raw_dir = source_artifacts.get(self.raw_activations_key)
        if raw_dir is None:
            raise KeyError(
                f"Missing raw activations key '{self.raw_activations_key}' in source artifacts"
            )
        raw_files = sorted(Path(raw_dir).glob(self.raw_activation_glob))
        if not raw_files:
            raise FileNotFoundError(
                f"No raw activation files found in '{raw_dir}' matching '{self.raw_activation_glob}'"
            )
        return self._to_2d(np.load(raw_files[self.source_file_index]))

    def _load_teacher_activation_map(self, source_artifacts: Dict[str, Any]) -> Dict[str, np.ndarray]:
        raw_dir = source_artifacts.get(self.raw_activations_key)
        if raw_dir is not None:
            raw_files = sorted(Path(raw_dir).glob(self.raw_activation_glob))
            if raw_files:
                return {
                    self._layer_name_from_stem(path.stem): self._to_2d(np.load(path))
                    for path in raw_files
                }

        primary = source_artifacts.get(self.teacher_activations_key)
        if isinstance(primary, str) and primary:
            return {"default": self._to_2d(np.load(primary))}

        teacher_acts = self._load_teacher_activations(source_artifacts)
        return {"default": teacher_acts}

    def _resolve_teacher_activations_for_diffusion(
        self,
        diffusion_path: Path,
        teacher_acts_by_layer: Dict[str, np.ndarray],
    ) -> tuple[str, np.ndarray]:
        if not teacher_acts_by_layer:
            raise ValueError("No teacher activations available for alignment")

        layer_name = self._layer_name_from_stem(diffusion_path.stem)
        if layer_name in teacher_acts_by_layer:
            return layer_name, teacher_acts_by_layer[layer_name]

        if len(teacher_acts_by_layer) == 1:
            only_layer, acts = next(iter(teacher_acts_by_layer.items()))
            return only_layer, acts

        available = ", ".join(sorted(teacher_acts_by_layer.keys()))
        raise KeyError(
            f"Could not match diffusion operator '{diffusion_path.name}' to teacher activations. "
            f"Available layers: {available}"
        )

    def _load_probe_ids(self, source_artifacts: Dict[str, Any], n_default: int) -> list[int]:
        probe_ids_path = source_artifacts.get(self.probe_ids_key)
        if isinstance(probe_ids_path, str):
            try:
                payload = json.loads(Path(probe_ids_path).read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    return [int(v) for v in payload]
            except Exception:
                pass
        return list(range(n_default))

    def _phate_target(self, diffusion_path: Path) -> np.ndarray:
        merged_operator = np.load(diffusion_path)
        x = torch.from_numpy(merged_operator).float()
        model = PHATEModule(**self.phate_params)
        target = model.fit_transform(x)
        target_np = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
        return self._to_2d(target_np)

    @staticmethod
    def _plot_coords(coords: np.ndarray, output_path: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("PHATE-1")
        ax.set_ylabel("PHATE-2")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _wandb_log_teacher_phate(layer_name: str, coords: np.ndarray, image_path: Path) -> None:
        if wandb is None or wandb.run is None:
            return
        safe = layer_name.replace(".", "_").replace("[", "_").replace("]", "_")
        table = wandb.Table(columns=["probe_index", "x", "y", "layer"])
        for idx, (x, y) in enumerate(coords):
            table.add_data(int(idx), float(x), float(y), str(layer_name))
        wandb.log(
            {
                f"teacher_phate/{safe}/table": table,
                f"teacher_phate/{safe}/image": wandb.Image(str(image_path)),
            }
        )

    def _procrustes_align(self, phate_target: np.ndarray, teacher_acts: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        n = min(phate_target.shape[0], teacher_acts.shape[0])
        if n < 2:
            raise ValueError("Need at least 2 samples for Procrustes alignment")

        x = phate_target[:n]
        y = teacher_acts[:n]

        target_dim = x.shape[1]
        teacher_reduced = y
        dim_reduced = False
        if y.shape[1] != target_dim:
            teacher_reduced = self._pca_reduce(y, n_components=target_dim)
            dim_reduced = True

        x_normed, x_mean, x_scale = self._center_scale(x)
        y_normed, y_mean, y_scale = self._center_scale(teacher_reduced)

        u, _s, vt = np.linalg.svd(x_normed.T @ y_normed, full_matrices=False)
        rotation = u @ vt
        x_aligned = (x_normed @ rotation) * y_scale + y_mean

        meta = {
            "n_samples_used": int(n),
            "phate_dim": int(x.shape[1]),
            "teacher_dim_original": int(y.shape[1]),
            "teacher_dim_aligned": int(teacher_reduced.shape[1]),
            "teacher_dim_reduced_with_pca": bool(dim_reduced),
            "x_scale": float(x_scale),
            "y_scale": float(y_scale),
            "rotation_shape": [int(v) for v in rotation.shape],
        }
        return x_aligned.astype(np.float32), meta

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        if self.external_manifest_path:
            return self._load_external_outputs()

        source_artifacts = self._get_source_artifacts(context)
        diffusion_paths = self._resolve_diffusion_paths(source_artifacts)
        teacher_acts_by_layer = self._load_teacher_activation_map(source_artifacts)

        stage_output_dir = stage_dir / self.output_subdir
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        aligned_targets: List[str] = []
        teacher_phate_targets: List[str] = []
        teacher_phate_targets_2d: List[str] = []
        teacher_phate_target_images_2d: List[str] = []
        aligned_target_paths_by_layer: Dict[str, str] = {}
        teacher_phate_target_paths_by_layer: Dict[str, str] = {}
        teacher_phate_target_paths_2d_by_layer: Dict[str, str] = {}
        teacher_phate_target_image_paths_2d_by_layer: Dict[str, str] = {}
        aligned_target_layers: List[str] = []
        index_rows: List[Dict[str, Any]] = []
        first_teacher_acts = next(iter(teacher_acts_by_layer.values()))
        probe_ids = self._load_probe_ids(source_artifacts, n_default=int(first_teacher_acts.shape[0]))

        for diffusion_path in diffusion_paths:
            layer_name, teacher_acts = self._resolve_teacher_activations_for_diffusion(
                diffusion_path=diffusion_path,
                teacher_acts_by_layer=teacher_acts_by_layer,
            )
            phate_target = self._phate_target(diffusion_path)
            phate_target_2d = phate_target[:, :2] if phate_target.shape[1] >= 2 else np.pad(
                phate_target,
                ((0, 0), (0, max(0, 2 - phate_target.shape[1]))),
            )
            aligned_target, align_meta = self._procrustes_align(phate_target, teacher_acts)

            stem = diffusion_path.stem
            phate_out_path = stage_output_dir / f"{stem}__teacher_phate_target.npy"
            np.save(phate_out_path, phate_target.astype(np.float32))
            teacher_phate_targets.append(str(phate_out_path))
            teacher_phate_target_paths_by_layer[layer_name] = str(phate_out_path)

            phate_2d_out_path = stage_output_dir / f"{stem}__teacher_phate_target_2d.npy"
            np.save(phate_2d_out_path, phate_target_2d.astype(np.float32))
            teacher_phate_targets_2d.append(str(phate_2d_out_path))
            teacher_phate_target_paths_2d_by_layer[layer_name] = str(phate_2d_out_path)

            phate_2d_image_path = stage_output_dir / f"{stem}__teacher_phate_target_2d.png"
            self._plot_coords(phate_target_2d, phate_2d_image_path, title=f"{layer_name} teacher PHATE")
            teacher_phate_target_images_2d.append(str(phate_2d_image_path))
            teacher_phate_target_image_paths_2d_by_layer[layer_name] = str(phate_2d_image_path)
            self._wandb_log_teacher_phate(layer_name, phate_target_2d, phate_2d_image_path)

            out_path = stage_output_dir / f"{stem}{self.output_suffix}"
            np.save(out_path, aligned_target)
            aligned_targets.append(str(out_path))
            aligned_target_paths_by_layer[layer_name] = str(out_path)
            aligned_target_layers.append(layer_name)

            meta = {
                "source_diffusion_operator": str(diffusion_path),
                "source_stage": self.source_stage,
                "layer_name": layer_name,
                "output_teacher_phate_target": str(phate_out_path),
                "output_teacher_phate_target_2d": str(phate_2d_out_path),
                "output_teacher_phate_target_2d_image": str(phate_2d_image_path),
                "output_aligned_target": str(out_path),
                "teacher_activation_shape": [int(v) for v in teacher_acts.shape],
                "phate_target_shape": [int(v) for v in phate_target.shape],
                "phate_target_2d_shape": [int(v) for v in phate_target_2d.shape],
                "aligned_target_shape": [int(v) for v in aligned_target.shape],
                "phate_params": self.phate_params,
                "procrustes": align_meta,
            }
            meta_path = stage_output_dir / f"{stem}.meta.json"
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

            index_rows.append(
                {
                    "source_diffusion_operator": str(diffusion_path),
                    "layer_name": layer_name,
                    "output_teacher_phate_target": str(phate_out_path),
                    "output_teacher_phate_target_2d": str(phate_2d_out_path),
                    "output_teacher_phate_target_2d_image": str(phate_2d_image_path),
                    "output_aligned_target": str(out_path),
                    "meta": str(meta_path),
                }
            )

        index_path = stage_output_dir / "aligned_targets_index.json"
        index_path.write_text(json.dumps(index_rows, indent=2, sort_keys=True), encoding="utf-8")
        aligned_probe_ids_path = stage_output_dir / "aligned_probe_ids.json"
        aligned_probe_ids_path.write_text(json.dumps(probe_ids, indent=2), encoding="utf-8")

        if aligned_targets:
            idx = min(max(self.source_file_index, -len(aligned_targets)), len(aligned_targets) - 1)
            aligned_target_path = aligned_targets[idx]
        else:
            aligned_target_path = ""

        return StageResult(
            outputs={
                "stage_output_dir": str(stage_output_dir),
                "aligned_targets": aligned_targets,
                "aligned_targets_index": str(index_path),
                "aligned_probe_ids_path": str(aligned_probe_ids_path),
                "aligned_target_layers": aligned_target_layers,
                "aligned_target_paths_by_layer": aligned_target_paths_by_layer,
                "teacher_phate_targets": teacher_phate_targets,
                "teacher_phate_target_paths_by_layer": teacher_phate_target_paths_by_layer,
                "teacher_phate_targets_2d": teacher_phate_targets_2d,
                "teacher_phate_target_images_2d": teacher_phate_target_images_2d,
                "teacher_phate_target_paths_2d_by_layer": teacher_phate_target_paths_2d_by_layer,
                "teacher_phate_target_image_paths_2d_by_layer": teacher_phate_target_image_paths_2d_by_layer,
                "aligned_target_path": aligned_target_path,
                # Alias for compatibility with downstream stage configs expecting target lists
                "phate_targets": aligned_targets,
                "probe_ids_path": str(aligned_probe_ids_path),
            },
            metadata={
                "num_diffusion_sources": len(diffusion_paths),
                "num_aligned_targets": len(aligned_targets),
                "n_components": self.phate_params["n_components"],
                "artifact_contract": "probe_target_v1",
            },
        )
