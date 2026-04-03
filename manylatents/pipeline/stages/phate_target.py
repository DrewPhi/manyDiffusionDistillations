"""PHATE target construction stage for merged diffusion operators."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from manylatents.algorithms.latent.phate import PHATEModule
from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult


class PHATETargetStage(PipelineStage):
    """Build PHATE embeddings from merged diffusion operators."""

    def __init__(
        self,
        stage_name: str,
        source_stage: str = "diffusion_merge",
        source_key: str = "merged_diffusion_operators",
        output_subdir: str | None = None,
        output_suffix: str = "__phate_target.npy",
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
        backend: str | None = None,
        device: str | None = None,
        neighborhood_size: int | None = None,
    ):
        super().__init__(stage_name=stage_name)
        self.source_stage = source_stage
        self.source_key = source_key
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
            "backend": backend,
            "device": device,
            "neighborhood_size": neighborhood_size,
        }

    def _source_paths(self, context: StageContext) -> List[Path]:
        artifacts = context.artifacts.get(self.source_stage)
        if artifacts is None:
            raise KeyError(f"No artifacts found for source_stage '{self.source_stage}'")

        value = artifacts.get(self.source_key)
        if value is None:
            raise KeyError(f"Missing source key '{self.source_key}' in stage '{self.source_stage}' artifacts")

        if isinstance(value, str):
            return [Path(value)]

        if isinstance(value, Sequence):
            return [Path(v) for v in value]

        raise TypeError(f"Unsupported source type for '{self.source_key}': {type(value)}")

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        source_paths = self._source_paths(context)
        stage_output_dir = stage_dir / self.output_subdir
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        phate_targets: List[str] = []
        index_rows: List[Dict[str, Any]] = []

        for source_path in source_paths:
            merged_operator = np.load(source_path)
            x = torch.from_numpy(merged_operator).float()

            model = PHATEModule(**self.phate_params)
            target = model.fit_transform(x)
            target_np = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)

            stem = source_path.stem
            out_path = stage_output_dir / f"{stem}{self.output_suffix}"
            np.save(out_path, target_np)
            phate_targets.append(str(out_path))

            meta_path = stage_output_dir / f"{stem}.meta.json"
            meta = {
                "source_operator": str(source_path),
                "output_target": str(out_path),
                "input_shape": list(merged_operator.shape),
                "target_shape": list(target_np.shape),
                "phate_params": self.phate_params,
            }
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

            index_rows.append({
                "source_operator": str(source_path),
                "output_target": str(out_path),
                "meta": str(meta_path),
            })

        index_path = stage_output_dir / "phate_targets_index.json"
        index_path.write_text(json.dumps(index_rows, indent=2, sort_keys=True), encoding="utf-8")

        return StageResult(
            outputs={
                "stage_output_dir": str(stage_output_dir),
                "phate_targets": phate_targets,
                "phate_targets_index": str(index_path),
            },
            metadata={
                "num_source_operators": len(source_paths),
                "num_targets": len(phate_targets),
                "n_components": self.phate_params["n_components"],
            },
        )
