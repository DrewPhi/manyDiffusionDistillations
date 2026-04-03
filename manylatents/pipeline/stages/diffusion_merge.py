"""Diffusion-operator merge stage for staged representation pipelines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from manylatents.algorithms.latent.merging import DiffusionMerging
from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult


class DiffusionMergeStage(PipelineStage):
    """Merge diffusion operators emitted by upstream stages.

    Typical use:
    - Run multiple probe extraction stages (one per model).
    - Merge their diffusion operators via frobenius_mean or ot_barycenter.
    """

    def __init__(
        self,
        stage_name: str,
        source_stages: List[str] | None = None,
        source_key: str = "diffusion_operators",
        operator_index: int = -1,
        strategy: str = "frobenius_mean",
        weights: Dict[str, float] | None = None,
        normalize_output: bool = True,
        merge_mode: str = "all",
        explicit_pairs: List[List[str]] | None = None,
        output_subdir: str | None = None,
    ):
        super().__init__(stage_name=stage_name)
        self.source_stages = source_stages
        self.source_key = source_key
        self.operator_index = operator_index
        self.strategy = strategy
        self.weights = weights or {}
        self.normalize_output = normalize_output
        self.merge_mode = merge_mode
        self.explicit_pairs = explicit_pairs or []
        self.output_subdir = output_subdir or stage_name

    def _resolve_source_stages(self, context: StageContext) -> List[str]:
        stages = self.source_stages or sorted(context.artifacts.keys())
        if not stages:
            raise ValueError("DiffusionMergeStage found no source stages in artifacts")
        return stages

    def _get_operator_path(self, stage_artifacts: Dict[str, Any]) -> str:
        value = stage_artifacts.get(self.source_key)
        if value is None:
            raise KeyError(f"Missing source key '{self.source_key}' in stage artifacts")

        if isinstance(value, str):
            return value

        if isinstance(value, Sequence):
            items = list(value)
            if not items:
                raise ValueError(f"Source key '{self.source_key}' has an empty list")
            return str(items[self.operator_index])

        raise TypeError(f"Unsupported source artifact type for key '{self.source_key}': {type(value)}")

    def _load_stage_operators(self, context: StageContext, source_stages: List[str]) -> Dict[str, np.ndarray]:
        operators: Dict[str, np.ndarray] = {}
        for stage_name in source_stages:
            stage_artifacts = context.artifacts.get(stage_name)
            if stage_artifacts is None:
                raise KeyError(f"No artifacts found for source stage '{stage_name}'")
            path = Path(self._get_operator_path(stage_artifacts))
            operators[stage_name] = np.load(path)
        return operators

    def _pair_list(self, source_stages: List[str]) -> List[Tuple[str, ...]]:
        if self.merge_mode == "all":
            return [tuple(source_stages)]

        if self.merge_mode == "adjacent_pairs":
            if len(source_stages) < 2:
                return []
            return [tuple(source_stages[i : i + 2]) for i in range(len(source_stages) - 1)]

        if self.merge_mode == "explicit_pairs":
            if not self.explicit_pairs:
                raise ValueError("merge_mode='explicit_pairs' requires explicit_pairs")
            return [tuple(pair) for pair in self.explicit_pairs]

        raise ValueError(
            f"Unknown merge_mode '{self.merge_mode}'. Expected one of: all, adjacent_pairs, explicit_pairs"
        )

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        source_stages = self._resolve_source_stages(context)
        stage_output_dir = stage_dir / self.output_subdir
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        operators = self._load_stage_operators(context, source_stages)
        grouping = self._pair_list(source_stages)

        merger = DiffusionMerging(
            strategy=self.strategy,
            weights=self.weights,
            normalize_output=self.normalize_output,
        )

        merged_paths: List[str] = []
        merge_index: List[Dict[str, Any]] = []

        for members in grouping:
            selected = {name: operators[name] for name in members}
            merged = merger.merge(selected)
            name = "__".join(members)
            output_path = stage_output_dir / f"merged_{self.strategy}__{name}.npy"
            np.save(output_path, merged)

            merged_paths.append(str(output_path))
            merge_index.append(
                {
                    "group": list(members),
                    "output": str(output_path),
                    "shape": list(merged.shape),
                }
            )

        index_path = stage_output_dir / "merge_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "strategy": self.strategy,
                    "merge_mode": self.merge_mode,
                    "source_stages": source_stages,
                    "operator_index": self.operator_index,
                    "groups": merge_index,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return StageResult(
            outputs={
                "stage_output_dir": str(stage_output_dir),
                "merged_diffusion_operators": merged_paths,
                "merge_index": str(index_path),
            },
            metadata={
                "num_sources": len(source_stages),
                "num_outputs": len(merged_paths),
                "strategy": self.strategy,
                "merge_mode": self.merge_mode,
            },
        )
