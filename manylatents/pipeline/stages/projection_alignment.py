"""Projection alignment training + evaluation stage."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from manylatents.pipeline.stages.base import PipelineStage, StageContext, StageResult


class ProjectionAlignmentStage(PipelineStage):
    """Train projection heads to align source activations to PHATE targets."""

    def __init__(
        self,
        stage_name: str,
        source_stages: List[str] | None = None,
        target_stage: str = "phate_targets",
        target_key: str = "phate_targets",
        target_selection: str = "single",
        target_index: int = 0,
        raw_activations_key: str = "raw_activations_dir",
        fallback_source_key: str = "embeddings",
        raw_activation_glob: str = "*.npy",
        source_file_index: int = -1,
        test_fraction: float = 0.2,
        random_state: int = 42,
        ridge_alpha: float = 1e-3,
        fit_intercept: bool = True,
        standardize_inputs: bool = True,
        output_subdir: str | None = None,
    ):
        super().__init__(stage_name=stage_name)
        self.source_stages = source_stages
        self.target_stage = target_stage
        self.target_key = target_key
        self.target_selection = target_selection
        self.target_index = target_index
        self.raw_activations_key = raw_activations_key
        self.fallback_source_key = fallback_source_key
        self.raw_activation_glob = raw_activation_glob
        self.source_file_index = source_file_index
        self.test_fraction = test_fraction
        self.random_state = random_state
        self.ridge_alpha = ridge_alpha
        self.fit_intercept = fit_intercept
        self.standardize_inputs = standardize_inputs
        self.output_subdir = output_subdir or stage_name

    def _resolve_source_stages(self, context: StageContext) -> List[str]:
        stages = self.source_stages or sorted([k for k in context.artifacts.keys() if k != self.target_stage])
        if not stages:
            raise ValueError("ProjectionAlignmentStage found no source stages")
        return stages

    def _target_paths(self, context: StageContext) -> List[Path]:
        artifacts = context.artifacts.get(self.target_stage)
        if artifacts is None:
            raise KeyError(f"No artifacts found for target_stage '{self.target_stage}'")

        value = artifacts.get(self.target_key)
        if value is None:
            raise KeyError(f"Missing target key '{self.target_key}' in stage '{self.target_stage}' artifacts")

        if isinstance(value, str):
            return [Path(value)]
        if isinstance(value, Sequence):
            return [Path(v) for v in value]
        raise TypeError(f"Unsupported target value type: {type(value)}")

    def _pick_target_path(self, target_paths: List[Path], source_idx: int) -> Path:
        if not target_paths:
            raise ValueError("No PHATE target paths available")

        if self.target_selection == "single":
            return target_paths[self.target_index]
        if self.target_selection == "by_source_index":
            idx = min(source_idx, len(target_paths) - 1)
            return target_paths[idx]
        raise ValueError("target_selection must be one of: single, by_source_index")

    def _load_source_matrix(self, stage_artifacts: Dict[str, Any]) -> np.ndarray:
        raw_dir = stage_artifacts.get(self.raw_activations_key)
        if raw_dir is not None:
            raw_files = sorted(Path(raw_dir).glob(self.raw_activation_glob))
            if raw_files:
                x = np.load(raw_files[self.source_file_index])
                return self._to_2d(x)

        fallback = stage_artifacts.get(self.fallback_source_key)
        if fallback is None:
            raise KeyError(
                f"Missing both '{self.raw_activations_key}' and fallback '{self.fallback_source_key}' in source artifacts"
            )

        return self._to_2d(np.load(fallback))

    @staticmethod
    def _to_2d(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.reshape(-1, 1)
        if x.ndim == 2:
            return x
        return x.reshape(x.shape[0], -1)

    def _train_test_split(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n < 2:
            raise ValueError("Need at least 2 samples for projection alignment")
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n)
        rng.shuffle(indices)
        n_test = max(1, int(round(n * self.test_fraction)))
        n_test = min(n_test, n - 1)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        return train_idx, test_idx

    def _fit_ridge(self, x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, np.ndarray]:
        eps = 1e-8

        if self.standardize_inputs:
            x_mean = x_train.mean(axis=0)
            x_std = x_train.std(axis=0)
            x_std = np.where(x_std < eps, 1.0, x_std)
            x_used = (x_train - x_mean) / x_std
        else:
            x_mean = np.zeros(x_train.shape[1], dtype=x_train.dtype)
            x_std = np.ones(x_train.shape[1], dtype=x_train.dtype)
            x_used = x_train

        if self.fit_intercept:
            y_mean = y_train.mean(axis=0)
            y_centered = y_train - y_mean
        else:
            y_mean = np.zeros(y_train.shape[1], dtype=y_train.dtype)
            y_centered = y_train

        xtx = x_used.T @ x_used
        reg = self.ridge_alpha * np.eye(xtx.shape[0], dtype=xtx.dtype)
        xty = x_used.T @ y_centered
        w = np.linalg.solve(xtx + reg, xty)

        if self.fit_intercept:
            b = y_mean
        else:
            b = np.zeros(y_train.shape[1], dtype=y_train.dtype)

        return {
            "weights": w,
            "bias": b,
            "x_mean": x_mean,
            "x_std": x_std,
        }

    def _predict(self, x: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
        x_used = (x - model["x_mean"]) / model["x_std"]
        return x_used @ model["weights"] + model["bias"]

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        resid = y_true - y_pred
        mse = float(np.mean(resid ** 2))

        ss_res = float(np.sum(resid ** 2))
        y_mean = y_true.mean(axis=0, keepdims=True)
        ss_tot = float(np.sum((y_true - y_mean) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        y_true_norm = np.linalg.norm(y_true, axis=1) + 1e-8
        y_pred_norm = np.linalg.norm(y_pred, axis=1) + 1e-8
        cosine = np.sum(y_true * y_pred, axis=1) / (y_true_norm * y_pred_norm)
        cosine_mean = float(np.mean(cosine))

        return {
            "mse": mse,
            "r2": r2,
            "cosine_mean": cosine_mean,
        }

    def run(self, context: StageContext, stage_dir: Path) -> StageResult:
        source_stages = self._resolve_source_stages(context)
        target_paths = self._target_paths(context)

        stage_output_dir = stage_dir / self.output_subdir
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        report_rows: List[Dict[str, Any]] = []
        model_paths: Dict[str, str] = {}
        metrics_paths: Dict[str, str] = {}

        for i, source_stage in enumerate(source_stages):
            source_artifacts = context.artifacts.get(source_stage)
            if source_artifacts is None:
                raise KeyError(f"No artifacts found for source stage '{source_stage}'")

            x = self._load_source_matrix(source_artifacts)
            target_path = self._pick_target_path(target_paths, i)
            y = self._to_2d(np.load(target_path))

            n = min(x.shape[0], y.shape[0])
            x = x[:n]
            y = y[:n]

            train_idx, test_idx = self._train_test_split(n)
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]

            model = self._fit_ridge(x_train, y_train)
            y_train_pred = self._predict(x_train, model)
            y_test_pred = self._predict(x_test, model)

            train_metrics = self._metrics(y_train, y_train_pred)
            test_metrics = self._metrics(y_test, y_test_pred)

            source_dir = stage_output_dir / source_stage
            source_dir.mkdir(parents=True, exist_ok=True)

            model_path = source_dir / "projection_model.npz"
            np.savez(
                model_path,
                weights=model["weights"],
                bias=model["bias"],
                x_mean=model["x_mean"],
                x_std=model["x_std"],
                source_stage=source_stage,
                target_path=str(target_path),
            )

            np.save(source_dir / "y_test.npy", y_test)
            np.save(source_dir / "y_test_pred.npy", y_test_pred)

            metrics = {
                "source_stage": source_stage,
                "target_path": str(target_path),
                "n_samples": int(n),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train": train_metrics,
                "test": test_metrics,
            }
            metrics_path = source_dir / "alignment_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

            row = {
                "source_stage": source_stage,
                "target_path": str(target_path),
                "n_samples": int(n),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_mse": train_metrics["mse"],
                "train_r2": train_metrics["r2"],
                "train_cosine_mean": train_metrics["cosine_mean"],
                "test_mse": test_metrics["mse"],
                "test_r2": test_metrics["r2"],
                "test_cosine_mean": test_metrics["cosine_mean"],
            }
            report_rows.append(row)

            model_paths[source_stage] = str(model_path)
            metrics_paths[source_stage] = str(metrics_path)

        report_json_path = stage_output_dir / "projection_alignment_report.json"
        report_json_path.write_text(json.dumps(report_rows, indent=2, sort_keys=True), encoding="utf-8")

        report_csv_path = stage_output_dir / "projection_alignment_report.csv"
        fieldnames = [
            "source_stage",
            "target_path",
            "n_samples",
            "n_train",
            "n_test",
            "train_mse",
            "train_r2",
            "train_cosine_mean",
            "test_mse",
            "test_r2",
            "test_cosine_mean",
        ]
        with report_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)

        return StageResult(
            outputs={
                "stage_output_dir": str(stage_output_dir),
                "projection_report_json": str(report_json_path),
                "projection_report_csv": str(report_csv_path),
                "projection_models": model_paths,
                "projection_metrics": metrics_paths,
            },
            metadata={
                "num_sources": len(source_stages),
                "num_targets": len(target_paths),
                "ridge_alpha": self.ridge_alpha,
                "test_fraction": self.test_fraction,
            },
        )
