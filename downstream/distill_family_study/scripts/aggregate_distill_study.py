#!/usr/bin/env python3
"""Aggregate publication-scale distillation study runs into study-level tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("results/study_manifests/within_family_publication"),
        help="Directory containing run_specs/ from submit_distill_study.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/publication_within_family"),
        help="Directory for aggregated study artifacts.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    return out


def _override_value(overrides: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for override in overrides:
        if override.startswith(prefix):
            return override[len(prefix):]
    return None


def _run_output_dir(repo_root: Path, run_spec: dict[str, Any]) -> Path:
    output_dir = _override_value(run_spec["hydra_overrides"], "output_dir") or "./outputs"
    output_root = (repo_root / output_dir).resolve()
    return output_root / "pipelines" / run_spec["run_name"]


def _load_run_specs(run_specs_dir: Path) -> list[dict[str, Any]]:
    return [_read_json(path) for path in sorted(run_specs_dir.glob("*.json"))]


def _load_result_rows(repo_root: Path, run_specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for run_spec in run_specs:
        run_dir = _run_output_dir(repo_root, run_spec)
        result_path = run_dir / "distill_sweep_grid" / "distill_sweep_grid" / "sweep_results.json"
        if not result_path.exists():
            missing.append(
                {
                    "run_name": run_spec["run_name"],
                    "expected_results_path": str(result_path),
                    "family": run_spec.get("family"),
                    "student_key": run_spec.get("student_key"),
                }
            )
            continue

        payload = _read_json(result_path)
        if not isinstance(payload, list):
            raise TypeError(f"Expected JSON list at {result_path}")
        for row in payload:
            combined = dict(row)
            combined["study_name"] = run_spec["study_name"]
            combined["study_run_name"] = run_spec["run_name"]
            combined["study_family"] = run_spec["family"]
            combined["study_student_key"] = run_spec["student_key"]
            combined["study_layer_scheme"] = run_spec["layer_scheme"]
            combined["study_lambda_align"] = run_spec["lambda_align"]
            combined["study_seed"] = run_spec["seed"]
            combined["study_probe_size"] = run_spec["probe_size"]
            combined["study_run_spec_path"] = run_spec.get("run_spec_path", "")
            combined["study_expected_run_dir"] = str(run_dir)
            rows.append(combined)

    return rows, missing


def _group_mean(rows: list[dict[str, Any]], group_keys: list[str], metric_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        bucket_key = tuple(row.get(key) for key in group_keys)
        buckets.setdefault(bucket_key, []).append(row)

    out: list[dict[str, Any]] = []
    for bucket_key, bucket_rows in sorted(buckets.items()):
        summary = {key: value for key, value in zip(group_keys, bucket_key)}
        summary["num_runs"] = len(bucket_rows)
        for metric in metric_keys:
            values = [_safe_float(row.get(metric)) for row in bucket_rows]
            finite = [value for value in values if math.isfinite(value)]
            summary[f"{metric}_mean"] = sum(finite) / len(finite) if finite else float("nan")
        out.append(summary)
    return out


def _best_per_student(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("family"),
            row.get("student_model"),
        )
        buckets.setdefault(key, []).append(row)

    winners: list[dict[str, Any]] = []
    for _key, bucket_rows in sorted(buckets.items()):
        ranked = sorted(
            bucket_rows,
            key=lambda row: (
                not math.isfinite(_safe_float(row.get("val_loss"))),
                _safe_float(row.get("val_loss")),
                _safe_float(row.get("align_mse")),
            ),
        )
        winners.append(ranked[0])
    return winners


def _study_summary(rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    for row in rows:
        family = str(row.get("family"))
        family_counts[family] = family_counts.get(family, 0) + 1
    return {
        "num_completed_rows": len(rows),
        "num_missing_runs": len(missing),
        "families": family_counts,
        "result_schema_versions": sorted({str(row.get("result_schema_version")) for row in rows}),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    run_specs_dir = args.manifest_dir / "run_specs"
    if not run_specs_dir.exists():
        raise FileNotFoundError(f"Run specs directory does not exist: {run_specs_dir}")

    run_specs = _load_run_specs(run_specs_dir)
    rows, missing = _load_result_rows(repo_root=repo_root, run_specs=run_specs)

    metric_keys = [
        "val_loss",
        "val_perplexity",
        "test_loss",
        "test_perplexity",
        "align_mse",
        "train_lm_loss_last",
        "train_align_loss_last",
        "train_total_loss_last",
    ]
    family_summary = _group_mean(
        rows,
        group_keys=["family", "layer_scheme"],
        metric_keys=metric_keys,
    )
    best_rows = _best_per_student(rows)
    summary = _study_summary(rows, missing)

    _write_json(args.output_dir / "master_rows.json", rows)
    _write_csv(args.output_dir / "master_rows.csv", rows)
    _write_json(args.output_dir / "family_summary.json", family_summary)
    _write_csv(args.output_dir / "family_summary.csv", family_summary)
    _write_json(args.output_dir / "best_per_student.json", best_rows)
    _write_csv(args.output_dir / "best_per_student.csv", best_rows)
    _write_json(args.output_dir / "missing_runs.json", missing)
    _write_json(args.output_dir / "study_summary.json", summary)

    print(f"Loaded {len(run_specs)} run specs from {run_specs_dir}")
    print(f"Aggregated {len(rows)} completed result rows")
    print(f"Missing runs: {len(missing)}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
