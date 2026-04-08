#!/usr/bin/env python3
"""Submit a materialized distillation study to SLURM and save a job manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from materialize_distill_study import materialize_study
except ModuleNotFoundError:  # pragma: no cover - module import path under pytest/package execution
    from downstream.distill_family_study.scripts.materialize_distill_study import materialize_study
from omegaconf import OmegaConf


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                list(args),
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return None

    return {
        "git_commit": _run("git", "rev-parse", "HEAD"),
        "git_branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_porcelain": _run("git", "status", "--short"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-config",
        type=Path,
        default=Path("downstream/distill_family_study/configs/study/within_family_publication.yaml"),
    )
    parser.add_argument(
        "--runner-script",
        type=Path,
        default=Path("downstream/distill_family_study/scripts/run_distill_family_study_single.sbatch"),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("results/study_manifests/within_family_publication"),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually call sbatch. Without this flag, only write run specs and a dry-run manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of runs to materialize/submit.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        help="Restrict to one or more families, e.g. --family pythia --family qwen.",
    )
    parser.add_argument(
        "--student-key",
        action="append",
        default=None,
        help="Restrict to one or more student keys, e.g. --student-key pythia_410m.",
    )
    parser.add_argument(
        "--layer-scheme",
        action="append",
        default=None,
        help="Restrict to one or more layer schemes, e.g. --layer-scheme penultimate_only.",
    )
    parser.add_argument(
        "--probe-teacher-manifest",
        type=Path,
        default=None,
        help="Optional manifest for reusing probe_teacher artifacts across submitted runs.",
    )
    parser.add_argument(
        "--phate-target-manifest",
        type=Path,
        default=None,
        help="Optional manifest for reusing phate_teacher_target artifacts across submitted runs. Only valid when all submitted runs share the same student penultimate dimension.",
    )
    return parser.parse_args()


def _ensure_dirs(base: Path) -> tuple[Path, Path]:
    run_specs_dir = base / "run_specs"
    base.mkdir(parents=True, exist_ok=True)
    run_specs_dir.mkdir(parents=True, exist_ok=True)
    return base, run_specs_dir


def _write_run_spec(path: Path, run: dict[str, Any]) -> None:
    path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")


def _build_sbatch_command(runner_script: Path, run_spec_path: Path, run_name: str) -> list[str]:
    return [
        "sbatch",
        "--parsable",
        f"--job-name={run_name[:128]}",
        f"--export=ALL,RUN_SPEC_PATH={run_spec_path}",
        str(runner_script),
    ]


def _write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "family",
        "student_key",
        "layer_scheme",
        "training_regime",
        "staged_training_enabled",
        "seed",
        "run_spec_path",
        "submitted",
        "job_id",
        "study_config_path",
        "runner_script",
        "git_commit",
        "sbatch_command",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fieldnames}
            flat["sbatch_command"] = json.dumps(flat["sbatch_command"], separators=(",", ":"))
            writer.writerow(flat)


def _inject_reusable_artifacts(
    run: dict[str, Any],
    probe_manifest: Path | None,
    phate_manifest: Path | None,
) -> dict[str, Any]:
    updated = dict(run)
    overrides = list(run["hydra_overrides"])
    if probe_manifest is not None:
        overrides.append(
            "stage_pipeline.params.reusable_artifacts.probe_teacher_manifest_path="
            f"{probe_manifest.resolve()}"
        )
    if phate_manifest is not None:
        overrides.append(
            "stage_pipeline.params.reusable_artifacts.phate_teacher_target_manifest_path="
            f"{phate_manifest.resolve()}"
        )
    updated["hydra_overrides"] = overrides
    return updated


def _filter_runs(
    runs: list[dict[str, Any]],
    *,
    families: list[str] | None = None,
    student_keys: list[str] | None = None,
    layer_schemes: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    filtered = list(runs)

    if families:
        allowed = set(families)
        filtered = [run for run in filtered if run["family"] in allowed]
    if student_keys:
        allowed_students = set(student_keys)
        filtered = [run for run in filtered if run["student_key"] in allowed_students]
    if layer_schemes:
        allowed_layer_schemes = set(layer_schemes)
        filtered = [run for run in filtered if run["layer_scheme"] in allowed_layer_schemes]
    if limit is not None:
        filtered = filtered[:limit]

    return filtered


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(args.study_config)
    runs = [run.as_dict() for run in materialize_study(cfg)]
    git_meta = _git_metadata(repo_root)
    runs = _filter_runs(
        runs,
        families=args.family,
        student_keys=args.student_key,
        layer_schemes=args.layer_scheme,
        limit=args.limit,
    )

    if args.phate_target_manifest is not None:
        dims = {run["student_penultimate_dim"] for run in runs}
        if len(dims) > 1:
            raise ValueError(
                "--phate-target-manifest can only be used when the submitted runs share one student penultimate dimension"
            )

    runs = [
        _inject_reusable_artifacts(
            run,
            probe_manifest=args.probe_teacher_manifest,
            phate_manifest=args.phate_target_manifest,
        )
        for run in runs
    ]

    manifest_dir, run_specs_dir = _ensure_dirs(args.manifest_dir)
    manifest_rows: list[dict[str, Any]] = []
    submitted_at = datetime.now(timezone.utc).isoformat()
    study_metadata = {
        "study_config_path": str(args.study_config.resolve()),
        "runner_script": str(args.runner_script.resolve()),
        "submitted_at": submitted_at,
        "git": git_meta,
        "study_name": cfg.study.name,
        "study_description": cfg.study.description,
        "family_order": list(cfg.study.family_order),
        "layer_schemes": list(cfg.study.layer_schemes),
        "seeds": list(cfg.study.seeds),
    }

    for index, run in enumerate(runs):
        run_spec_path = run_specs_dir / f"{index:03d}_{run['run_name']}.json"
        _write_run_spec(run_spec_path, run)

        sbatch_command = _build_sbatch_command(
            runner_script=args.runner_script,
            run_spec_path=run_spec_path.resolve(),
            run_name=run["run_name"],
        )
        row = {
            "run_name": run["run_name"],
            "family": run["family"],
            "student_key": run["student_key"],
            "layer_scheme": run["layer_scheme"],
            "training_regime": run["training_regime"],
            "staged_training_enabled": run["staged_training_enabled"],
            "seed": run["seed"],
            "run_spec_path": str(run_spec_path.resolve()),
            "submitted": args.submit,
            "submitted_at": submitted_at,
            "job_id": None,
            "study_config_path": str(args.study_config.resolve()),
            "runner_script": str(args.runner_script.resolve()),
            "git_commit": git_meta.get("git_commit"),
            "sbatch_command": sbatch_command,
        }

        if args.submit:
            completed = subprocess.run(
                sbatch_command,
                check=True,
                capture_output=True,
                text=True,
            )
            row["job_id"] = completed.stdout.strip()

        manifest_rows.append(row)

    manifest_json = manifest_dir / "submission_manifest.json"
    manifest_csv = manifest_dir / "submission_manifest.csv"
    study_metadata_json = manifest_dir / "study_metadata.json"
    manifest_json.write_text(json.dumps(manifest_rows, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest_csv(manifest_csv, manifest_rows)
    study_metadata_json.write_text(json.dumps(study_metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {len(manifest_rows)} run specs to {run_specs_dir}")
    print(f"Manifest JSON: {manifest_json}")
    print(f"Manifest CSV:  {manifest_csv}")
    print(f"Study metadata: {study_metadata_json}")
    if args.submit:
        print("Submitted jobs:")
        for row in manifest_rows:
            print(f"  {row['job_id']}  {row['run_name']}")
    else:
        print("Dry run only. Re-run with --submit to call sbatch.")


if __name__ == "__main__":
    main()
