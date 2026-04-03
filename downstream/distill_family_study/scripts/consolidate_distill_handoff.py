#!/usr/bin/env python3
"""Consolidate distillation handoff status from actual result artifacts.

This script exists as a sanity check layer for handoff. It reads the latest
result JSON artifacts and emits a compact JSON summary so reviewers do not need
to trust narrative notes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _latest_dir(base: Path) -> Path | None:
    dirs = [path for path in base.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_final_validation(repo_root: Path) -> dict[str, Any]:
    base = repo_root / "results" / "final_validation"
    latest = _latest_dir(base) if base.exists() else None
    report = _load_json(latest / "validation_report.json" if latest else None)
    return {
        "latest_dir": str(latest) if latest else None,
        "report": report,
    }


def _latest_mini_validation(repo_root: Path) -> dict[str, Any]:
    base = repo_root / "results" / "mini_launch_validation"
    latest = _latest_dir(base) if base.exists() else None
    report = _load_json(latest / "report" / "mini_validation_report.json" if latest else None)
    manifest = _load_json(latest / "manifest" / "submission_manifest.json" if latest else None)
    metadata = _load_json(latest / "manifest" / "study_metadata.json" if latest else None)
    return {
        "latest_dir": str(latest) if latest else None,
        "report": report,
        "submission_manifest": manifest,
        "study_metadata": metadata,
    }


def _checkpoint_evidence(repo_root: Path) -> dict[str, Any]:
    base = repo_root / "outputs" / "pipelines"
    if not base.exists():
        return {"latest_checkpoints": []}

    checkpoint_rows: list[dict[str, Any]] = []
    for path in base.rglob("student_step*.pt"):
        stat = path.stat()
        checkpoint_rows.append(
            {
                "path": str(path),
                "mtime": stat.st_mtime,
                "size_bytes": stat.st_size,
            }
        )

    checkpoint_rows.sort(key=lambda row: row["mtime"])
    return {
        "latest_checkpoints": checkpoint_rows[-10:],
    }


def _summarize(final_validation: dict[str, Any], mini_validation: dict[str, Any], checkpoints: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "final_validation_passed": None,
        "family_smokes_passed": None,
        "targeted_pytest_passed": None,
        "manifest_dry_run_passed": None,
        "mini_validation_present": mini_validation["report"] is not None,
        "mini_validation_passed": None,
        "latest_checkpoint_count": len(checkpoints.get("latest_checkpoints", [])),
        "latest_checkpoint_paths": [row["path"] for row in checkpoints.get("latest_checkpoints", [])[-3:]],
    }

    final_report = final_validation.get("report") or {}
    if final_report:
        summary["final_validation_passed"] = bool(final_report.get("passed"))
        checks = {check["name"]: check for check in final_report.get("checks", [])}
        summary["family_smokes_passed"] = bool(checks.get("family_smoke_logs", {}).get("passed"))
        summary["targeted_pytest_passed"] = bool(checks.get("targeted_pytest", {}).get("passed"))
        summary["manifest_dry_run_passed"] = bool(checks.get("study_manifest_dry_run", {}).get("passed"))

    mini_report = mini_validation.get("report") or {}
    if mini_report:
        summary["mini_validation_passed"] = bool(mini_report.get("passed"))
        run_outcomes = mini_report.get("run_outcomes", [])
        summary["mini_validation_run_count"] = int(mini_report.get("manifest_run_count", 0))
        summary["mini_validation_recorded_runs"] = len(run_outcomes)
        summary["mini_validation_successful_runs"] = sum(1 for row in run_outcomes if row.get("success"))

    return summary


def build_handoff_summary(repo_root: Path) -> dict[str, Any]:
    final_validation = _latest_final_validation(repo_root)
    mini_validation = _latest_mini_validation(repo_root)
    checkpoints = _checkpoint_evidence(repo_root)
    summary = _summarize(final_validation, mini_validation, checkpoints)
    return {
        "repo_root": str(repo_root),
        "final_validation": final_validation,
        "mini_validation": mini_validation,
        "checkpoint_evidence": checkpoints,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/distill_handoff_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_handoff_summary(args.repo_root.resolve())
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
