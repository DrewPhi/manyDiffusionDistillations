#!/usr/bin/env python3
"""Run the cheap final validation suite and write a report."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "results" / "final_validation"

SMOKE_RUNS = {
    "pythia": {
        "job_id": "9145958",
        "log": REPO_ROOT / "outputs" / "slurm" / "py70_loss_preflight-9145958.err",
        "must_contain": [
            "Syncing run pythia70_teacher14b_loss_logging_preflight_9145958",
            "Run summary:",
            "final_val_loss",
        ],
    },
    "qwen": {
        "job_id": "9145956",
        "log": REPO_ROOT / "outputs" / "slurm" / "qwen05_loss_preflight-9145956.out",
        "must_contain": [
            "Completed stage 'probe_teacher'",
            "Completed stage 'phate_teacher_target'",
            "Completed stage 'distill_sweep_grid'",
            "Completed stage 'sweep_results_sheet'",
        ],
    },
    "t5": {
        "job_id": "9146136",
        "log": REPO_ROOT / "outputs" / "slurm" / "t5small_loss_preflight-9146136.out",
        "must_contain": [
            "Completed stage 'probe_teacher'",
            "Completed stage 'phate_teacher_target'",
            "Completed stage 'distill_sweep_grid'",
            "Completed stage 'sweep_results_sheet'",
        ],
    },
}

PYTEST_TARGETS = [
    "tests/pipeline/test_distill_study_config.py",
    "tests/pipeline/test_distill_study_aggregator.py",
    "tests/pipeline/test_distillation_sweep_stage.py",
]


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: dict[str, Any]


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _check_smoke_logs() -> CheckResult:
    failures: list[str] = []
    details: dict[str, Any] = {}
    for family, cfg in SMOKE_RUNS.items():
        path = cfg["log"]
        if not path.exists():
            failures.append(f"{family}: missing log {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [needle for needle in cfg["must_contain"] if needle not in text]
        traceback_seen = "Traceback (most recent call last)" in text
        status = {
            "job_id": cfg["job_id"],
            "log": str(path),
            "missing_markers": missing,
            "traceback_seen": traceback_seen,
        }
        details[family] = status
        if missing:
            failures.append(f"{family}: missing markers {missing}")
        if traceback_seen:
            failures.append(f"{family}: traceback present in log")
    return CheckResult(
        name="family_smoke_logs",
        passed=not failures,
        details={"runs": details, "failures": failures},
    )


def _check_pytest() -> CheckResult:
    cmd = [sys.executable, "-m", "pytest", "-q", *PYTEST_TARGETS]
    completed = _run(cmd)
    return CheckResult(
        name="targeted_pytest",
        passed=completed.returncode == 0,
        details={
            "command": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )


def _check_study_manifest(run_root: Path) -> CheckResult:
    manifest_dir = run_root / "study_manifest"
    cmd = [
        sys.executable,
        "downstream/distill_family_study/scripts/submit_distill_study.py",
        "--manifest-dir",
        str(manifest_dir),
    ]
    completed = _run(cmd)

    details: dict[str, Any] = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "manifest_dir": str(manifest_dir),
    }
    if completed.returncode != 0:
        return CheckResult(name="study_manifest_dry_run", passed=False, details=details)

    manifest_json = manifest_dir / "submission_manifest.json"
    run_specs_dir = manifest_dir / "run_specs"
    if not manifest_json.exists() or not run_specs_dir.exists():
        details["missing_outputs"] = {
            "manifest_json_exists": manifest_json.exists(),
            "run_specs_dir_exists": run_specs_dir.exists(),
        }
        return CheckResult(name="study_manifest_dry_run", passed=False, details=details)

    rows = json.loads(manifest_json.read_text(encoding="utf-8"))
    family_counts = Counter(row["family"] for row in rows)
    layer_schemes = sorted({row["layer_scheme"] for row in rows})
    students = sorted({row["student_key"] for row in rows})
    submitted_flags = sorted({row["submitted"] for row in rows})
    run_spec_count = len(list(run_specs_dir.glob("*.json")))

    checks = {
        "row_count_is_54": len(rows) == 54,
        "run_spec_count_is_54": run_spec_count == 54,
        "family_counts_match": dict(family_counts) == {"pythia": 18, "qwen": 18, "t5": 18},
        "layer_schemes_match": layer_schemes == ["penultimate_only", "second_plus_penultimate"],
        "submitted_flags_false_only": submitted_flags == [False],
        "unique_run_names": len({row["run_name"] for row in rows}) == len(rows),
    }
    details.update(
        {
            "manifest_json": str(manifest_json),
            "run_specs_dir": str(run_specs_dir),
            "row_count": len(rows),
            "run_spec_count": run_spec_count,
            "family_counts": dict(family_counts),
            "layer_schemes": layer_schemes,
            "student_keys": students,
            "submitted_flags": submitted_flags,
            "checks": checks,
        }
    )
    return CheckResult(
        name="study_manifest_dry_run",
        passed=all(checks.values()),
        details=details,
    )


def _write_report(run_root: Path, results: list[CheckResult]) -> tuple[Path, Path]:
    report_json = run_root / "validation_report.json"
    report_md = run_root / "validation_report.md"
    overall_passed = all(result.passed for result in results)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "passed": overall_passed,
        "checks": [asdict(result) for result in results],
    }
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Final Validation Report",
        "",
        f"Overall result: {'PASS' if overall_passed else 'FAIL'}",
        "",
    ]
    for result in results:
        lines.append(f"## {result.name}")
        lines.append("")
        lines.append(f"Status: {'PASS' if result.passed else 'FAIL'}")
        lines.append("")
        if result.name == "family_smoke_logs":
            for family, info in result.details.get("runs", {}).items():
                lines.append(
                    f"- {family}: job {info['job_id']}, traceback={info['traceback_seen']}, "
                    f"missing_markers={info['missing_markers']}"
                )
        elif result.name == "targeted_pytest":
            lines.append(f"- command: `{' '.join(result.details['command'])}`")
            lines.append(f"- returncode: `{result.details['returncode']}`")
            stdout = (result.details.get("stdout") or "").strip()
            stderr = (result.details.get("stderr") or "").strip()
            if stdout:
                lines.extend(["", "```text", stdout, "```"])
            if stderr:
                lines.extend(["", "```text", stderr, "```"])
        elif result.name == "study_manifest_dry_run":
            lines.append(f"- command: `{' '.join(result.details['command'])}`")
            lines.append(f"- returncode: `{result.details['returncode']}`")
            for key in (
                "row_count",
                "run_spec_count",
                "family_counts",
                "layer_schemes",
                "student_keys",
                "submitted_flags",
            ):
                if key in result.details:
                    lines.append(f"- {key}: `{result.details[key]}`")
            checks = result.details.get("checks", {})
            for name, value in checks.items():
                lines.append(f"- {name}: `{value}`")
            stderr = (result.details.get("stderr") or "").strip()
            if stderr:
                lines.extend(["", "```text", stderr, "```"])
        failures = result.details.get("failures")
        if failures:
            lines.append("")
            for failure in failures:
                lines.append(f"- failure: {failure}")
        lines.append("")

    report_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_json, report_md


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = OUTPUT_ROOT / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    results = [
        _check_smoke_logs(),
        _check_pytest(),
        _check_study_manifest(run_root=run_root),
    ]
    report_json, report_md = _write_report(run_root=run_root, results=results)

    print(f"Wrote JSON report: {report_json}")
    print(f"Wrote Markdown report: {report_md}")
    for result in results:
        print(f"{result.name}: {'PASS' if result.passed else 'FAIL'}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
