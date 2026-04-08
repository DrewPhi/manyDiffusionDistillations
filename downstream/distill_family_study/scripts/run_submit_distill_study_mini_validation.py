#!/usr/bin/env python3
"""Run the minimal sequential launcher validation and write a final report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_SCRIPT = REPO_ROOT / "downstream" / "distill_family_study" / "scripts" / "run_distill_family_study_single.sbatch"
SUBMIT_SCRIPT = REPO_ROOT / "downstream" / "distill_family_study" / "scripts" / "submit_distill_study.py"
AGGREGATE_SCRIPT = REPO_ROOT / "downstream" / "distill_family_study" / "scripts" / "aggregate_distill_study.py"
PLOT_SCRIPT = REPO_ROOT / "downstream" / "distill_family_study" / "scripts" / "plot_distill_study.py"


@dataclass
class RunOutcome:
    run_name: str
    family: str
    student_key: str
    layer_scheme: str
    staged_training_enabled: bool
    run_spec_path: str
    slurm_job_id: str | None
    returncode: int
    success: bool
    stdout: str
    stderr: str


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _materialize_manifest(manifest_dir: Path) -> Path:
    cmd = [
        sys.executable,
        str(SUBMIT_SCRIPT),
        "--manifest-dir",
        str(manifest_dir),
        "--family",
        "pythia",
        "--family",
        "qwen",
        "--family",
        "bert",
        "--family",
        "deberta_v3",
        "--student-key",
        "pythia_410m",
        "--student-key",
        "qwen2_5_0_5b",
        "--student-key",
        "bert_11m",
        "--student-key",
        "deberta_v3_xsmall",
        "--layer-scheme",
        "penultimate_only",
    ]
    completed = _run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to materialize mini validation manifest.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    manifest_json = manifest_dir / "submission_manifest.json"
    if not manifest_json.exists():
        raise FileNotFoundError(f"Missing manifest JSON after dry run: {manifest_json}")
    return manifest_json


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected list payload in {path}")
    return rows


def _submit_run(row: dict[str, Any], *, prepare_hf_cache: str = "1") -> RunOutcome:
    env = os.environ.copy()
    env["PREPARE_HF_CACHE"] = prepare_hf_cache
    cmd = [
        "sbatch",
        "--wait",
        "--parsable",
        f"--job-name={row['run_name'][:128]}",
        f"--export=ALL,PREPARE_HF_CACHE={prepare_hf_cache},RUN_SPEC_PATH={row['run_spec_path']}",
        str(RUNNER_SCRIPT),
    ]
    completed = _run(cmd, env=env)
    stdout = completed.stdout.strip()
    job_id = stdout.split(";")[0].strip() if stdout else None
    return RunOutcome(
        run_name=row["run_name"],
        family=row["family"],
        student_key=row["student_key"],
        layer_scheme=row["layer_scheme"],
        staged_training_enabled=bool(row["staged_training_enabled"]),
        run_spec_path=row["run_spec_path"],
        slurm_job_id=job_id or None,
        returncode=completed.returncode,
        success=completed.returncode == 0,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _aggregate_and_plot(manifest_dir: Path, output_dir: Path) -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str]]:
    aggregate = _run(
        [
            sys.executable,
            str(AGGREGATE_SCRIPT),
            "--manifest-dir",
            str(manifest_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    if aggregate.returncode != 0:
        return aggregate, subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="plot skipped because aggregation failed")

    plot = _run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--input-dir",
            str(output_dir),
            "--output-dir",
            str(output_dir / "figures"),
        ]
    )
    return aggregate, plot


def _write_report(report_dir: Path, manifest_rows: list[dict[str, Any]], outcomes: list[RunOutcome], aggregate: subprocess.CompletedProcess[str], plot: subprocess.CompletedProcess[str], aggregate_output_dir: Path) -> tuple[Path, Path]:
    report_json = report_dir / "mini_validation_report.json"
    report_md = report_dir / "mini_validation_report.md"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_run_count": len(manifest_rows),
        "manifest_runs": manifest_rows,
        "run_outcomes": [asdict(outcome) for outcome in outcomes],
        "aggregate": {
            "returncode": aggregate.returncode,
            "stdout": aggregate.stdout,
            "stderr": aggregate.stderr,
        },
        "plot": {
            "returncode": plot.returncode,
            "stdout": plot.stdout,
            "stderr": plot.stderr,
        },
        "aggregate_output_dir": str(aggregate_output_dir),
        "passed": all(outcome.success for outcome in outcomes) and aggregate.returncode == 0 and plot.returncode == 0,
    }
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Mini Launcher Validation Report",
        "",
        f"Overall result: {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        f"Manifest run count: {len(manifest_rows)}",
        "",
        "## Sequential Runs",
        "",
    ]
    for outcome in outcomes:
        lines.append(
            f"- {outcome.run_name}: success={outcome.success}, job_id={outcome.slurm_job_id}, "
            f"family={outcome.family}, student={outcome.student_key}, staged={outcome.staged_training_enabled}"
        )
    lines.extend(
        [
            "",
            "## Aggregation",
            "",
            f"- returncode: `{aggregate.returncode}`",
            f"- output_dir: `{aggregate_output_dir}`",
        ]
    )
    if aggregate.stdout.strip():
        lines.extend(["", "```text", aggregate.stdout.strip(), "```"])
    if aggregate.stderr.strip():
        lines.extend(["", "```text", aggregate.stderr.strip(), "```"])

    lines.extend(
        [
            "",
            "## Plotting",
            "",
            f"- returncode: `{plot.returncode}`",
        ]
    )
    if plot.stdout.strip():
        lines.extend(["", "```text", plot.stdout.strip(), "```"])
    if plot.stderr.strip():
        lines.extend(["", "```text", plot.stderr.strip(), "```"])

    report_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_json, report_md


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_dir = REPO_ROOT / "results" / "mini_launch_validation" / timestamp
    manifest_dir = base_dir / "manifest"
    aggregate_output_dir = base_dir / "aggregated"
    report_dir = base_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = _materialize_manifest(manifest_dir)
    manifest_rows = _load_manifest(manifest_path)

    outcomes: list[RunOutcome] = []
    for row in manifest_rows:
        outcome = _submit_run(row)
        outcomes.append(outcome)
        if not outcome.success:
            break

    if all(outcome.success for outcome in outcomes) and len(outcomes) == len(manifest_rows):
        aggregate, plot = _aggregate_and_plot(manifest_dir, aggregate_output_dir)
    else:
        aggregate = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="aggregation skipped because at least one submitted run failed")
        plot = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="plot skipped because aggregation did not run")

    report_json, report_md = _write_report(
        report_dir,
        manifest_rows,
        outcomes,
        aggregate,
        plot,
        aggregate_output_dir,
    )
    print(f"Wrote JSON report: {report_json}")
    print(f"Wrote Markdown report: {report_md}")
    print(f"Manifest dir: {manifest_dir}")
    print(f"Aggregate output dir: {aggregate_output_dir}")
    print(f"Submitted sequential runs: {len(outcomes)}/{len(manifest_rows)}")
    if all(outcome.success for outcome in outcomes) and len(outcomes) == len(manifest_rows):
        print("Sequential run stage: PASS")
    else:
        print("Sequential run stage: FAIL")
    print(f"Aggregation stage: {'PASS' if aggregate.returncode == 0 else 'FAIL'}")
    print(f"Plotting stage: {'PASS' if plot.returncode == 0 else 'FAIL'}")
    return 0 if all(outcome.success for outcome in outcomes) and len(outcomes) == len(manifest_rows) and aggregate.returncode == 0 and plot.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
