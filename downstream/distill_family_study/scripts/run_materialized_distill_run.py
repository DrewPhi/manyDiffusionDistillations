#!/usr/bin/env python3
"""Execute one materialized distillation study run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-spec",
        type=Path,
        required=True,
        help="Path to one JSON run spec emitted by materialize_distill_study.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_spec = json.loads(args.run_spec.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[3]
    downstream_config_dir = repo_root / "downstream" / "distill_family_study" / "configs"

    command = [
        "python",
        "-m",
        "manylatents.main",
        f"hydra.searchpath=[file://{downstream_config_dir}]",
        *run_spec["hydra_overrides"],
    ]
    print(f"Executing run: {run_spec['run_name']}")
    print(f"Run spec: {args.run_spec}")
    print("Command:")
    print(" ".join(command))

    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
