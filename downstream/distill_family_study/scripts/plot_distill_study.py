#!/usr/bin/env python3
"""Generate publication-style plots for the within-family distillation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/publication_within_family"),
        help="Directory containing aggregated study CSV/JSON outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/publication_within_family/figures"),
        help="Directory where figure assets should be written.",
    )
    return parser.parse_args()


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.fontsize": 9,
        }
    )


def _load_rows(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "master_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregated master rows: {path}")
    return pd.read_csv(path)


def _family_order(df: pd.DataFrame) -> list[str]:
    preferred = ["pythia", "qwen", "t5"]
    present = [family for family in preferred if family in set(df["family"])]
    extras = sorted(set(df["family"]) - set(preferred))
    return present + extras


def _student_order(df: pd.DataFrame, family: str) -> list[str]:
    subset = df[df["family"] == family]
    return list(dict.fromkeys(subset["student_model"].tolist()))


def plot_tradeoff_by_family(df: pd.DataFrame, outdir: Path) -> None:
    families = _family_order(df)
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), squeeze=False)
    lambda_colors = {0.0: "#1f77b4", 0.5: "#ff7f0e", 1.0: "#2ca02c"}
    scheme_markers = {"penultimate_only": "o", "second_plus_penultimate": "s", "custom": "D"}

    for ax, family in zip(axes[0], families):
        subset = df[df["family"] == family].copy()
        for _, row in subset.iterrows():
            ax.scatter(
                row["align_mse"],
                row["test_perplexity"],
                s=120,
                color=lambda_colors.get(float(row["lambda_align"]), "#444444"),
                marker=scheme_markers.get(str(row["layer_scheme"]), "x"),
                alpha=0.9,
            )
        ax.set_title(f"{family.capitalize()} Tradeoff")
        ax.set_xlabel("Alignment MSE")
        ax.set_ylabel("Test Perplexity")

    lambda_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=9, label=f"lambda={lam}")
        for lam, color in lambda_colors.items()
    ]
    scheme_handles = [
        plt.Line2D([0], [0], marker=marker, color="#555555", linestyle="", markersize=8, label=scheme)
        for scheme, marker in scheme_markers.items()
    ]
    fig.legend(handles=lambda_handles + scheme_handles, loc="upper center", ncol=5)
    fig.suptitle("Alignment vs Language Modeling Tradeoff by Family")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outdir / "tradeoff_by_family.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_ablation(df: pd.DataFrame, outdir: Path) -> None:
    families = _family_order(df)
    fig, axes = plt.subplots(len(families), 1, figsize=(12, 4 * len(families)), squeeze=False)
    scheme_colors = {"penultimate_only": "#1565c0", "second_plus_penultimate": "#c62828"}

    for ax, family in zip(axes[:, 0], families):
        subset = df[df["family"] == family].copy()
        grouped = (
            subset.groupby(["layer_scheme", "lambda_align"], as_index=False)["test_perplexity"]
            .mean()
            .sort_values(["layer_scheme", "lambda_align"])
        )
        for scheme, scheme_df in grouped.groupby("layer_scheme"):
            ax.plot(
                scheme_df["lambda_align"],
                scheme_df["test_perplexity"],
                marker="o",
                linewidth=2,
                color=scheme_colors.get(str(scheme), "#444444"),
                label=str(scheme),
            )
        ax.set_title(f"{family.capitalize()} Lambda Ablation")
        ax.set_xlabel("lambda_align")
        ax.set_ylabel("Mean Test Perplexity")
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "lambda_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_layer_scheme_comparison(df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for family in _family_order(df):
        family_df = df[df["family"] == family]
        for scheme in sorted(family_df["layer_scheme"].unique()):
            scheme_df = family_df[family_df["layer_scheme"] == scheme]
            rows.append(
                {
                    "family": family,
                    "layer_scheme": scheme,
                    "test_perplexity_mean": float(scheme_df["test_perplexity"].mean()),
                }
            )

    plot_df = pd.DataFrame(rows)
    families = _family_order(plot_df)
    schemes = sorted(plot_df["layer_scheme"].unique())
    x = range(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, scheme in enumerate(schemes):
        scheme_df = plot_df[plot_df["layer_scheme"] == scheme].set_index("family").reindex(families)
        ax.bar(
            [pos + (idx - (len(schemes) - 1) / 2) * width for pos in x],
            scheme_df["test_perplexity_mean"],
            width=width,
            label=scheme,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([family.capitalize() for family in families])
    ax.set_ylabel("Mean Test Perplexity")
    ax.set_title("Layer-Scheme Comparison by Family")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "layer_scheme_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_scale_comparison(df: pd.DataFrame, outdir: Path) -> None:
    best_rows = []
    for family in _family_order(df):
        family_df = df[df["family"] == family]
        for student_model in _student_order(df, family):
            student_df = family_df[family_df["student_model"] == student_model].copy()
            best = student_df.sort_values(["val_loss", "align_mse"]).iloc[0]
            best_rows.append(best)

    plot_df = pd.DataFrame(best_rows)
    families = _family_order(plot_df)
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), squeeze=False)

    for ax, family in zip(axes[0], families):
        family_df = plot_df[plot_df["family"] == family].copy()
        family_df = family_df.sort_values("probe_size")
        ax.plot(
            family_df["probe_size"],
            family_df["val_loss"],
            marker="o",
            linewidth=2,
            color="#5d4037",
        )
        for _, row in family_df.iterrows():
            ax.annotate(
                str(row["student_model"]).split("/")[-1],
                (row["probe_size"], row["val_loss"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(f"{family.capitalize()} Best-Per-Student")
        ax.set_xlabel("Probe Size")
        ax.set_ylabel("Validation Loss")

    fig.tight_layout()
    fig.savefig(outdir / "scale_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, outdir: Path) -> None:
    families = _family_order(df)
    summary = {
        "num_rows": int(len(df)),
        "families": families,
        "figures": [
            "tradeoff_by_family.png",
            "lambda_ablation.png",
            "layer_scheme_comparison.png",
            "scale_comparison.png",
        ],
    }
    _best = []
    for family in families:
        family_df = df[df["family"] == family].copy()
        best = family_df.sort_values(["val_loss", "align_mse"]).iloc[0]
        _best.append(
            {
                "family": family,
                "student_model": best["student_model"],
                "layer_scheme": best["layer_scheme"],
                "lambda_align": float(best["lambda_align"]),
                "val_loss": float(best["val_loss"]),
                "test_perplexity": float(best["test_perplexity"]),
                "align_mse": float(best["align_mse"]),
            }
        )
    summary["best_by_family"] = _best
    (outdir / "plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_rows(args.input_dir)

    plot_tradeoff_by_family(df, args.output_dir)
    plot_lambda_ablation(df, args.output_dir)
    plot_layer_scheme_comparison(df, args.output_dir)
    plot_scale_comparison(df, args.output_dir)
    write_summary(df, args.output_dir)

    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
