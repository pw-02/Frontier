#!/usr/bin/env python3
"""Plot matched Frontier vs Polaris fixed-capacity results.

Requires: pip install pandas matplotlib

Usage (from Frontier root):
  python validation/plot_frontier_polaris.py \
      --input validation/frontier_vs_polaris.csv \
      --out validation/plots

Only rows marked comparable_traffic=True AND frontier_summary_found=True are
used for accuracy/ranking plots. This prevents missing traffic and HTTP failures
from being interpreted as valid simulation error measurements.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No display server needed (Docker, WSL, Windows).
import matplotlib.pyplot as plt
import pandas as pd

REQUIRED = {
    "workers", "offered_rps", "comparable_traffic", "frontier_summary_found",
    "polaris_p95_s", "frontier_p95_s", "polaris_p99_s", "frontier_p99_s",
    "polaris_slo_attainment_pct", "frontier_slo_attainment_pct",
    "polaris_slo_goodput_rps", "frontier_slo_goodput_rps",
}
NUMERIC = [
    "workers", "offered_rps", "polaris_p95_s", "frontier_p95_s",
    "polaris_p99_s", "frontier_p99_s", "polaris_slo_attainment_pct",
    "frontier_slo_attainment_pct", "polaris_slo_goodput_rps",
    "frontier_slo_goodput_rps",
]


def as_bool(col: pd.Series) -> pd.Series:
    return col.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(path)
    missing = REQUIRED.difference(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["is_comparable"] = as_bool(data["comparable_traffic"]) & as_bool(data["frontier_summary_found"])
    data.loc[data[["polaris_p95_s", "frontier_p95_s"]].isna().any(axis=1), "is_comparable"] = False
    valid = data[data["is_comparable"]].copy()
    if valid.empty:
        raise ValueError("No valid comparable rows found")
    duplicates = valid.duplicated(["workers", "offered_rps"], keep=False)
    if duplicates.any():
        cells = valid.loc[duplicates, ["workers", "offered_rps"]].drop_duplicates()
        raise ValueError(
            "Multiple comparable repetitions in the same (workers, rate) cell. "
            "Aggregate or select the repetitions first:\n" + cells.to_string(index=False)
        )
    valid["p95_error_pct"] = 100 * (valid["frontier_p95_s"] / valid["polaris_p95_s"] - 1)
    return data, valid.sort_values(["workers", "offered_rps"])


def save(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(outdir / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png/.pdf")


def axes(title: str, xlabel: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_p95_by_rate(valid: pd.DataFrame, out: Path) -> None:
    for workers, group in valid.groupby("workers", sort=True):
        group = group.sort_values("offered_rps")
        fig, ax = axes(
            f"p95 latency vs offered load — {int(workers)} workers",
            "Offered request rate (requests/s)", "p95 end-to-end latency (seconds)",
        )
        ax.plot(group["offered_rps"], group["polaris_p95_s"], marker="o", linewidth=2, label="Polaris (real)")
        ax.plot(group["offered_rps"], group["frontier_p95_s"], marker="s", linestyle="--", linewidth=2, label="Frontier (simulated)")
        ax.set_xticks(sorted(group["offered_rps"].unique()))
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.text(0, -0.21, "Comparable runs only; incomplete/failed Polaris traffic excluded.",
                transform=ax.transAxes, fontsize=9)
        save(fig, out, f"p95_vs_rate_w{int(workers)}")


def plot_parity(valid: pd.DataFrame, out: Path) -> None:
    fig, ax = axes("p95 parity — matched valid experiments", "Polaris p95 (seconds)", "Frontier p95 (seconds)")
    for workers, group in valid.groupby("workers", sort=True):
        ax.scatter(group["polaris_p95_s"], group["frontier_p95_s"],
                   label=f"{int(workers)} worker(s)", s=55, alpha=0.85)
    maximum = max(valid["polaris_p95_s"].max(), valid["frontier_p95_s"].max()) * 1.08
    ax.plot([0, maximum], [0, maximum], linestyle=":", label="Perfect agreement")
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, maximum)
    ax.set_aspect("equal", adjustable="box")
    rho = valid["polaris_p95_s"].rank().corr(valid["frontier_p95_s"].rank())
    median_error = valid["p95_error_pct"].median()
    ax.text(0.04, 0.96, f"n = {len(valid)}\nSpearman ρ = {rho:.3f}\nMedian error = {median_error:+.1f}%",
            transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    ax.legend(loc="lower right")
    save(fig, out, "p95_parity")


def plot_error_by_rate(valid: pd.DataFrame, out: Path) -> None:
    fig, ax = axes("Frontier p95 relative error", "Offered request rate (requests/s)", "(Frontier / Polaris − 1) × 100 (%)")
    for workers, group in valid.groupby("workers", sort=True):
        group = group.sort_values("offered_rps")
        ax.plot(group["offered_rps"], group["p95_error_pct"], marker="o", label=f"{int(workers)} worker(s)")
    ax.axhline(0, linestyle=":", linewidth=1.3)
    ax.legend(title="Comparable runs only")
    save(fig, out, "p95_relative_error")


def plot_scaling(valid: pd.DataFrame, out: Path, rate: float) -> None:
    group = valid[valid["offered_rps"].eq(rate)].sort_values("workers")
    if len(group) < 2:
        print(f"Skipping scaling plot for {rate:g} rps: fewer than 2 comparable worker counts")
        return
    fig, ax = axes(f"Scale-out p95 latency at {rate:g} requests/s", "Number of workers (one A100 each)", "p95 end-to-end latency (seconds)")
    ax.plot(group["workers"], group["polaris_p95_s"], marker="o", linewidth=2, label="Polaris (real)")
    ax.plot(group["workers"], group["frontier_p95_s"], marker="s", linestyle="--", linewidth=2, label="Frontier (simulated)")
    ax.set_xticks(group["workers"])
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, out, f"p95_scaleout_r{rate:g}")


def plot_slo_diagnostic(all_data: pd.DataFrame, out: Path) -> None:
    # Show overloaded cells *explicitly* as non-comparable diagnostic points.
    # Their observed real goodput is invalid for a strict capacity comparison
    # when k6 dropped submissions / HTTP requests failed.
    for workers in (1, 2):
        subset = all_data[all_data["workers"].eq(workers)].sort_values("offered_rps")
        if subset.empty:
            continue
        fig, ax = axes(
            f"SLO attainment — {workers} workers (diagnostic)",
            "Offered request rate (requests/s)", "Requests meeting 30 s SLO (%)",
        )
        valid = subset[subset["is_comparable"]]
        invalid = subset[~subset["is_comparable"]]
        ax.plot(subset["offered_rps"], subset["polaris_slo_attainment_pct"], marker="o", label="Polaris observed")
        ax.plot(subset["offered_rps"], subset["frontier_slo_attainment_pct"], marker="s", linestyle="--", label="Frontier simulated")
        if not invalid.empty:
            ax.scatter(invalid["offered_rps"], invalid["polaris_slo_attainment_pct"],
                       marker="x", s=140, linewidths=2, label="Polaris: incomplete/failed traffic")
        ax.set_ylim(-4, 105)
        ax.set_xticks(sorted(subset["offered_rps"].unique()))
        ax.legend()
        ax.text(0, -0.25, "X markers: not comparable; do not infer a precise capacity boundary.",
                transform=ax.transAxes, fontsize=9)
        save(fig, out, f"slo_attainment_diagnostic_w{workers}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("validation/frontier_vs_polaris.csv"))
    parser.add_argument("--out", type=Path, default=Path("validation/plots"))
    parser.add_argument("--scaling-rate", type=float, default=10.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data, valid = load_data(args.input)
    print(f"Loaded {len(data)} rows, {len(valid)} comparable; {len(data)-len(valid)} excluded from accuracy plots")
    plot_p95_by_rate(valid, args.out)
    plot_parity(valid, args.out)
    plot_error_by_rate(valid, args.out)
    plot_scaling(valid, args.out, args.scaling_rate)
    plot_slo_diagnostic(data, args.out)
    rho = valid["polaris_p95_s"].rank().corr(valid["frontier_p95_s"].rank())
    summary = (
        f"Comparable experiments: {len(valid)}/{len(data)}\n"
        f"Median p95 relative error: {valid['p95_error_pct'].median():+.2f}%\n"
        f"Mean p95 relative error: {valid['p95_error_pct'].mean():+.2f}%\n"
        f"Spearman rank correlation (p95): {rho:.4f}\n"
        f"Note: comparability is based on columns in the input CSV.\n"
        f"Note: SLO diagnostic includes incomplete/failed Polaris runs, marked X.\n"
    )
    (args.out / "validation_stats.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Figures saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
