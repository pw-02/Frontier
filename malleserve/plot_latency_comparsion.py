#!/usr/bin/env python3
"""Publication-ready Polaris vs Frontier latency comparison.

Usage:
  python plot_latency_comparison.py \
    --csv outputs/malleserve/validation/fixed_capacity_comparison.csv \
    --workers 4 --rps 10 --output outputs/malleserve/figures/fixed4_r10

Requires: pip install matplotlib
Creates both <output>.png (300 dpi) and <output>.pdf (vector).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

METRICS = [
    ("Mean", "polaris_latency_mean_s", "frontier_mean_engine_latency_s"),
    ("P50", "polaris_latency_p50_s", "frontier_p50_engine_latency_s"),
    ("P95", "latency_p95_s", "frontier_p95_engine_latency_s"),
    ("P99", "polaris_latency_p99_s", "frontier_p99_engine_latency_s"),
]


def parse_value(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw in (None, ""):
        raise ValueError(f"Missing {key!r}. Ensure the Frontier simulation exists for this cell.")
    return float(raw)


def select_row(csv_path: Path, workers: int, rps: int) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        matches = [row for row in csv.DictReader(handle)
                   if int(row["workers"]) == workers
                   and float(row["offered_rps"]) == rps
                   and row.get("frontier_found", "").lower() == "true"]
    if not matches:
        raise ValueError(f"No completed comparison for {workers} GPUs / {rps} RPS in {csv_path}")
    # Polaris can contain repeated measurements of the same cell.
    # Choose latest valid-load measurement, falling back to latest recorded run.
    valid = [row for row in matches if row.get("polaris_valid_load", "").lower() == "true"]
    return sorted(valid or matches, key=lambda r: r.get("polaris_run_id", ""))[-1]


def create_figure(row: dict[str, str], output: Path) -> None:
    labels = [item[0] for item in METRICS]
    real = np.array([parse_value(row, item[1]) for item in METRICS])
    simulated = np.array([parse_value(row, item[2]) for item in METRICS])
    error = 100 * (simulated / real - 1)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    navy = "#223C61"
    teal = "#19A398"
    ink = "#18253A"
    muted = "#607086"

    fig, ax = plt.subplots(figsize=(10.5, 5.9), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.77, bottom=0.22)
    x = np.arange(len(labels))
    width = 0.31
    bars_real = ax.bar(x - width / 2, real, width, color=navy, label="Polaris (measured)", zorder=3)
    bars_sim = ax.bar(x + width / 2, simulated, width, color=teal, label="Frontier (simulated)", zorder=3)

    for bars in (bars_real, bars_sim):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", (bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=10, fontweight="semibold", color=ink)

    ceiling = max(real.max(), simulated.max())
    ax.set_ylim(0, ceiling * 1.29)
    ax.set_xticks(x, labels, fontsize=11, fontweight="semibold", color=ink)
    ax.set_ylabel("Request latency (seconds)", fontsize=11, color=ink, labelpad=10)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0, colors=muted)
    ax.yaxis.grid(True, linewidth=0.8, color="#E8EDF3")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DCE3EA")

    fig.text(0.10, 0.92, "LATENCY VALIDATION", fontsize=10, color=teal,
             fontweight="bold", ha="left")
    fig.text(0.10, 0.855, f"Polaris vs Frontier  |  {int(row['workers'])} GPUs · {float(row['offered_rps']):g} RPS",
             fontsize=18, fontweight="bold", color=ink, ha="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.10),
              ncol=2, columnspacing=2.5, fontsize=10)

    # Error annotations in a separate footer keep the bars uncluttered.
    fig.text(0.10, 0.12, "FRONTIER RELATIVE ERROR", fontsize=9, color=muted,
             fontweight="bold", ha="left")
    for i, (label, err) in enumerate(zip(labels, error)):
        fig.text(0.35 + 0.145 * i, 0.12, f"{label}: {err:+.1f}%", fontsize=10,
                 fontweight="semibold", color=ink, ha="left")
    fig.text(0.10, 0.055,
             "Polaris: HTTP client end-to-end latency   ·   Frontier: simulated request completion latency",
             color=muted, fontsize=9, ha="left")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(f"Saved {output.with_suffix('.png')} and {output.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("outputs/malleserve/validation/fixed_capacity_comparison.csv"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/malleserve/figures/fixed4_r10"))
    args = parser.parse_args()
    create_figure(select_row(args.csv, args.workers, args.rps), args.output)


if __name__ == "__main__":
    main()
