#!/usr/bin/env python3
"""Publication-ready latency-by-output-length comparison for fixed4/r10.

Values are from the previously analysed Polaris and Frontier request logs.
Polaris and Frontier logs came from different real-run IDs; treat as
an exploratory comparison, not a paired run-level validation.

Usage:
    python plot_latency_by_output_length.py
    python plot_latency_by_output_length.py --output-prefix figures/fixed4_r10_bins
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("fixed4_r10_latency_by_output_length"))
    args = parser.parse_args()

    bins = ["1–7", "8–31", "32–63", "64–127", "128"]
    counts = np.array([815, 934, 222, 80, 949])
    polaris = np.array([0.434, 0.987, 1.824, 4.174, 5.812])
    frontier = np.array([0.504, 1.404, 2.509, 5.821, 7.931])
    relative_error = (frontier / polaris - 1) * 100

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(len(bins))
    width = 0.34
    navy, teal = "#193B63", "#11A6A0"
    bars1 = ax.bar(x - width/2, polaris, width, label="Polaris · measured",
                   color=navy, zorder=3)
    bars2 = ax.bar(x + width/2, frontier, width, label="Frontier · simulated",
                   color=teal, zorder=3)

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .10,
                    f"{bar.get_height():.2f}", ha="center", va="bottom",
                    fontsize=9, color="#2C3542")

    for i, err in enumerate(relative_error):
        top = max(polaris[i], frontier[i])
        ax.text(x[i], top + .55, f"+{err:.0f}%", ha="center", va="bottom",
                fontsize=9, color="#475569", fontweight="semibold")

    labels = [f"{label}\n(n={count:,})" for label, count in zip(bins, counts)]
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 9.4)
    ax.set_ylabel("Mean end-to-end request latency (seconds)")
    ax.set_xlabel("Generated output tokens")
    ax.yaxis.grid(True, linestyle="-", linewidth=.65, alpha=.16)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=10)
    ax.set_title("Mean latency increases with output length", loc="left", pad=22)
    fig.text(.125, .88, "Qwen3-8B  |  4 × A100 GPUs  |  10 requests/s  |  3,000 measured requests",
             ha="left", fontsize=10, color="#657386")
    fig.text(.125, .035,
             "Labels above groups: Frontier relative error. Polaris client and Frontier engine latency have different timing boundaries.\n"
             "Exploratory comparison: Polaris and Frontier values were extracted from separate run archives.",
             ha="left", fontsize=8.5, color="#667085")
    fig.subplots_adjust(left=.125, right=.97, top=.82, bottom=.21)

    path = args.output_prefix
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        target = path.with_suffix("." + ext)
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print(f"Saved {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
