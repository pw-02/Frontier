#!/usr/bin/env python3
"""Publication-style, chart-only Polaris vs Frontier latency comparison.

Usage:
  python plot_latency_comparison.py \
    --csv outputs/malleserve/validation/fixed_capacity_comparison.csv \
    --workers 4 --rps 10 --output outputs/malleserve/figures/fixed4_r10

Requires: pip install matplotlib
Creates representative <output>.png/.pdf and <output>_sweep_p95.png/.pdf
and <output>_sweep_error.png/.pdf. Excludes invalid Polaris loads by default.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
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
    valid = [row for row in matches if row.get("polaris_valid_load", "").lower() == "true"]
    if not valid:
        raise ValueError(f"No valid Polaris + Frontier comparison for {workers} GPUs / {rps} RPS in {csv_path}")
    return sorted(valid, key=lambda r: r.get("polaris_run_id", ""))[-1]


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

    fig, ax = plt.subplots(figsize=(7.0, 4.5), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.20)
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
    ax.set_ylim(0, ceiling * 1.20)
    ax.set_xticks(x, labels, fontsize=11, fontweight="semibold", color=ink)
    ax.set_ylabel("Request latency (seconds)", fontsize=11, color=ink, labelpad=10)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0, colors=muted)
    ax.yaxis.grid(True, linewidth=0.8, color="#E8EDF3")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DCE3EA")

    title = f"{int(row['workers'])} GPUs | {float(row['offered_rps']):g} RPS"
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=ink, pad=14)
    ax.legend(frameon=True)
    # ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.005),
    #           ncol=2, columnspacing=1.4, fontsize=9.3)
    # Keep numeric relative-error data accessible without a slide-style footer.
    print("Frontier relative error:", ", ".join(
        f"{label} {err:+.1f}%" for label, err in zip(labels, error)))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(f"Saved {output.with_suffix('.png')} and {output.with_suffix('.pdf')}")



def read_sweep(csv_path: Path, workers: list[int]) -> tuple[dict[int, list[dict]], dict]:
    """Select latest *valid*, complete paired record for each worker/RPS cell.

    Polaris-invalid rows and missing Frontier cells are excluded and counted;
    there is no fallback to invalid samples and no interpolation.
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in all_rows:
        try:
            worker, rate = int(row["workers"]), float(row["offered_rps"])
        except (ValueError, KeyError, TypeError):
            continue
        if worker in workers:
            grouped[(worker, rate)].append(row)
    result = {w: [] for w in workers}
    skipped = defaultdict(int)
    for (worker, rate), candidates in sorted(grouped.items()):
        chosen = None
        for row in sorted(candidates, key=lambda x: x.get("polaris_run_id", ""), reverse=True):
            if row.get("polaris_valid_load", "").lower() != "true":
                continue
            if row.get("frontier_found", "").lower() != "true":
                continue
            try:
                polaris = parse_value(row, "latency_p95_s")
                frontier = parse_value(row, "frontier_p95_engine_latency_s")
            except (ValueError, TypeError):
                continue
            if not (math.isfinite(polaris) and math.isfinite(frontier) and polaris > 0 and frontier >= 0):
                continue
            chosen = (polaris, frontier, row)
            break
        if chosen is None:
            skipped["no_valid_complete_pair"] += 1
            continue
        p, f, row = chosen
        result[worker].append({"rate": rate, "polaris": p, "frontier": f, "row": row})
    return result, dict(skipped)


def _save_both(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {path.with_suffix('.png')} and {path.with_suffix('.pdf')}")


def _theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def plot_sweep_p95(data: dict[int, list[dict]], output: Path):
    """Faceted P95 vs load: common scale, missing/invalid cells left blank."""
    active = [(w, rows) for w, rows in data.items() if rows]
    if not active:
        raise ValueError("No valid paired Polaris/Frontier cells available for sweep plot")
    _theme()
    navy, teal, ink, muted = "#223C61", "#19A398", "#18253A", "#607086"
    ncols = min(3, len(active))
    nrows = math.ceil(len(active) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.7*nrows), squeeze=False)
    fig.subplots_adjust(left=.08, right=.985, bottom=.19 if nrows==1 else .11,
                        top=.82 if nrows==1 else .88, hspace=.45, wspace=.27)
    max_p95 = max(v[k] for _, rows in active for v in rows for k in ("polaris", "frontier"))
    max_rate = max(v["rate"] for _, rows in active for v in rows)
    for ax, (worker, rows) in zip(axes.flat, active):
        x = np.array([r["rate"] for r in rows])
        p = np.array([r["polaris"] for r in rows])
        f = np.array([r["frontier"] for r in rows])
        # Do not connect across missing/invalid RPS cells.
        standard_step = 2.0
        for value, color, marker, label in ((p,navy,"o","Polaris (measured)"),
                                             (f,teal,"s","Frontier (simulated)")):
            ax.scatter(x,value,s=39,marker=marker,c=color,zorder=4)
            for j in range(len(x)-1):
                if x[j+1]-x[j] <= standard_step+1e-9:
                    ax.plot(x[j:j+2],value[j:j+2],color=color,lw=2.1,zorder=3)
        ax.set_title(f"{worker} GPU{'s' if worker != 1 else ''}",loc="left",
                     fontsize=12,fontweight="bold",color=ink,pad=8)
        ax.set_xlim(0,max_rate+1)
        ax.set_ylim(0,max_p95*1.13)
        ax.set_xticks(np.arange(2,int(max_rate)+1,2))
        ax.grid(axis="y",color="#E8EDF3",lw=.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#DCE3EA")
        ax.tick_params(length=0,colors=muted)
        ax.set_xlabel("Offered load (req/s)",color=ink)
        if ax is axes.flat[0] or (len(active)>ncols and list(axes.flat).index(ax)%ncols==0):
            ax.set_ylabel("P95 latency (s)",color=ink)
    for ax in list(axes.flat)[len(active):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color=navy,marker="o",lw=2,label="Polaris (measured)"),
                        Line2D([0],[0],color=teal,marker="s",lw=2,label="Frontier (simulated)")],
               loc="upper left",bbox_to_anchor=(.08,.995),frameon=False,ncol=2)
    _save_both(fig,output.with_name(output.name+"_sweep_p95"))


def plot_sweep_error(data: dict[int, list[dict]], output: Path):
    """Absolute percentage gap at each valid matched cell, for scientific context."""
    active = [(w,rows) for w,rows in data.items() if rows]
    if not active:
        return
    _theme()
    fig, ax = plt.subplots(figsize=(7.0,4.5))
    fig.subplots_adjust(left=.12,right=.98,top=.84,bottom=.17)
    palette = ["#223C61","#19A398","#C26D20","#7B66A4","#4B879D"]
    for i,(w,rows) in enumerate(active):
        x=[v["rate"] for v in rows]
        y=[100*(v["frontier"]/v["polaris"]-1) for v in rows]
        ax.scatter(x,y,color=palette[i%len(palette)],s=46,zorder=3)
        for j in range(len(x)-1):
            if x[j+1]-x[j] <= 2.000001:
                ax.plot(x[j:j+2],y[j:j+2],color=palette[i%len(palette)],lw=1.8)
        ax.plot([],[],color=palette[i%len(palette)],marker="o",label=f"{w} GPU{'s' if w!=1 else ''}")
    ax.axhline(0,color="#91A0B0",linestyle="--",lw=1)
    ax.set_xlabel("Offered load (req/s)")
    ax.set_ylabel("Frontier P95 relative difference (%)")
    ax.set_xticks(np.arange(2,int(max(v["rate"] for _,rows in active for v in rows))+1,2))
    ax.grid(axis="y",color="#E8EDF3")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DCE3EA")
    ax.legend(frameon=False,ncol=min(len(active),5),loc="upper left",bbox_to_anchor=(0,1.13))
    _save_both(fig,output.with_name(output.name+"_sweep_error"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("outputs/malleserve/validation/fixed_capacity_comparison.csv"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/malleserve/figures/fixed4_r10"))
    parser.add_argument("--sweep-workers", default="4,6,8",
                        help="GPU counts for sweep facets (e.g. 1,2,4,6,8)")
    parser.add_argument("--single-only", action="store_true", help="Only render representative comparison")
    parser.add_argument("--sweep-only", action="store_true", help="Only render sweep figures")
    args = parser.parse_args()
    if args.single_only and args.sweep_only:
        parser.error("--single-only and --sweep-only are mutually exclusive")
    workers = [int(x.strip()) for x in args.sweep_workers.split(",") if x.strip()]
    if not workers or len(set(workers)) != len(workers):
        parser.error("--sweep-workers must contain distinct GPU counts")
    if not args.sweep_only:
        create_figure(select_row(args.csv, args.workers, args.rps), args.output)
    if not args.single_only:
        dataset, skipped = read_sweep(args.csv, workers)
        print("Paired valid cells:", {k: len(v) for k,v in dataset.items()}, "excluded:", skipped)
        plot_sweep_p95(dataset,args.output)
        plot_sweep_error(dataset,args.output)


if __name__ == "__main__":
    main()
