#!/usr/bin/env python3
"""Publication-style one-reclaim Polaris vs Frontier validation plots.

Usage:
  python -m malleserve.tools.plot_one_reclaim_comparison \
    --csv outputs/malleserve/validation/one_reclaim_aligned.csv \
    --output outputs/malleserve/figures/one_reclaim_r10

Creates <output>_p95.png/.pdf and <output>_metrics.png/.pdf.
Only accepts arrival-time aligned windows with both systems present.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WINDOWS = ((30, 180), (180, 240), (240, 299), (299, 330))
METRICS = (("Mean", "mean_s"), ("P50", "p50_s"), ("P95", "p95_s"), ("P99", "p99_s"))
NAVY, TEAL, INK, MUTED = "#223C61", "#19A398", "#18253A", "#607086"


def read_comparison(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    indexed = {}
    for row in records:
        key = (row["system"].strip().lower(),
               int(row["arrival_start_s"]), int(row["arrival_end_s"]))
        if key in indexed:
            raise ValueError(f"Duplicate system/window: {key}")
        indexed[key] = row
    paired = []
    for start, end in WINDOWS:
        key = (start, end)
        if ("frontier", *key) not in indexed or ("polaris", *key) not in indexed:
            raise ValueError(f"Missing aligned window {key}; rerun with --aligned-windows")
        f, p = (indexed[(system, *key)] for system in ("frontier", "polaris"))
        for system, row in (("frontier", f), ("polaris", p)):
            if int(row["no_completion_event"]) != 0:
                raise ValueError(f"{system} has unfinished requests in {key}; latency percentiles are censored")
            if int(row["arrived"]) != int(row["completed_eventually"]):
                raise ValueError(f"{system} completion mismatch in {key}")
        paired.append((f, p))
    return paired


def styling():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def save(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.20)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.20)
    plt.close(fig)
    print(f"Saved {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")


def plot_p95(paired, output):
    labels = [f"{a}–{b}s" for a, b in WINDOWS]
    real = np.array([float(p["p95_s"]) for f, p in paired])
    sim = np.array([float(f["p95_s"]) for f, p in paired])
    nreal = [int(p["arrived"]) for f, p in paired]
    nsim = [int(f["arrived"]) for f, p in paired]
    errors = 100 * (sim / real - 1)
    fig, ax = plt.subplots(figsize=(11.8, 6.4), facecolor="white")
    fig.subplots_adjust(left=.09, right=.97, top=.75, bottom=.23)
    x, width = np.arange(len(labels)), .31
    bars = (
        ax.bar(x-width/2, real, width, color=NAVY, label="Polaris (measured)", zorder=3),
        ax.bar(x+width/2, sim, width, color=TEAL, label="Frontier (simulated)", zorder=3),
    )
    for group in bars:
        for bar in group:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=10, color=INK, fontweight="semibold")
    ax.set_ylim(0, max(real.max(),sim.max())*1.28)
    ax.set_xticks(x, labels, color=INK, fontweight="semibold")
    ax.set_ylabel("P95 request latency (seconds)", color=INK)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", length=0, colors=MUTED)
    ax.yaxis.grid(True, color="#E8EDF3", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DCE3EA")
    fig.text(.09,.93,"TRANSIENT CAPACITY VALIDATION",fontsize=10,color=TEAL,fontweight="bold")
    fig.text(.09,.865,"One reclaim  |  Polaris vs Frontier  |  10 RPS",fontsize=18,color=INK,fontweight="bold")
    ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(0,1.12),ncol=2)
    fig.text(.09,.135,"FRONTIER P95 RELATIVE ERROR",fontsize=9,color=MUTED,fontweight="bold")
    for i, error in enumerate(errors):
        fig.text(.33+.155*i,.135,f"{error:+.1f}%",fontsize=10,color=INK,fontweight="semibold")
    fig.text(.09,.073,"Arrivals per phase (Frontier / Polaris): "+
             "  ·  ".join(f"{a}/{b}" for a,b in zip(nsim,nreal)),fontsize=8.5,color=MUTED)
    fig.text(.09,.040,"Arrival-time bins; 4 transient GPUs reclaimed at 180s and ready again at 299s.",
             fontsize=8.5,color=MUTED)
    save(fig, output.with_name(output.name+"_p95"))


def plot_metrics(paired, output):
    labels = [f"{a}–{b}s" for a,b in WINDOWS]
    fig, axes = plt.subplots(2,2,figsize=(12,8.0),facecolor="white",sharex=True)
    fig.subplots_adjust(left=.09,right=.97,top=.84,bottom=.11,hspace=.30,wspace=.22)
    x=np.arange(len(labels))
    for ax, (name,key) in zip(axes.flat,METRICS):
        real=np.array([float(p[key]) for f,p in paired])
        sim=np.array([float(f[key]) for f,p in paired])
        ax.plot(x,real,"o-",color=NAVY,linewidth=2,markersize=6,label="Polaris")
        ax.plot(x,sim,"s-",color=TEAL,linewidth=2,markersize=6,label="Frontier")
        ax.set_title(name,color=INK,fontweight="semibold",loc="left")
        ax.set_ylabel("Latency (s)",color=MUTED)
        ax.set_xticks(x,labels)
        ax.tick_params(axis="x",rotation=0,length=0,labelsize=9)
        ax.tick_params(axis="y",length=0,colors=MUTED)
        ax.yaxis.grid(True,color="#E8EDF3",linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#DCE3EA")
        ax.set_ylim(bottom=0)
    fig.text(.09,.95,"TRANSIENT CAPACITY VALIDATION",fontsize=10,color=TEAL,fontweight="bold")
    fig.text(.09,.905,"Latency distribution by request arrival phase",fontsize=18,color=INK,fontweight="bold")
    handles, leglabels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,leglabels,loc="upper right",bbox_to_anchor=(.97,.915),ncol=2,frameon=False)
    fig.text(.09,.043,"Absolute experiment clock  ·  10 RPS  ·  restoration at 299s  ·  all requests completed",
             fontsize=9,color=MUTED)
    save(fig,output.with_name(output.name+"_metrics"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("outputs/malleserve/figures/one_reclaim_r10"))
    args=parser.parse_args()
    paired=read_comparison(args.csv)
    styling()
    plot_p95(paired,args.output)
    plot_metrics(paired,args.output)


if __name__=="__main__":
    main()
