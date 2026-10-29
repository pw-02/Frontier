#!/usr/bin/env python3
"""Plot the transient proxy's per-window dynamics, without hiding failures."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

MODES = ("fixed_persistent", "fixed_stable", "naive", "ideal")


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def plot_scenario(path: Path, bin_seconds: float = 10.0) -> None:
    import matplotlib.pyplot as plt

    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")
    summaries = {
        mode: json.loads((path / mode / "summary.json").read_text())
        for mode in MODES
    }
    begin, end = summaries["naive"]["measurement_window_s"]
    x_bins = list(range(int(math.ceil((end - begin) / bin_seconds))))
    x = [begin + (i + 0.5) * bin_seconds for i in x_bins]
    series = {}
    for mode in MODES:
        records = load_csv(path / mode / "requests.csv")
        latency_by_bin: dict[int, list[float]] = defaultdict(list)
        success_by_bin: dict[int, int] = defaultdict(int)
        arrivals_by_bin: dict[int, int] = defaultdict(int)
        for row in records:
            arrival = float(row["arrival_s"])
            if not begin <= arrival < end:
                continue
            arrival_bin = int((arrival - begin) / bin_seconds)
            arrivals_by_bin[arrival_bin] += 1
            if not row["completion_s"]:
                continue
            completion = float(row["completion_s"])
            latency = completion - arrival
            latency_by_bin[arrival_bin].append(latency)
            if latency <= float(summaries[mode]["slo_seconds"]):
                success_by_bin[arrival_bin] += 1
        series[mode] = (latency_by_bin, success_by_bin, arrivals_by_bin)

    path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for mode in MODES:
        latencies, _, _ = series[mode]
        values = [percentile(latencies[i], 0.95)
                  if i in latencies else float("nan") for i in x_bins]
        ax.plot(x, values, marker=".", label=mode)
    ax.axhline(summaries["naive"]["slo_seconds"], linestyle="--",
               color="grey", label="SLO")
    ax.set(xlabel="Request arrival time (s)", ylabel="p95 latency (s)",
           title="p95 by arrival cohort — exploratory macro model")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "latency_by_arrival_window.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for mode in MODES:
        _, ok, arrivals = series[mode]
        # SLO-goodput is counted by ARRIVAL cohort (allowing drain);
        # a failed/uncompleted request contributes zero.
        values = [ok[i] / bin_seconds for i in x_bins]
        ax.plot(x, values, marker=".", label=mode)
    ax.set(xlabel="Request arrival time (s)",
           ylabel=f"SLO-qualified requests per second (arrival cohort, {bin_seconds:g}s bins)",
           title="SLO goodput by arrival cohort — exploratory macro model")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "slo_goodput_by_arrival_window.png", dpi=160)
    plt.close(fig)

    names = list(MODES)
    values = [summaries[mode]["slo_goodput_rps"] for mode in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, values)
    ax.set(ylabel="SLO goodput (requests/s)",
           title="Identical arrivals; different worker availability policies")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(path / "slo_goodput_comparison.png", dpi=160)
    plt.close(fig)
    print(f"Saved three exploratory plots to {path}")


def plot_sweep(path: Path) -> None:
    import matplotlib.pyplot as plt

    records = load_csv(path / "sweep_comparison.csv")
    if not records:
        raise ValueError("No sweep data")
    for lifetime in sorted({float(r["lifetime_s"]) for r in records}):
        fig, ax = plt.subplots(figsize=(9, 5))
        for mode in MODES:
            sub = sorted(
                (r for r in records if r["mode"] == mode
                 and float(r["lifetime_s"]) == lifetime),
                key=lambda r: float(r["offered_rps"]),
            )
            ax.plot([float(r["offered_rps"]) for r in sub],
                    [float(r["slo_goodput_rps"]) for r in sub],
                    marker="o", label=mode)
        ax.set(xlabel="Offered request rate (rps)", ylabel="SLO goodput (rps)",
               title=f"Macro feasibility: transient lifetime {lifetime:g}s")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        filename = f"slo_vs_rate_lifetime_{lifetime:g}.png".replace(".", "p")
        # Keep .png extension after replacing decimal punctuation
        filename = filename.replace("ppng", ".png")
        fig.savefig(path / filename, dpi=160)
        plt.close(fig)
    print(f"Saved lifetime sweep plots under {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=Path)
    group.add_argument("--sweep", type=Path)
    p.add_argument("--time-bin-seconds", type=float, default=10)
    args = p.parse_args()
    if args.scenario:
        plot_scenario(args.scenario, args.time_bin_seconds)
    else:
        plot_sweep(args.sweep)


if __name__ == "__main__":
    main()
