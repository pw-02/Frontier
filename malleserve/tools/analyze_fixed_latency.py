#!/usr/bin/env python3
"""Fixed-capacity Frontier latency diagnostic (no simulator modifications).

Join exact scheduled arrivals in the Mooncake trace to native Frontier events.
Produces per-request, output-length-bin, arrival-window CSVs, JSON diagnostics
and compact PNG/PDF figures. Uses event timestamps in SECONDS; do not mix
Frontier request_metrics.csv values, which are in milliseconds.

Run:
 python -m malleserve.tools.analyze_fixed_latency \
   --run outputs/malleserve/runs/fixed4_r10 \
   --trace data/malleserve/mooncake_r10_330s.csv \
   --output-prefix outputs/malleserve/figures/fixed4_r10_diagnostic
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from malleserve.tools.analyze_mooncake import percentile

BINS = [(1, 7, "1-7"), (8, 31, "8-31"), (32, 63, "32-63"),
        (64, 127, "64-127"), (128, math.inf, "128+")]
NAVY, TEAL, ORANGE, GREY = "#223C61", "#19A398", "#C26D20", "#607086"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def time_key(value):
    # Mooncake's fixed-rate arrivals are serialized to 9 decimal places.
    # Normalize to microseconds and reject collisions rather than guessing IDs.
    return round(float(value) * 1_000_000)


def sample_stats(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    return {
        "n": len(values),
        "mean_s": mean(values) if values else None,
        "p50_s": percentile(values, .5),
        "p95_s": percentile(values, .95),
        "p99_s": percentile(values, .99),
    }


def token_bin(n):
    for low, high, label in BINS:
        if low <= n <= high:
            return label
    raise ValueError(f"Invalid output-token length {n}")


def load(run, trace, start, end):
    source = read_csv(trace)
    scheduled = {}
    for item in source:
        key = time_key(item["arrived_at"])
        if key in scheduled:
            raise ValueError("Mooncake arrival timestamp collision at microsecond precision")
        scheduled[key] = item

    arrivals, completions = {}, {}
    with (run / "metrics_ground_truth.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get("event_type")
            rid = str(event.get("request_id"))
            if kind == "request_arrival":
                if rid in arrivals:
                    raise ValueError(f"Duplicate request arrival event {rid}")
                arrivals[rid] = float(event["arrived_at"])
            elif kind == "request_completion":
                if rid in completions:
                    raise ValueError(f"Duplicate request completion event {rid}")
                completions[rid] = event
    if len(arrivals) != len(source):
        raise ValueError(f"Trace contains {len(source)} requests, events contain "
                         f"{len(arrivals)} arrivals. Do not infer token lengths.")
    mapped = set()
    records = []
    missing = 0
    unavailable_wait = unavailable_ttft = 0
    for rid, arrived in arrivals.items():
        key = time_key(arrived)
        if key not in scheduled:
            raise ValueError(f"Unmatched event arrival {arrived}: no matching trace timestamp")
        if key in mapped:
            raise ValueError(f"Two event requests match the same trace timestamp: {arrived}")
        mapped.add(key)
        item = scheduled[key]
        if not (start <= arrived < end):
            continue
        ev = completions.get(rid)
        if ev is None:
            missing += 1
            continue
        completed = float(ev["completed_at"])
        latency = completed - arrived
        if latency < -1e-7 or not math.isfinite(latency):
            raise ValueError(f"Invalid latency for {rid}")
        wait = ev.get("request_waiting_time_total_s")
        ttft = ev.get("ttft_s")
        if wait is None:
            unavailable_wait += 1
        if ttft is None:
            unavailable_ttft += 1
        wait = float(wait) if wait is not None else None
        ttft = float(ttft) if ttft is not None else None
        if (wait is not None and (wait < -1e-7 or not math.isfinite(wait))) or (
                ttft is not None and (ttft < -1e-7 or not math.isfinite(ttft))):
            raise ValueError(f"Invalid wait/TTFT for {rid}")
        n_out = int(item["num_decode_tokens"])
        records.append({
            "request_id": rid, "arrival_s": arrived, "completion_s": completed,
            "latency_s": latency, "waiting_total_s": wait, "ttft_s": ttft,
            "post_first_token_s": latency - ttft if ttft is not None else None,
            "nonwaiting_elapsed_s": latency - wait if wait is not None else None,
            "prompt_tokens": int(item["num_prefill_tokens"]),
            "output_tokens": n_out, "output_bin": token_bin(n_out),
        })
    return records, {
        "total_trace_requests": len(source), "total_event_arrivals": len(arrivals),
        "measured_arrivals": sum(start <= x < end for x in arrivals.values()),
        "measured_completed": len(records), "measured_missing_completions": missing,
        "missing_waiting_field": unavailable_wait, "missing_ttft_field": unavailable_ttft,
        "join": "request event arrival timestamp matched to scheduled trace arrival within 0.5 microseconds",
        "units": "seconds for timing; request_metrics.csv is NOT used",
    }


def grouped(records, window, start, end):
    output_bins = []
    for _, _, label in BINS:
        subset = [r for r in records if r["output_bin"] == label]
        if not subset:
            continue
        output_bins.append({
            "output_bin": label,
            "n": len(subset),
            **{k + "_" + suffix: stat
               for k, field in (("latency", "latency_s"), ("waiting", "waiting_total_s"),
                                ("ttft", "ttft_s"), ("post_first_token", "post_first_token_s"))
               for suffix, stat in sample_stats([r[field] for r in subset]).items()
               if suffix != "n"},
        })
    time_bins = []
    low = start
    while low < end - 1e-9:
        high = min(end, low + window)
        subset = [r for r in records if low <= r["arrival_s"] < high]
        time_bins.append({
            "start_s": low, "end_s": high, "completed": len(subset),
            **{k + "_" + suffix: stat
               for k, field in (("latency", "latency_s"), ("waiting", "waiting_total_s"),
                                ("ttft", "ttft_s"))
               for suffix, stat in sample_stats([r[field] for r in subset]).items()
               if suffix != "n"},
        })
        low = high
    return output_bins, time_bins


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=.11)


def plots(output_bins, windows, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    x = np.arange(len(output_bins))
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    fig.subplots_adjust(left=.12, right=.98, bottom=.19, top=.82)
    for name, key, color, marker in (
        ("End-to-end", "latency_mean_s", NAVY, "o"),
        ("Waiting", "waiting_mean_s", ORANGE, "s"),
        ("TTFT", "ttft_mean_s", TEAL, "^"),
    ):
        y = [r[key] if r[key] is not None else float("nan") for r in output_bins]
        ax.plot(x, y, marker=marker, label=name, color=color, linewidth=2, markersize=5)
    ax.set_xticks(x, [f'{r["output_bin"]}\n(n={r["n"]})' for r in output_bins])
    ax.set_xlabel("Generated output tokens")
    ax.set_ylabel("Mean time (s)")
    ax.set_title("Fixed capacity | latency vs output length", fontsize=11.5, color="#18253A",
                 fontweight="bold", pad=13)
    ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.grid(axis="y", color="#E8EDF3")
    ax.set_axisbelow(True)
    save(fig, prefix.with_name(prefix.name + "_by_output"))
    plt.close(fig)

    centers = [(r["start_s"] + r["end_s"]) / 2 for r in windows]
    widths = [(r["end_s"] - r["start_s"]) * .88 for r in windows]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True)
    fig.subplots_adjust(left=.10, right=.98, top=.91, bottom=.12, hspace=.32)
    for ax, key, color, label in (
        (axes[0], "latency_p95_s", NAVY, "P95 request latency (s)"),
        (axes[1], "waiting_p95_s", ORANGE, "P95 waiting time (s)"),
    ):
        heights = [r[key] if r[key] is not None else float("nan") for r in windows]
        ax.bar(centers, heights, width=widths, color=color, alpha=.88)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#E8EDF3")
        ax.set_axisbelow(True)
    axes[0].set_title("Fixed capacity | arrival-window latency and waiting",
                      fontsize=11.5, fontweight="bold", color="#18253A", pad=10)
    axes[1].set_xlabel("Arrival time (simulated seconds)")
    save(fig, prefix.with_name(prefix.name + "_windows"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--trace", type=Path, default=Path("data/malleserve/mooncake_r10_330s.csv"))
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--start", type=float, default=30.)
    p.add_argument("--end", type=float, default=330.)
    p.add_argument("--window-seconds", type=float, default=20.)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    if not (args.start < args.end and args.window_seconds > 0):
        p.error("Requires start < end and window-seconds > 0")
    records, diagnostics = load(args.run, args.trace, args.start, args.end)
    bins, windows = grouped(records, args.window_seconds, args.start, args.end)
    path = args.output_prefix
    write_rows(path.with_name(path.name + "_requests.csv"), records)
    write_rows(path.with_name(path.name + "_output_bins.csv"), bins)
    write_rows(path.with_name(path.name + "_windows.csv"), windows)
    diagnostics["overall"] = {
        name: sample_stats([r[field] for r in records])
        for name, field in (("latency", "latency_s"),
                            ("waiting", "waiting_total_s"), ("ttft", "ttft_s"),
                            ("post_first_token", "post_first_token_s"))
    }
    diagnostics["interpretation"] = (
        "TTFT and waiting overlap: do not sum their percentiles. "
        "Nonwaiting_elapsed_s is latency minus reported waiting, NOT isolated "
        "GPU compute. Post-first-token interval includes scheduling, queuing, "
        "decode and overhead. Comparing output bins suggests sensitivity but "
        "does not prove a decode model error or isolate its causal contribution."
    )
    summary_path = path.with_name(path.name + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    if not args.no_plots:
        plots(bins, windows, path)
    print(json.dumps(diagnostics, indent=2))
    print(f"Wrote output under prefix: {path}")


if __name__ == "__main__":
    main()
