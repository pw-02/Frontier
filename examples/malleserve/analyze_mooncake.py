#!/usr/bin/env python3
"""Summarize Frontier's Mooncake run using event-level request evidence.

Frontier metrics_ground_truth.jsonl contains request_arrival and
request_completion events with SIMULATED seconds. Request_metrics.csv exports
time distributions in MILLISECONDS; never mix those units.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    a = sorted(values)
    k = (len(a) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return a[lo] * (hi - k) + a[hi] * (k - lo)


def summary(events: list[dict], manifest: dict, slo_seconds: float) -> dict:
    if slo_seconds <= 0:
        raise ValueError("SLO must be positive")
    begin = float(manifest["warmup_seconds"])
    duration = float(manifest["measurement_seconds"])
    end = begin + duration
    expected_warmup = int(manifest["warmup_count"])
    expected_measured = int(manifest["measurement_count"])

    arrivals = {}
    completions = {}
    for event in events:
        typ = event.get("event_type")
        ident = str(event.get("request_id"))
        if typ == "request_arrival":
            if ident in arrivals:
                raise ValueError(f"Duplicate arrival event for request {ident}")
            arrivals[ident] = float(event["arrived_at"])
        elif typ == "request_completion":
            if ident in completions:
                raise ValueError(f"Duplicate completion event for request {ident}")
            completions[ident] = event

    expected_all = expected_warmup + expected_measured
    if len(arrivals) != expected_all:
        raise ValueError(
            f"Expected {expected_all} arrival events, got {len(arrivals)}. "
            "Check the run's event trace and simulation errors."
        )
    measured = {rid: a for rid, a in arrivals.items() if begin <= a < end}
    if len(measured) != expected_measured:
        raise ValueError(
            f"Expected {expected_measured} measured arrivals in [{begin}, {end}), "
            f"got {len(measured)}. Check Frontier timestamps/trace alignment."
        )
    completed_measured = {rid: completions[rid] for rid in measured if rid in completions}
    latencies, waits, ttfts = [], [], []
    on_time = in_window = late_drain = 0
    for rid, event in completed_measured.items():
        latency = float(event["completed_at"]) - measured[rid]
        if latency < -1e-8:
            raise ValueError(f"Completion precedes arrival for request {rid}")
        latencies.append(latency)
        waits.append(float(event.get("request_waiting_time_total_s", 0)))
        ttfts.append(float(event.get("ttft_s", 0)))
        if latency <= slo_seconds:
            on_time += 1
        if begin <= float(event["completed_at"]) < end:
            in_window += 1
        if float(event["completed_at"]) >= end:
            late_drain += 1

    all_finished_window = sum(
        begin <= float(event["completed_at"]) < end
        for event in completions.values()
    )
    backlog_end = sum(a < end for a in arrivals.values()) - sum(
        float(e["completed_at"]) < end for e in completions.values()
    )
    return {
        "simulated_only": True,
        "arrival_rate_offered_rps": float(manifest["rate_per_second"]),
        "measurement_window_s": [begin, end],
        "warmup_arrivals": expected_warmup,
        "measured_arrivals": len(measured),
        "measured_completed_eventually": len(completed_measured),
        "measured_no_completion_event": len(measured) - len(completed_measured),
        "measured_completed_inside_window": in_window,
        "measured_completions_during_drain": late_drain,
        "measured_still_outstanding_at_window_end": len(measured) - in_window,
        "all_completion_throughput_in_window_rps": all_finished_window / duration,
        "measured_completion_throughput_in_window_rps": in_window / duration,
        "slo_seconds": slo_seconds,
        "measured_slo_successes_allowing_drain": on_time,
        "measured_slo_attainment": on_time / len(measured) if measured else 0,
        "measured_slo_goodput_rps_allowing_drain": on_time / duration,
        "all_request_backlog_at_window_end": backlog_end,
        "latency_mean_s_for_completed_measured": mean(latencies) if latencies else None,
        "latency_p50_s_for_completed_measured": percentile(latencies, .50),
        "latency_p95_s_for_completed_measured": percentile(latencies, .95),
        "latency_p99_s_for_completed_measured": percentile(latencies, .99),
        "queue_wait_mean_s_for_completed_measured": mean(waits) if waits else None,
        "queue_wait_p95_s_for_completed_measured": percentile(waits, .95),
        "ttft_p95_s_for_completed_measured": percentile(ttfts, .95),
        "warning": (
            "Requests completed after the window count toward SLO attainment "
            "but not toward within-window completion throughput. Dummy "
            "execution times are not calibrated to Polaris."
        ),
    }


def load_events(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Frontier ground-truth events: {path}. "
            "Run with --metrics_config_enable_metrics_ground_truth_trace."
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_request_csv_metrics(path: Path) -> dict:
    """Optional descriptive metrics (exported in milliseconds)."""
    if not path.exists():
        return {"request_metrics_csv": "not found"}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    def numeric(key: str) -> list[float]:
        values = []
        for row in rows:
            try:
                value = float(row[key])
                if math.isfinite(value):
                    values.append(value)
            except (KeyError, ValueError, TypeError):
                pass
        return values
    cached = numeric("request_cached_prefill_tokens")
    prompt = numeric("request_num_prefill_tokens")
    delay_ms = numeric("request_first_scheduling_delay")
    return {
        "request_metrics_csv_rows": len(rows),
        "csv_cached_prefill_tokens_mean": mean(cached) if cached else None,
        "csv_input_tokens_mean": mean(prompt) if prompt else None,
        "csv_first_scheduling_delay_p95_ms": percentile(delay_ms, .95),
        "csv_time_units": "milliseconds; event trace time units are seconds",
        "csv_scope": "whole run (includes warmup); per-request CSV lacks arrival timestamps",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Converter's .manifest.json")
    p.add_argument("--metrics-dir", type=Path, required=True)
    p.add_argument("--slo-seconds", type=float, default=30.0)
    p.add_argument("--output", type=Path)
    a = p.parse_args()

    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    events = load_events(a.metrics_dir / "metrics_ground_truth.jsonl")
    data = summary(events, manifest, a.slo_seconds)
    data.update(read_request_csv_metrics(a.metrics_dir / "request_metrics.csv"))
    system_file = a.metrics_dir / "system_metrics.json"
    data["system_metrics_json"] = str(system_file) if system_file.exists() else "not found"
    dest = a.output or a.metrics_dir / "malleserve_summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print("Frontier Mooncake summary (SIMULATED; not Polaris-calibrated)")
    print(f"  Measured arrivals:     {data['measured_arrivals']}")
    print(f"  Eventual completions:  {data['measured_completed_eventually']}")
    print(f"  SLO attainment:        {data['measured_slo_attainment']:.1%} "
          f"(SLO={a.slo_seconds:g}s; includes drain)")
    print(f"  SLO goodput:           {data['measured_slo_goodput_rps_allowing_drain']:.3f} rps")
    print(f"  Completion throughput: {data['measured_completion_throughput_in_window_rps']:.3f} rps "
          "(measured arrivals completed inside window)")
    print(f"  Backlog at end:        {data['all_request_backlog_at_window_end']}")
    for label, key in (
        ("Mean latency", "latency_mean_s_for_completed_measured"),
        ("P95 latency", "latency_p95_s_for_completed_measured"),
        ("P95 queue wait", "queue_wait_p95_s_for_completed_measured"),
        ("P95 TTFT", "ttft_p95_s_for_completed_measured"),
    ):
        val = data[key]
        print(f"  {label}: {val:.3f}s" if val is not None else f"  {label}: n/a")
    print(f"  Full report: {dest}")


if __name__ == "__main__":
    main()
