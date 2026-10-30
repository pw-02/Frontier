"""Compare Frontier and Polaris latency and request accounting by arrival window.

Counts ALL arrivals first, then matches completion records. Incomplete requests
are reported, never silently excluded from the denominator. Timing origins:
Frontier arrival times minus manifest warmup_seconds; Polaris Unix times minus
the supplied schedule_started Unix epoch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from malleserve.tools.analyze_mooncake import percentile

WINDOWS = ((0, 180), (180, 240), (240, 300))
ALIGNED_WINDOWS = ((30, 180), (180, 240), (240, 299), (299, 330))


def frontier_records(events, warmup):
    arrivals, completions = {}, {}
    for event in events:
        kind = event.get("event_type")
        request_id = str(event.get("request_id"))
        if kind == "request_arrival":
            if request_id in arrivals:
                raise ValueError(f"Duplicate Frontier arrival: {request_id}")
            arrivals[request_id] = float(event["arrived_at"])
        elif kind == "request_completion":
            if request_id in completions:
                raise ValueError(f"Duplicate Frontier completion: {request_id}")
            completions[request_id] = float(event["completed_at"])
    unknown = set(completions) - set(arrivals)
    if unknown:
        raise ValueError(f"Frontier completions without arrivals: {len(unknown)}")
    records = []
    for request_id, arrived in arrivals.items():
        if arrived < warmup:
            continue
        finished = completions.get(request_id)
        if finished is not None and finished < arrived:
            raise ValueError(f"Completion before arrival: {request_id}")
        records.append({"request_id": request_id,
                        "arrival_s": arrived - warmup,
                        "latency_s": None if finished is None else finished - arrived,
                        "completed_at_s": None if finished is None else finished - warmup})
    return records


def polaris_records(requests, zero):
    records = []
    for row in requests:
        if row.get("phase") != "measured":
            continue
        arrived = float(row["started_at_unix"])
        finish = row.get("completed_at_unix")
        if finish is None:
            finish = row.get("finished_at_unix")
        if finish is not None and float(finish) < arrived:
            raise ValueError("Polaris completion before arrival")
        records.append({"request_id": str(row.get("request_id", row.get("id", ""))),
                        "arrival_s": arrived-zero,
                        "latency_s": None if finish is None else float(finish)-arrived,
                        "completed_at_s": None if finish is None else float(finish)-zero})
    return records


def window_rows(records, system, windows=WINDOWS):
    result = []
    for start, end in windows:
        selected = [r for r in records if start <= r["arrival_s"] < end]
        completed = [r for r in selected if r["latency_s"] is not None]
        values = [r["latency_s"] for r in completed]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Nonfinite or negative latency")
        outstanding = [r for r in selected if r["latency_s"] is None]
        completed_after_window = sum(r["completed_at_s"] >= end for r in completed)
        row = {
            "system": system,
            "arrival_start_s": start,
            "arrival_end_s": end,
            "arrived": len(selected),
            "completed_eventually": len(completed),
            "no_completion_event": len(outstanding),
            "completed_after_window_end": completed_after_window,
            "completion_fraction": len(completed) / len(selected) if selected else "",
            "n": len(values),  # backwards-compatible alias for completed-only sample
            "mean_s": mean(values) if values else "",
            "p50_s": percentile(values, .50),
            "p95_s": percentile(values, .95),
            "p99_s": percentile(values, .99),
            "latency_scope": "completed_requests_only",
        }
        result.append(row)
    return result


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-run", type=Path, required=True)
    parser.add_argument("--frontier-manifest", type=Path, required=True)
    parser.add_argument("--polaris-requests", type=Path, required=True)
    parser.add_argument("--polaris-zero-unix", type=float, required=True,
                        help="Polaris schedule_started epoch (not first measured request)")
    parser.add_argument("--aligned-windows", action="store_true",
                        help="Use matched 30-180s baseline and isolate 299-330s post-restore arrivals")
    parser.add_argument("--output", type=Path,
                        default=Path("one_reclaim_latency_by_arrival.csv"))
    args = parser.parse_args()
    manifest = json.loads(args.frontier_manifest.read_text(encoding="utf-8"))
    frontier = frontier_records(
        read_jsonl(args.frontier_run / "metrics_ground_truth.jsonl"),
        float(manifest["warmup_seconds"]))
    polaris = polaris_records(read_jsonl(args.polaris_requests), args.polaris_zero_unix)
    windows = ALIGNED_WINDOWS if args.aligned_windows else WINDOWS
    rows = window_rows(frontier, "frontier", windows) + window_rows(polaris, "polaris", windows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print("Completion-only percentiles can be biased when no_completion_event > 0.")
    print("Reclamation interrupt totals require separate lifecycle diagnostics.")


if __name__ == "__main__":
    main()
