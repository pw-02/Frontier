"""Compare Frontier and Polaris request latency by arrival-phase bins.

Polaris source: transient_capacity runs' data/requests.jsonl; timestamp
origin comes from workload_start/measured phase event in data/events.jsonl.
Frontier source: metrics_ground_truth.jsonl, with manifest warmup offset.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from malleserve.tools.analyze_mooncake import percentile


WINDOWS = ((0,180),(180,240),(240,300))


def frontier_points(events, warmup):
    arrivals = {}
    done = {}
    for ev in events:
        ident = str(ev.get("request_id"))
        if ev.get("event_type") == "request_arrival":
            arrivals[ident] = float(ev["arrived_at"])
        elif ev.get("event_type") == "request_completion":
            done[ident] = float(ev["completed_at"])
    return [(arrival-warmup,done[ident]-arrival)
            for ident,arrival in arrivals.items()
            if ident in done and arrival >= warmup]


def polaris_points(requests, zero):
    result=[]
    for row in requests:
        if row.get("phase") != "measured":
            continue
        start = float(row["started_at_unix"])
        finish = row.get("completed_at_unix",row.get("finished_at_unix"))
        if finish is None: continue
        result.append((start-zero,float(finish)-start))
    return result


def bin_points(points, label):
    rows=[]
    for begin,end in WINDOWS:
        vals=[lat for t,lat in points if begin<=t<end and math.isfinite(lat) and lat>=0]
        rows.append({"system":label,"arrival_start_s":begin,"arrival_end_s":end,
                     "n":len(vals),"mean_s":statistics.mean(vals) if vals else "",
                     "p50_s":percentile(vals,.5),"p95_s":percentile(vals,.95),
                     "p99_s":percentile(vals,.99)})
    return rows


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frontier-run",type=Path,required=True)
    p.add_argument("--frontier-manifest",type=Path,required=True)
    p.add_argument("--polaris-requests",type=Path,required=True)
    p.add_argument("--polaris-zero-unix",type=float,required=True,
                   help="Exact measured-phase start timestamp from Polaris events")
    p.add_argument("--output",type=Path,default=Path("one_reclaim_latency_by_arrival.csv"))
    a=p.parse_args()
    manifest=json.loads(a.frontier_manifest.read_text())
    f=frontier_points(read_jsonl(a.frontier_run/"metrics_ground_truth.jsonl"),
                      float(manifest["warmup_seconds"]))
    z=polaris_points(read_jsonl(a.polaris_requests),a.polaris_zero_unix)
    rows=bin_points(f,"frontier")+bin_points(z,"polaris")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
