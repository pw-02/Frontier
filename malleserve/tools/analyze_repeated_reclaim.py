"""Evidence-only analysis of native Frontier repeated-reclamation runs.

Reads existing output artifacts; never changes scheduling or runtime.
Usage:
 python -m malleserve.tools.analyze_repeated_reclaim \
   --run outputs/malleserve/runs/naive_r10 \
   --output outputs/malleserve/validation/naive_reclaim_windows.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from malleserve.tools.analyze_mooncake import percentile


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(run, start=30., end=330., step=20.):
    events = read_jsonl(run / "metrics_ground_truth.jsonl")
    arrivals, finishes = {}, {}
    for e in events:
        ident = str(e.get("request_id"))
        if e.get("event_type") == "request_arrival":
            arrivals[ident] = float(e["arrived_at"])
        elif e.get("event_type") == "request_completion":
            finishes[ident] = float(e["completed_at"])
    lifecycle = json.loads((run/"transient_lifecycle.json").read_text())
    transitions = lifecycle.get("events", [])
    reclaim = [e for e in transitions if e.get("action") == "reclaim"]
    ready = [e for e in transitions if e.get("action") == "ready"]
    summary = json.loads((run/"malleserve_summary.json").read_text())
    rows = []
    t = start
    while t < end - 1e-9:
        hi = min(t+step,end)
        ids = [i for i,a in arrivals.items() if t <= a < hi]
        completed = [i for i in ids if i in finishes]
        lats = [finishes[i] - arrivals[i] for i in completed]
        if any(not math.isfinite(x) or x<0 for x in lats):
            raise ValueError("Invalid latency")
        rows.append(dict(
            start_s=round(t,4), end_s=round(hi,4),
            arrived=len(ids), completed=len(completed),
            incomplete=len(ids)-len(completed),
            mean_s=mean(lats) if lats else "",
            p50_s=percentile(lats,.5),
            p95_s=percentile(lats,.95),
            p99_s=percentile(lats,.99),
            completed_after_window=sum(finishes[i]>=hi for i in completed),
            reclaims_in_window=sum(t<=e["timestamp"]<hi for e in reclaim),
            ready_in_window=sum(t<=e["timestamp"]<hi for e in ready),
        ))
        t = hi
    metadata = dict(
        total_arrivals=len(arrivals), total_completed=len(finishes),
        retries=lifecycle.get("retries"),
        lost_processed_tokens=lifecycle.get("lost_processed_tokens"),
        purged_stale_gpu_events=lifecycle.get("purged_stale_gpu_events"),
        measured_p95_s=summary.get("latency_p95_s_for_completed_measured"),
        measured_queue_wait_p95_s=summary.get("queue_wait_p95_s_for_completed_measured"),
        measured_ttft_p95_s=summary.get("ttft_p95_s_for_completed_measured"),
        reclamations=[dict(time_s=e["timestamp"],
                           node=e.get("node_id"),
                           retried_requests=e.get("retried_requests"))
                      for e in reclaim],
        readme="Window latencies use arrival clock and all eventual completions. "
               "Retry counts are lifecycle attempts, not necessarily distinct requests. "
               "No claim of queueing causality follows from aggregate percentiles.",
    )
    return rows, metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--start",type=float,default=30.)
    p.add_argument("--end",type=float,default=330.)
    p.add_argument("--step",type=float,default=20.)
    a=p.parse_args()
    rows, info=analyze(a.run,a.start,a.end,a.step)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    info_path=a.output.with_suffix(".summary.json")
    info_path.write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(info,indent=2))
    print("Wrote",a.output,"and",info_path)


if __name__=="__main__":
    main()
