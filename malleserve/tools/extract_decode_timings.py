#!/usr/bin/env python3
"""Extract predicted decode-only batch timings from Frontier's stage ledger.

A decode-only stage here is a MONOLITHIC stage whose scheduled requests
each consume exactly one token; this matches the existing diagnostic
convention but does not establish current KV-context matching.

Usage:
  python -m malleserve.tools.extract_decode_timings \\
    --ledger outputs/malleserve/runs/fixed/fixed4_r10/frontier_stage_batch_ledger.jsonl \\
    --output-dir outputs/malleserve/decode_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def extract(ledger: Path):
    groups = defaultdict(list)
    records = []
    total = 0
    with ledger.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if row.get("cluster_type") != "MONOLITHIC":
                continue
            tokens = row.get("request_num_tokens") or []
            if not tokens or any(int(n) != 1 for n in tokens):
                continue
            timing = row.get("execution_time") or {}
            if "model_time_ms" not in timing or "total_time_ms" not in timing:
                raise ValueError(f"Missing timing fields at ledger line {total}")
            model_ms = float(timing["model_time_ms"])
            total_ms = float(timing["total_time_ms"])
            record = {
                "batch_id": row.get("batch_id"),
                "batch_size": len(tokens),
                "model_ms": model_ms,
                "stage_total_ms": total_ms,
                "non_model_ms": total_ms - model_ms,
            }
            records.append(record)
            groups[len(tokens)].append(record)
    summary = []
    for batch_size, values in sorted(groups.items()):
        models = [r["model_ms"] for r in values]
        totals = [r["stage_total_ms"] for r in values]
        summary.append({
            "batch_size": batch_size,
            "count": len(values),
            "mean_model_ms": statistics.mean(models),
            "median_model_ms": statistics.median(models),
            "mean_stage_ms": statistics.mean(totals),
            "median_stage_ms": statistics.median(totals),
        })
    return total, records, summary


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    total, records, summary = extract(args.ledger)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "decode_stages.csv", records,
              ["batch_id", "batch_size", "model_ms", "stage_total_ms", "non_model_ms"])
    write_csv(args.output_dir / "decode_batch_summary.csv", summary,
              ["batch_size", "count", "mean_model_ms", "median_model_ms",
               "mean_stage_ms", "median_stage_ms"])
    print(f"Ledger entries: {total:,}; decode-only stages: {len(records):,}")
    for item in summary:
        print(f"Batch {item['batch_size']:>3}: {item['count']:>7,} stages, "
              f"median model {item['median_model_ms']:.3f} ms, "
              f"median total {item['median_stage_ms']:.3f} ms")
    print(f"Saved {args.output_dir / 'decode_stages.csv'}")
    print(f"Saved {args.output_dir / 'decode_batch_summary.csv'}")
    print("NOTE: predicted stage times, not measured GPU times; "
          "current KV contexts are not matched.")


if __name__ == "__main__":
    main()
