#!/usr/bin/env python3
"""Run a small factorial transient-capacity feasibility sweep.

All scenarios use native Frontier's Mooncake trace converter for arrivals,
but the transient simulations are exploratory, standalone macro models.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    tools_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--service-profile", type=Path, required=True)
    p.add_argument("--rates", type=int, nargs="+", default=[8, 10, 12, 14, 16])
    p.add_argument("--lifetimes", type=float, nargs="+", default=[30, 60, 120])
    p.add_argument("--startup-seconds", type=float, required=True)
    p.add_argument("--warmup-seconds", type=int, default=30)
    p.add_argument("--measurement-seconds", type=int, default=300)
    p.add_argument("--slo-seconds", type=float, default=30)
    p.add_argument("--drain-seconds", type=float, default=600)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if any(rate <= 0 for rate in args.rates):
        p.error("Rates must be positive")
    if any(life <= 0 for life in args.lifetimes):
        p.error("Lifetimes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = args.output_dir / "traces"
    traces.mkdir(exist_ok=True)
    records = []
    for rate in args.rates:
        trace = traces / f"mooncake_r{rate}.csv"
        subprocess.run([
            sys.executable, str(root / "examples/malleserve/convert_mooncake.py"),
            "--dataset", str(args.dataset), "--rate", str(rate),
            "--warmup-seconds", str(args.warmup_seconds),
            "--measurement-seconds", str(args.measurement_seconds),
            "--output", str(trace),
        ], check=True)
        for life in args.lifetimes:
            label = f"r{rate}_life{life:g}".replace(".", "p")
            out = args.output_dir / label
            subprocess.run([
                sys.executable, str(tools_dir / "simulate.py"),
                "--trace", str(trace),
                "--service-profile", str(args.service_profile),
                "--startup-seconds", str(args.startup_seconds),
                "--lifetime-seconds", str(life),
                "--warmup-seconds", str(args.warmup_seconds),
                "--measurement-seconds", str(args.measurement_seconds),
                "--slo-seconds", str(args.slo_seconds),
                "--drain-seconds", str(args.drain_seconds),
                "--output-dir", str(out),
            ], check=True)
            with (out / "comparison.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    records.append({
                        "offered_rps": rate,
                        "lifetime_s": life,
                        "startup_s_assumed": args.startup_seconds,
                        **row,
                    })
    output = args.output_dir / "sweep_comparison.csv"
    if records:
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (args.output_dir / "sweep_manifest.json").write_text(
        json.dumps({
            "model_kind": "unvalidated Frontier-derived macro processor-sharing model",
            "dataset": str(args.dataset),
            "profile": str(args.service_profile),
            "rates": args.rates,
            "lifetimes_s": args.lifetimes,
            "startup_s_assumed": args.startup_seconds,
            "warning": "Not native Frontier continuous batching; quantitative results are exploratory.",
        }, indent=2) + "\n"
    )
    print(f"Saved {len(records)} scenario-policy rows to {output}")


if __name__ == "__main__":
    main()
