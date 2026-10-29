#!/usr/bin/env python3
"""Run a reproducible native Frontier transient-resource sensitivity sweep.

This invokes the EXISTING native vLLM v1 event engine via run_native_transient.sh.
It deliberately does not use the historical processor-sharing macro proxy.
Assumed startup delays are sensitivity parameters, not Polaris measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def positive_finite(values: list[float], label: str, allow_zero: bool = False) -> None:
    if not values or any(not math.isfinite(v) or (v < 0 if allow_zero else v <= 0) for v in values):
        raise ValueError(f"{label} values must be finite and {'nonnegative' if allow_zero else 'positive'}")


def load_summary(metrics_root: Path) -> tuple[dict, dict]:
    summaries = sorted(metrics_root.rglob("malleserve_summary.json"))
    lifecycles = sorted(metrics_root.rglob("transient_lifecycle.json"))
    if len(summaries) != 1 or len(lifecycles) != 1:
        raise ValueError(
            f"Expected one summary and lifecycle file in {metrics_root}; "
            f"found {len(summaries)} summaries and {len(lifecycles)} lifecycles"
        )
    return (json.loads(summaries[0].read_text(encoding="utf-8")),
            json.loads(lifecycles[0].read_text(encoding="utf-8")))


def flatten(rate: int, lifetime: float, startup: float, run_id: str,
            metrics_root: Path, summary: dict, lifecycle: dict) -> dict:
    # Keep lifecycle's complete shape in the saved JSON; only stable scalar
    # fields are flattened into the comparison table.
    row = {
        "run_id": run_id,
        "offered_rps": rate,
        "lifetime_s": lifetime,
        "startup_s_assumed": startup,
        "metrics_root": str(metrics_root),
    }
    for key in (
        "measured_arrivals", "measured_completed_eventually",
        "measured_no_completion_event", "measured_slo_attainment",
        "measured_slo_goodput_rps_allowing_drain",
        "measured_completion_throughput_in_window_rps",
        "all_request_backlog_at_window_end",
        "latency_p95_s_for_completed_measured",
        "queue_wait_p95_s_for_completed_measured",
        "ttft_p95_s_for_completed_measured",
    ):
        row[key] = summary.get(key)
    for key in ("retries", "lost_processed_tokens", "stale_event_purges"):
        row[key] = lifecycle.get(key)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, default=ROOT / "data/malleserve",
                   help="Contains mooncake_r<RATE>_330s.csv and matching .manifest.json")
    p.add_argument("--rates", type=int, nargs="+", default=[2, 10, 12])
    p.add_argument("--lifetimes", type=float, nargs="+", default=[30, 60, 120])
    p.add_argument("--startups", type=float, nargs="+", default=[0, 20, 40, 80])
    p.add_argument("--duration-seconds", type=float, default=330.0)
    p.add_argument("--slo-seconds", type=float, default=30.0)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/examples/malleserve/native_sweep")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    positive_finite([float(x) for x in a.rates], "rates")
    positive_finite(a.lifetimes, "lifetimes")
    positive_finite(a.startups, "startups", allow_zero=True)
    positive_finite([a.duration_seconds, a.slo_seconds], "duration/SLO")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    schedules = a.output_dir / "schedules"
    schedules.mkdir(exist_ok=True)
    records = []
    planned = []
    for rate in a.rates:
        trace = a.trace_dir / f"mooncake_r{rate}_330s.csv"
        manifest = trace.with_suffix(".manifest.json")
        if not a.dry_run and (not trace.is_file() or not manifest.is_file()):
            raise FileNotFoundError(f"Missing trace or manifest: {trace}, {manifest}")
        for lifetime in a.lifetimes:
            life_label = f"{lifetime:g}".replace(".", "p")
            schedule = schedules / f"rotation_l{life_label}.jsonl"
            generation = [
                "python", str(HERE / "generate_native_schedule.py"),
                "--duration-seconds", str(a.duration_seconds),
                "--lifetime-seconds", str(lifetime), "--output", str(schedule),
            ]
            if not a.dry_run:
                subprocess.run(generation, cwd=ROOT, check=True)
            for startup in a.startups:
                start_label = f"{startup:g}".replace(".", "p")
                run_id = f"native_r{rate}_l{life_label}_s{start_label}"
                metrics_root = a.output_dir / "runs" / run_id
                command = [
                    "bash", str(HERE / "run_native_transient.sh"),
                    str(trace), str(schedule), run_id,
                ]
                planned.append({"run_id": run_id, "trace": str(trace),
                                "schedule": str(schedule), "startup_s": startup,
                                "command": command})
                if a.dry_run:
                    print("DRY RUN:", " ".join(command), f"startup={startup:g}")
                    continue
                env = dict(os.environ,
                           TRANSIENT_STARTUP_SECONDS=str(startup),
                           METRICS_OUTPUT_DIR=str(metrics_root),
                           SLO_SECONDS=str(a.slo_seconds))
                subprocess.run(command, cwd=ROOT, env=env, check=True)
                summary, lifecycle = load_summary(metrics_root)
                records.append(flatten(rate, lifetime, startup, run_id,
                                       metrics_root, summary, lifecycle))
                # Save incrementally so successful runs survive later failures.
                with (a.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
    (a.output_dir / "experiment_manifest.json").write_text(
        json.dumps({
            "engine": "native Frontier monolithic vLLM v1 DES",
            "policy": "naive activate-all; kill and retry on reclaim",
            "trace_dir": str(a.trace_dir),
            "rates": a.rates, "lifetimes_s": a.lifetimes,
            "startup_delays_s_assumed": a.startups,
            "duration_s": a.duration_seconds,
            "slo_s": a.slo_seconds,
            "dry_run": a.dry_run,
            "planned_runs": planned,
            "warning": "Startup delays are assumptions. No Polaris validation implied.",
        }, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(planned)} scenarios planned, {len(records)} completed")


if __name__ == "__main__":
    main()
