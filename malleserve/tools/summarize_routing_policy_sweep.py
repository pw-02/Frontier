"""Summarize the 12-run routing-policy sweep in one comparison CSV.

Usage: python -m malleserve.tools.summarize_routing_policy_sweep
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/malleserve/runs/routing_policy_sweep"
MODEL = "qwen_qwen3_8b"

METRICS = {
    "mean_s": "latency_mean_s_for_completed_measured",
    "p50_s": "latency_p50_s_for_completed_measured",
    "p95_s": "latency_p95_s_for_completed_measured",
    "p99_s": "latency_p99_s_for_completed_measured",
    "ttft_p95_s": "ttft_p95_s_for_completed_measured",
    "queue_p95_s": "queue_wait_p95_s_for_completed_measured",
    "throughput_rps": "measured_completion_throughput_in_window_rps",
    "outstanding": "measured_still_outstanding_at_window_end",
    "cached_prefill_tokens_mean": "csv_cached_prefill_tokens_mean",
}


def main() -> None:
    rows = []
    for label, mode, prefix in (
        ("Fixed 4", "fixed", "fixed4"),
        ("Fixed 8", "fixed", "fixed8"),
        ("Baseline Recovery", "transient", "transient_baseline_recovery"),
        ("Ideal Recovery", "transient", "transient_ideal_recovery"),
    ):
        for routing in ("rr", "wa", "cwa"):
            name = f"{prefix}_{routing}_r10"
            path = RUNS / mode / f"{name}_{MODEL}" / "malleserve_summary.json"
            row = {"configuration": label, "routing": routing, "name": name}
            if not path.is_file():
                row.update({"status": "MISSING"})
                rows.append(row)
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                row.update({"status": f"ERROR: {exc}"})
                rows.append(row)
                continue
            row["status"] = "ok"
            row.update({column: data.get(key) for column, key in METRICS.items()})
            rows.append(row)

    RUNS.mkdir(parents=True, exist_ok=True)
    destination = RUNS / "routing_policy_comparison.csv"
    columns = ["configuration", "routing", "name", "status", *METRICS]
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{"Configuration":<19} {"Routing":<7} {"Mean":>9} {"P95":>9} {"Status":>10}')
    print("-" * 59)
    for row in rows:
        if row["status"] != "ok":
            print(f'{row["configuration"]:<19} {row["routing"]:<7} {"-":>9} {"-":>9} {row["status"]:>10}')
        else:
            print(f'{row["configuration"]:<19} {row["routing"]:<7} {row["mean_s"]:>9.3f} {row["p95_s"]:>9.3f} {"ok":>10}')
    print(f"CSV: {destination}")


if __name__ == "__main__":
    main()
