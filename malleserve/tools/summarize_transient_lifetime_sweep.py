"""Aggregate the 12-RPS worker-lifetime sensitivity study.

Run with: python -m malleserve.tools.summarize_transient_lifetime_sweep
Writes CSV in the sweep root, with completion and recovery diagnostics.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUP = ROOT / "outputs/malleserve/runs/transient_lifetime_sweep_r12"
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
    "completed_eventually": "measured_completed_eventually",
    "slo_attainment": "measured_slo_attainment",
}
PATTERN = re.compile(
    r"^(?:fixed(?P<fixed>[48])_(?P<fixed_router>rr|wa|cwa)"
    r"|transient_l(?P<lifetime>\d+)_s(?P<startup>\d+)_"
    r"(?P<recovery>baseline_recovery|ideal_recovery)_(?P<transient_router>rr|wa|cwa)"
    r"|transient_no_reclamation_s(?P<no_startup>\\d+)_(?P<no_router>rr|wa|cwa)"
    r"|transient_single_reclaim_t(?P<reclaim_at>\\d+)_s(?P<one_startup>\\d+)_"
    r"(?P<one_recovery>baseline_recovery|ideal_recovery)_(?P<one_router>rr|wa|cwa))_r12$"
)


def ready_gpu_seconds(path: Path, startup: float, gpus_per_node: int = 4) -> float | None:
    """Integrate ready transient workers over availability trace [0, 330s]."""
    if not path.is_file():
        return None
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts: dict[str, float] = {}
    total = 0.0
    horizon = 330.0
    for event in sorted(events, key=lambda item: item["timestamp"]):
        node = event["node_id"]
        at = float(event["timestamp"])
        if event["action"] == "available":
            starts[node] = at
        elif event["action"] == "reclaim" and node in starts:
            ready = starts.pop(node) + startup
            total += max(0.0, min(at, horizon) - min(max(ready, 0.0), horizon))
    for started in starts.values():
        ready = started + startup
        total += max(0.0, horizon - min(max(ready, 0.0), horizon))
    return total * gpus_per_node


def main() -> None:
    fields = [
        "configuration", "routing", "name", "status", "startup_s",
        "lifetime_s", "extra_ready_gpu_seconds", "mean_ready_gpus",
        *METRICS, "retries", "lost_processed_tokens",
        "oracle_restored_requests", "oracle_restored_tokens",
    ]
    rows = []
    for mode in ("fixed", "transient"):
        parent = GROUP / mode
        if not parent.exists():
            continue
        for folder in parent.iterdir():
            if not folder.is_dir() or not folder.name.endswith("_" + MODEL):
                continue
            name = folder.name[: -len("_" + MODEL)]
            match = PATTERN.fullmatch(name)
            if match is None:
                continue
            info = match.groupdict()
            fixed = info["fixed"] is not None
            routing = (info["fixed_router"] or info["transient_router"]
                       or info["no_router"] or info["one_router"])
            if fixed:
                configuration = "Fixed " + info["fixed"]
            elif info["no_router"]:
                configuration = "No Reclamation"
            elif info["one_router"]:
                configuration = ("Single Reclamation / " +
                                 info["one_recovery"].replace("_", " ").title())
            else:
                configuration = info["recovery"].replace("_", " ").title() + " / " + info["lifetime"] + "s"
            startup_value = info["startup"] or info["no_startup"] or info["one_startup"]
            row = {
                "configuration": configuration, "routing": routing, "name": name,
                "status": "MISSING",
                "startup_s": None if fixed else int(startup_value),
                "lifetime_s": int(info["lifetime"]) if info["lifetime"] else None,
            }
            summary_file = folder / "malleserve_summary.json"
            completed = folder / "run_complete.json"
            if not summary_file.is_file() or not completed.is_file():
                rows.append(row)
                continue
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                row.update({k: summary.get(v) for k, v in METRICS.items()})
                if not fixed:
                    extra = ready_gpu_seconds(folder / "availability.jsonl", float(startup_value))
                    row["extra_ready_gpu_seconds"] = extra
                    row["mean_ready_gpus"] = (4.0 + extra / 330.0) if extra is not None else None
                    lifecycle_file = folder / "transient_lifecycle.json"
                    if lifecycle_file.is_file():
                        lifecycle = json.loads(lifecycle_file.read_text(encoding="utf-8"))
                        for k in ("retries", "lost_processed_tokens",
                                  "oracle_restored_requests", "oracle_restored_tokens"):
                            row[k] = lifecycle.get(k)
                row["status"] = "ok"
            except (OSError, ValueError, TypeError) as exc:
                row["status"] = f"ERROR: {exc}"
            rows.append(row)

    rows.sort(key=lambda r: (
        0 if r["name"].startswith("fixed4_") else
        1 if r["name"].startswith("fixed8_") else 2,
        r["lifetime_s"] or 0, r["configuration"], r["routing"],
    ))
    GROUP.mkdir(parents=True, exist_ok=True)
    path = GROUP / "transient_lifetime_comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{"Configuration":<34} {"Route":<6} {"Mean":>9} {"P95":>9} {"P99":>9} {"Status":>8}')
    print("-" * 78)
    for r in rows:
        vals = [r.get(k) for k in ("mean_s", "p95_s", "p99_s")]
        metrics = [f"{v:.3f}" if isinstance(v, (int, float)) else "-" for v in vals]
        print(f'{r["configuration"]:<34} {r["routing"]:<6} {metrics[0]:>9} {metrics[1]:>9} {metrics[2]:>9} {r["status"]:>8}')
    print(f"CSV: {path}")


if __name__ == "__main__":
    main()
