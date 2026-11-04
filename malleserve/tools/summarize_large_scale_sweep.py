"""Aggregate medium-scale HPC transient simulation runs by routing and seed."""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "outputs/malleserve/runs/large_scale_hpc_r34"
MODEL = "qwen_qwen3_8b"
METRICS = {
    "mean_s": "latency_mean_s_for_completed_measured",
    "p50_s": "latency_p50_s_for_completed_measured",
    "p95_s": "latency_p95_s_for_completed_measured",
    "p99_s": "latency_p99_s_for_completed_measured",
    "ttft_p95_s": "ttft_p95_s_for_completed_measured",
    "completion_rps": "measured_completion_throughput_in_window_rps",
    "outstanding": "measured_still_outstanding_at_window_end",
    "completed_eventually": "measured_completed_eventually",
    "slo_attainment": "measured_slo_attainment",
}
PATTERN = re.compile(
    r"^(?:fixed(?P<fixed>8|16|32)_(?P<fixed_route>rr|wa|cwa)"
    r"|large_s(?P<seed>\d+)_(?P<recovery>baseline_recovery|ideal_recovery)_"
    r"(?P<transient_route>rr|wa|cwa))_r34$"
)


def main() -> None:
    rows = []
    for mode in ("fixed", "transient"):
        parent = ROOT / mode
        if not parent.is_dir():
            continue
        for folder in sorted(parent.iterdir()):
            if not folder.is_dir() or not folder.name.endswith("_" + MODEL):
                continue
            name = folder.name[:-len("_" + MODEL)]
            match = PATTERN.fullmatch(name)
            if match is None:
                continue
            match_data = match.groupdict()
            route = match_data["fixed_route"] or match_data["transient_route"]
            row = {
                "configuration": ("Fixed " + match_data["fixed"] if mode == "fixed"
                                  else match_data["recovery"]),
                "routing": route,
                "seed": match_data["seed"] or "",
                "name": name, "status": "MISSING",
            }
            p = folder / "malleserve_summary.json"
            if not p.is_file() or not (folder / "run_complete.json").is_file():
                rows.append(row)
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            row.update({key: data.get(field) for key, field in METRICS.items()})
            if mode == "transient":
                lifecycle = folder / "transient_lifecycle.json"
                if lifecycle.is_file():
                    details = json.loads(lifecycle.read_text(encoding="utf-8"))
                    for key in ("retries", "lost_processed_tokens",
                                "oracle_restored_requests", "oracle_restored_tokens"):
                        row[key] = details.get(key)
                schedule = folder / "availability.jsonl"
                if schedule.is_file():
                    events = [json.loads(s) for s in schedule.read_text(encoding="utf-8").splitlines() if s]
                    row["candidate_nodes"] = len({event["node_id"] for event in events})
                    row["reclamations"] = sum(event["action"] == "reclaim" for event in events)
            row["status"] = "ok"
            rows.append(row)
    fields = (["configuration", "routing", "seed", "name", "status",
               "candidate_nodes", "reclamations", *METRICS,
               "retries", "lost_processed_tokens",
               "oracle_restored_requests", "oracle_restored_tokens"])
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / "large_scale_comparison.csv"
    with target.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{"Configuration":<20} {"Router":<7} {"Seed":<6} {"Mean":>9} {"P95":>9} {"P99":>9} {"Outstanding":>12}')
    for row in rows:
        show = lambda key: ("-" if row.get(key) is None else f"{row[key]:.3f}")
        print(f'{row["configuration"]:<20} {row["routing"]:<7} {row["seed"]:<6} '
              f'{show("mean_s"):>9} {show("p95_s"):>9} {show("p99_s"):>9} '
              f'{row.get("outstanding", "-"):>12}')
    print(f"CSV: {target}")


if __name__ == "__main__":
    main()
