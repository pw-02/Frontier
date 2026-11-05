"""Canonical comparison across E1 stable, E2 no reclaim, E3 no recovery, E4 full restart."""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "outputs/malleserve/runs/e1_e4_hpc_r34"
MODEL = "qwen_qwen3_8b"
METRICS = {
    "mean_s_successes": "latency_mean_s_for_completed_measured",
    "p95_s_successes": "latency_p95_s_for_completed_measured",
    "p99_s_successes": "latency_p99_s_for_completed_measured",
    "slo_goodput_rps": "measured_slo_goodput_rps_allowing_drain",
    "slo_attainment": "measured_slo_attainment",
    "completion_rps": "measured_completion_throughput_in_window_rps",
    "measured_arrivals": "measured_arrivals",
    "completed": "measured_completed_eventually",
    "failed": "measured_failed",
    "unresolved": "measured_unresolved",
    "outstanding_end": "measured_still_outstanding_at_window_end",
}
PATTERN = re.compile(
    r"^(?P<case>e[1-4])_(?:fixed(?P<gpus>8|16|32)|s(?P<seed>\\d+))_"
    r"(?P<route>rr|wa|cwa)_r34$"
)


def main() -> None:
    columns = ["case", "fixed_gpus", "seed", "routing", "name", "status",
               *METRICS, "retries", "failed_assignments",
               "lost_processed_tokens", "reclamations", "candidate_nodes"]
    rows = []
    for directory in (ROOT / "fixed", ROOT / "transient"):
        if not directory.is_dir():
            continue
        for folder in sorted(directory.iterdir()):
            if not folder.is_dir() or not folder.name.endswith("_" + MODEL):
                continue
            name = folder.name[:-len("_" + MODEL)]
            parsed = PATTERN.fullmatch(name)
            if not parsed:
                continue
            info = parsed.groupdict()
            row = {
                "case": info["case"], "fixed_gpus": info["gpus"] or "",
                "seed": info["seed"] or "", "routing": info["route"],
                "name": name, "status": "MISSING",
            }
            datafile = folder / "malleserve_summary.json"
            if not datafile.is_file() or not (folder / "run_complete.json").is_file():
                rows.append(row)
                continue
            data = json.loads(datafile.read_text(encoding="utf-8"))
            row.update({label: data.get(field) for label, field in METRICS.items()})
            if directory.name == "transient":
                lifecyclefile = folder / "transient_lifecycle.json"
                if lifecyclefile.is_file():
                    lifecycle = json.loads(lifecyclefile.read_text(encoding="utf-8"))
                    row["retries"] = lifecycle.get("retries")
                    row["failed_assignments"] = lifecycle.get("failed_requests")
                    row["lost_processed_tokens"] = lifecycle.get("lost_processed_tokens")
                schedulefile = folder / "availability.jsonl"
                if schedulefile.is_file():
                    events = [json.loads(line) for line in schedulefile.read_text(encoding="utf-8").splitlines() if line]
                    row["reclamations"] = sum(event["action"] == "reclaim" for event in events)
                    row["candidate_nodes"] = len({event["node_id"] for event in events})
            row["status"] = "ok"
            rows.append(row)
    rows.sort(key=lambda row: (row["case"], row["fixed_gpus"], row["seed"], row["routing"]))
    ROOT.mkdir(parents=True, exist_ok=True)
    outfile = ROOT / "e1_e4_comparison.csv"
    with outfile.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{"Case":<6} {"Router":<7} {"Seed":<5} {"P95*":>9} {"P99*":>9} {"Failed":>8} {"Goodput":>9} {"Outstanding":>12}')
    for row in rows:
        fmt = lambda key: "-" if row.get(key) is None else f'{row[key]:.3f}'
        print(f'{row["case"]:<6} {row["routing"]:<7} {row["seed"]:<5} '
              f'{fmt("p95_s_successes"):>9} {fmt("p99_s_successes"):>9} '
              f'{str(row.get("failed", "-")):>8} {fmt("slo_goodput_rps"):>9} '
              f'{str(row.get("outstanding_end", "-")):>12}')
    print("* P95/P99 are for completed requests only; E3 failures are not included.")
    print(f"CSV: {outfile}")


if __name__ == "__main__":
    main()
