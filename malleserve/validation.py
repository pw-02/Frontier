"""Compare Polaris fixed-capacity run(s) with native Frontier output.

Supports an uploaded ZIP of a single run or a directory tree of Polaris runs.
Never treats HTTP client latency as an exact match for simulated engine latency.
"""
from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path

NAME = re.compile(r"fixed-workers-(\d+)-r(\d+)")


def polaris_runs(path: Path):
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            for member in names:
                if not member.endswith("/summary.json"):
                    continue
                match = NAME.search(member)
                if not match:
                    continue
                base = member.rsplit("/", 1)[0] + "/"
                config_name = base + "config.json"
                if config_name not in names:
                    continue
                yield int(match[1]), int(match[2]), json.loads(z.read(member)), json.loads(z.read(config_name)), member
    else:
        for file in sorted(path.rglob("summary.json")):
            match = NAME.search(str(file))
            if not match or not (file.parent / "config.json").exists():
                continue
            yield int(match[1]), int(match[2]), json.loads(file.read_text()), json.loads((file.parent / "config.json").read_text()), str(file)


def find_frontier(root: Path, workers: int, rate: int):
    possible = []
    # Canonical Hydra names and historical matrix validation names.
    for name in (f"fixed{workers}_mooncake_r{rate}", f"validation_w{workers}_r{rate}"):
        possible.extend(root.glob(f"{name}/**/malleserve_summary.json"))
    if not possible:
        return None
    if len(possible) != 1:
        raise ValueError(f"Multiple Frontier results for {workers} GPUs / {rate} RPS: {possible}")
    return json.loads(possible[0].read_text())


def compare(polaris: Path, frontier: Path, output: Path) -> list[dict]:
    rows = []
    for workers, rate, measured, config, source in polaris_runs(polaris):
        exp = config["experiment"]
        if (exp["dp_size"] != workers or exp["rate"] != rate or
            exp["arrival_mode"] != "constant_rate"):
            raise ValueError(f"Incompatible Polaris configuration: {source}")
        client = measured["client"]
        latency = client["e2e_latency_seconds"]
        real_p95 = latency.get("p95")
        real_mean = latency.get("average")
        sim = find_frontier(frontier, workers, rate)
        sim_p95 = sim.get("latency_p95_s_for_completed_measured") if sim else None
        sim_mean = sim.get("latency_mean_s_for_completed_measured") if sim else None
        expected = rate * exp["measurement_seconds"]
        valid = (
            measured["validity"]["status"] == "completed"
            and (client["measured_load_requests"] if client["measured_load_requests"] is not None else client["successful_checks"]) == expected
            and client["measured_load_failures"] == 0
            and client["dropped_iterations"] == 0
        )
        rows.append({
            "workers": workers, "offered_rps": rate,
            "polaris_source": source, "polaris_valid_load": valid,
            "polaris_measured_requests": client["measured_load_requests"] if client["measured_load_requests"] is not None else client["successful_checks"],
            "polaris_failed_requests": client["measured_load_failures"],
            "polaris_mean_client_latency_s": real_mean,
            "frontier_mean_engine_latency_s": sim_mean,
            "polaris_p95_client_latency_s": real_p95,
            "frontier_p95_engine_latency_s": sim_p95,
            "p95_relative_difference_pct": (
                round((sim_p95 / real_p95 - 1) * 100, 2)
                if sim_p95 is not None and real_p95 else None
            ),
            "frontier_measured_arrivals": sim.get("measured_arrivals") if sim else None,
            "frontier_completed": sim.get("measured_completed_eventually") if sim else None,
            "frontier_slo_attainment": sim.get("measured_slo_attainment") if sim else None,
            "frontier_slo_goodput_rps": sim.get("measured_slo_goodput_rps_allowing_drain") if sim else None,
            "frontier_backlog_at_window_end": sim.get("all_request_backlog_at_window_end") if sim else None,
            "frontier_found": sim is not None,
            "comparison_note": "Polaris HTTP client e2e versus Frontier simulated completion latency; not identical measurement boundaries",
        })
    if not rows:
        raise ValueError(f"No Polaris fixed-workers-*-r* summary/config pairs found under {polaris}")
    rows.sort(key=lambda x: (x["workers"], x["offered_rps"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
