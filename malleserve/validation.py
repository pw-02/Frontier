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


def _optional_number(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _valid_summary(row: dict) -> bool:
    submitted = _optional_number(row["submission_pct"])
    failed = _optional_number(row["failed"])
    dropped = _optional_number(row.get("dropped_iterations"))
    # Missing dropped-iteration information is unknown, not zero.
    return (row["validity_status"] == "completed"
            and submitted is not None and submitted >= 99
            and failed == 0 and dropped == 0)


def _from_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            workers = int(row["workers"])
            rate_float = float(row["offered_rps"])
            if not rate_float.is_integer():
                raise ValueError(f"Unsupported fractional rate: {rate_float}")
            rate = int(rate_float)
            yield workers, rate, row


def _new_comparison_row(workers, rate, sim, reference):
    real_p95 = _optional_number(reference.get("latency_p95_s"))
    sim_p95 = sim.get("latency_p95_s_for_completed_measured") if sim else None
    return {
        "workers": workers, "offered_rps": rate,
        **reference,
        "frontier_found": sim is not None,
        "frontier_measured_arrivals": sim.get("measured_arrivals") if sim else None,
        "frontier_completed": sim.get("measured_completed_eventually") if sim else None,
        "frontier_mean_engine_latency_s": sim.get("latency_mean_s_for_completed_measured") if sim else None,
        "frontier_p50_engine_latency_s": sim.get("latency_p50_s_for_completed_measured") if sim else None,
        "frontier_p95_engine_latency_s": sim_p95,
        "frontier_p99_engine_latency_s": sim.get("latency_p99_s_for_completed_measured") if sim else None,
        "p95_relative_difference_pct": round(100 * (sim_p95 / real_p95 - 1), 2)
        if sim_p95 is not None and real_p95 and real_p95 > 0 else None,
        "frontier_slo_attainment_pct": round(100 * sim["measured_slo_attainment"], 3)
        if sim and sim.get("measured_slo_attainment") is not None else None,
        "frontier_slo_goodput_rps": sim.get("measured_slo_goodput_rps_allowing_drain") if sim else None,
        "frontier_completion_in_window_rps": sim.get("measured_completion_throughput_in_window_rps") if sim else None,
        "frontier_backlog_at_window_end": sim.get("all_request_backlog_at_window_end") if sim else None,
        "comparison_note": "Polaris HTTP client e2e versus Frontier simulated completion latency; not identical measurement boundaries",
    }


def compare(polaris: Path, frontier: Path, output: Path) -> list[dict]:
    rows = []
    if polaris.is_file() and polaris.suffix.lower() == ".csv":
        for workers, rate, record in _from_csv(polaris):
            sim = find_frontier(frontier, workers, rate)
            rows.append(_new_comparison_row(workers, rate, sim, {
                "polaris_source": str(polaris),
                "polaris_run_id": record["run_id"],
                "polaris_run_dir": record.get("run_dir", ""),
                "polaris_validity_status": record["validity_status"],
                "polaris_valid_load": _valid_summary(record),
                "polaris_submitted_pct": _optional_number(record["submission_pct"]),
                "polaris_expected_measured": _optional_number(record["expected_measured"]),
                "polaris_measured_requests": _optional_number(record["submitted_measured"]),
                "polaris_completed_ok": _optional_number(record["completed_ok"]),
                "polaris_failed_requests": _optional_number(record["failed"]),
                "polaris_dropped_iterations": _optional_number(record.get("dropped_iterations")),
                "polaris_latency_mean_s": _optional_number(record["latency_mean_s"]),
                "polaris_latency_p50_s": _optional_number(record["latency_p50_s"]),
                "latency_p95_s": _optional_number(record["latency_p95_s"]),
                "polaris_latency_p99_s": _optional_number(record["latency_p99_s"]),
                "polaris_slo_attainment_pct": _optional_number(record["slo_attainment_pct"]),
                "polaris_slo_goodput_rps": _optional_number(record["slo_goodput_rps"]),
                "polaris_completion_in_window_rps": _optional_number(record["completion_in_window_rps_est"]),
                "polaris_backlog_est": _optional_number(record["measured_outstanding_at_end_est"]),
                "polaris_prefix_cache_hit_pct": _optional_number(record.get("prefix_cache_hit_pct")),
            }))
    else:
        for workers, rate, measured, config, source in polaris_runs(polaris):
            exp = config["experiment"]
            if exp["dp_size"] != workers or exp["rate"] != rate or exp["arrival_mode"] != "constant_rate":
                raise ValueError(f"Incompatible Polaris configuration: {source}")
            client = measured["client"]
            latency = client["e2e_latency_seconds"]
            submitted = client.get("measured_load_requests")
            if submitted is None:
                submitted = client.get("successful_checks")
            dropped = client.get("dropped_iterations")
            valid = (measured["validity"]["status"] == "completed"
                     and submitted is not None
                     and 100 * submitted / (rate * exp["measurement_seconds"]) >= 99
                     and client.get("measured_load_failures") == 0 and dropped == 0)
            sim = find_frontier(frontier, workers, rate)
            rows.append(_new_comparison_row(workers, rate, sim, {
                "polaris_source": source,
                "polaris_run_id": Path(source).parent.name,
                "polaris_validity_status": measured["validity"]["status"],
                "polaris_valid_load": valid,
                "polaris_measured_requests": submitted,
                "polaris_failed_requests": client.get("measured_load_failures"),
                "polaris_dropped_iterations": dropped,
                "polaris_latency_mean_s": latency.get("average"),
                "polaris_latency_p50_s": latency.get("p50"),
                "latency_p95_s": latency.get("p95"),
                "polaris_latency_p99_s": latency.get("p99"),
            }))
    if not rows:
        raise ValueError(f"No Polaris reference rows found in {polaris}")
    rows.sort(key=lambda x: (x["workers"], x["offered_rps"], str(x.get("polaris_run_id", ""))))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Source formats can have different columns; preserve every column in CSV.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
