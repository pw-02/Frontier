"""Compare Polaris fixed-capacity run(s) with native Frontier output.

Uses the consolidated Polaris validation summary CSV as its sole input.
Never treats HTTP client latency as an exact match for simulated engine latency.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

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


def find_frontier(root: Path, workers: int, rate: int):
    names = [f"fixed{workers}_r{rate}", f"fixed{workers}_mooncake_r{rate}", f"validation_w{workers}_r{rate}"]
    matches = [path for name in names
               for path in (root / name).rglob("malleserve_summary.json") if path.is_file()]
    if len(matches) > 1:
        raise ValueError(f"Multiple Frontier summaries for GPUs={workers}, RPS={rate}")
    if not matches:
        return None
    path = matches[0]
    raw = path.read_text(encoding="utf-8")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Older MalleServe runs could leave a literal backslash-n suffix.
        # Accept *only* that exact artifact; never silently ignore extra JSON.
        if raw.endswith("\\n") and not raw.endswith("\n"):
            try:
                result = json.loads(raw[:-2])
            except json.JSONDecodeError:
                raise ValueError(f"Invalid Frontier summary {path}: {exc}") from exc
        else:
            raise ValueError(f"Invalid Frontier summary {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Frontier summary must be a JSON object: {path}")
    return result


def compare(polaris: Path, frontier: Path, output: Path) -> list[dict]:
    rows = []
    if not polaris.is_file() or polaris.suffix.lower() != ".csv":
        raise ValueError("Polaris input must be a summary CSV file")
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
