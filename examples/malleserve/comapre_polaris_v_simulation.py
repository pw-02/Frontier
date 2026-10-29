#!/usr/bin/env python3
"""Join measured-phase Polaris extraction with Frontier malleserve_summary.json per cell.

Use: python validation/compare_polaris_frontier.py \
  --real validation/polaris/polaris_validation_summary.csv \
  --sim-root outputs/examples/malleserve \
  --output validation/frontier_vs_polaris.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def display(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=Path("/home/pwatters/projects/malle-serve/experiments/fixed_capacity/analysis/polaris_validation_summary.csv"))
    parser.add_argument("--sim-root", type=Path, default=Path("outputs/examples/malleserve"))
    parser.add_argument("--output", type=Path, default=Path("validation/frontier_vs_polaris.csv"))
    args = parser.parse_args()
    with args.real.open(newline="", encoding="utf-8-sig") as f:
        real_rows = list(csv.DictReader(f))
    if not real_rows:
        parser.error(f"No rows in {args.real}")

    result = []
    for row in sorted(real_rows, key=lambda r: (int(r["workers"]), float(r["offered_rps"]), r["run_id"])):
        workers = int(row["workers"])
        rate = float(row["offered_rps"])
        if not rate.is_integer():
            raise ValueError(f"Expected integer rate, got {rate}")
        rate = int(rate)
        run_id = f"validation_w{workers}_r{rate}"
        candidates = sorted((args.sim_root / run_id).rglob("malleserve_summary.json"))
        if len(candidates) > 1:
            raise ValueError(f"Multiple Frontier summaries for {run_id}: {candidates}")
        sim = json.loads(candidates[0].read_text()) if candidates else {}

        real_p95 = number(row.get("latency_p95_s"))
        sim_p95 = number(sim.get("latency_p95_s_for_completed_measured"))
        real_goodput = number(row.get("slo_goodput_rps"))
        sim_goodput = number(sim.get("measured_slo_goodput_rps_allowing_drain"))
        submission_pct = number(row.get("submission_pct"))
        failures = number(row.get("failed"))
        dropped = number(row.get("dropped_iterations"))
        # Do not reject an overloaded run merely because it fails its latency SLO.
        comparable_traffic = (submission_pct is not None and submission_pct >= 99
                              and failures == 0 and (dropped is None or dropped == 0))
        result.append({
            "workers": workers,
            "offered_rps": rate,
            "polaris_run_id": row["run_id"],
            "polaris_submitted_pct": row.get("submission_pct", ""),
            "polaris_http_failures": row.get("failed", ""),
            "polaris_dropped_iterations": row.get("dropped_iterations", ""),
            "polaris_validity_status": row.get("validity_status", ""),
            "comparable_traffic": comparable_traffic,
            "polaris_p50_s": row.get("latency_p50_s", ""),
            "frontier_p50_s": sim.get("latency_p50_s_for_completed_measured", ""),
            "polaris_p95_s": row.get("latency_p95_s", ""),
            "frontier_p95_s": sim.get("latency_p95_s_for_completed_measured", ""),
            "p95_relative_error_pct": (
                round(100 * (sim_p95 / real_p95 - 1), 2)
                if sim_p95 is not None and real_p95 and real_p95 > 0 else ""
            ),
            "polaris_p99_s": row.get("latency_p99_s", ""),
            "frontier_p99_s": sim.get("latency_p99_s_for_completed_measured", ""),
            "polaris_slo_attainment_pct": row.get("slo_attainment_pct", ""),
            "frontier_slo_attainment_pct": (
                round(100 * sim["measured_slo_attainment"], 3)
                if "measured_slo_attainment" in sim else ""
            ),
            "polaris_slo_goodput_rps": row.get("slo_goodput_rps", ""),
            "frontier_slo_goodput_rps": sim.get("measured_slo_goodput_rps_allowing_drain", ""),
            "goodput_error_rps": (
                round(sim_goodput - real_goodput, 3)
                if sim_goodput is not None and real_goodput is not None else ""
            ),
            "polaris_backlog_est": row.get("measured_outstanding_at_end_est", ""),
            "frontier_backlog": sim.get("all_request_backlog_at_window_end", ""),
            "frontier_summary_found": bool(sim),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)

    print("workers  rate  valid-load  real-p95  sim-p95  p95-error  real-goodput  sim-goodput")
    print("-" * 83)
    for r in result:
        rp = number(r["polaris_p95_s"])
        sp = number(r["frontier_p95_s"])
        rg = number(r["polaris_slo_goodput_rps"])
        sg = number(r["frontier_slo_goodput_rps"])
        pe = number(r["p95_relative_error_pct"])
        print(f'{r["workers"]:>7} {r["offered_rps"]:>5} '
              f'{"yes" if r["comparable_traffic"] else "NO":>11} '
              f'{display(rp):>9} {display(sp):>8} '
              f'{("—" if pe is None else f"{pe:+.1f}%"):>10} '
              f'{display(rg):>13} {display(sg):>12}')
    print(f"\nWrote {len(result)} comparisons to {args.output}")
    print("'valid-load' means measured submissions >=99% expected, zero HTTP failures, "
          "and no reported dropped k6 iterations.")
    print("A run can fail the latency SLO and still have valid offered-load coverage.")


if __name__ == "__main__":
    main()
