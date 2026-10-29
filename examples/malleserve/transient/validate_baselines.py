#!/usr/bin/env python3
"""Fail closed when the transient macro fixed controls disagree with native Frontier.

This checks whether the processor-sharing proxy can at least reproduce
ordinary fixed-worker inference under the same trace. Passing is necessary
but never sufficient to claim accurate transient-churn predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

FIELD_MAP = {
    "latency_p95_s": "latency_p95_s_for_completed_measured",
    "latency_p99_s": "latency_p99_s_for_completed_measured",
    "slo_goodput_rps": "measured_slo_goodput_rps_allowing_drain",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(macro: dict, native: dict, max_error: float) -> list[str]:
    problems = []
    for proxy_key, native_key in FIELD_MAP.items():
        p, n = macro.get(proxy_key), native.get(native_key)
        if p is None or n is None:
            problems.append(f"{proxy_key}: missing comparison value")
            continue
        p, n = float(p), float(n)
        if not math.isfinite(p) or not math.isfinite(n):
            problems.append(f"{proxy_key}: nonfinite comparison value")
            continue
        diff = abs(p - n)
        tolerance = max(0.1, max_error * max(abs(n), 1e-9))
        print(f"  {proxy_key:20} proxy={p:10.4f} native={n:10.4f}"
              f" abs_diff={diff:8.4f}, tolerance={tolerance:8.4f}")
        if diff > tolerance:
            problems.append(
                f"{proxy_key}: difference {diff:g} > tolerance {tolerance:g}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--macro-scenario", type=Path, required=True)
    p.add_argument("--native-fixed-4", type=Path, required=True,
                   help="Native Frontier malleserve_summary.json for 4-worker same trace")
    p.add_argument("--native-fixed-8", type=Path, required=True,
                   help="Native Frontier malleserve_summary.json for 8-worker same trace")
    p.add_argument("--max-relative-error", type=float, default=0.20)
    a = p.parse_args()
    if not 0 <= a.max_relative_error < 1:
        p.error("--max-relative-error must be between 0 and 1")
    errors = []
    for name, native_path in (
        ("fixed_persistent", a.native_fixed_4),
        ("fixed_stable", a.native_fixed_8),
    ):
        print(f"Checking {name}:")
        proxy = read(a.macro_scenario / name / "summary.json")
        native = read(native_path)
        if proxy["measured_arrivals"] != native["measured_arrivals"]:
            errors.append(f"{name}: measured request count mismatch")
        for error in compare(proxy, native, a.max_relative_error):
            errors.append(f"{name}: {error}")
    if errors:
        print("\nNOT VALIDATED against native fixed-worker Frontier:")
        for error in errors:
            print(f"  - {error}")
        print("Do not use macro quantitative conclusions. Implement native "
              "dynamic replica lifecycle and rerun.", file=sys.stderr)
        sys.exit(2)
    print("\nFixed controls match within selected tolerance. This is not "
          "proof that churn/recovery predictions are accurate.")


if __name__ == "__main__":
    main()
