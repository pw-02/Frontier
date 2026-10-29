#!/usr/bin/env python3
"""Extract a low-load empirical service-time proxy from a real Frontier run.

This is deliberately NOT a GPU-work predictor. request_model_execution_time
is elapsed model-execution time seen by a request (which can overlap other
requests during continuous batching). The transient macro model that consumes
this proxy is exploratory and cannot replace Frontier's vLLM event simulator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def extract(trace_csv: Path, request_metrics_csv: Path) -> dict:
    with trace_csv.open(newline="", encoding="utf-8") as f:
        trace = list(csv.DictReader(f))
    with request_metrics_csv.open(newline="", encoding="utf-8") as f:
        metrics = list(csv.DictReader(f))
    if not trace or not metrics:
        raise ValueError("Trace and request metrics must be nonempty")
    if "source_index" not in trace[0]:
        raise ValueError("Mooncake trace must contain source_index")
    samples = []
    seen_ids = set()
    for row in metrics:
        ident = int(row["Request Id"])
        if ident in seen_ids or ident < 0 or ident >= len(trace):
            raise ValueError(f"Duplicate/out-of-range Frontier request ID {ident}")
        seen_ids.add(ident)
        tr = trace[ident]
        prompt = int(tr["num_prefill_tokens"])
        output = int(tr["num_decode_tokens"])
        if (int(row["request_num_prefill_tokens"]) != prompt
                or int(row["request_num_decode_tokens"]) != output):
            raise ValueError(f"Request ID {ident} does not match trace tokens")
        milliseconds = float(row["request_model_execution_time"])
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            raise ValueError(f"Invalid model execution time for request {ident}")
        samples.append({
            "source_index": int(tr["source_index"]),
            "input_tokens": prompt,
            "output_tokens": output,
            "proxy_service_s": milliseconds / 1000,
        })
    if len(seen_ids) != len(trace):
        raise ValueError(
            f"Metrics cover {len(seen_ids)} requests, trace has {len(trace)}; "
            "use a complete calibrated Frontier run"
        )
    return {
        "schema_version": 1,
        "method": "Frontier request_model_execution_time_ms / 1000",
        "warning": (
            "Request elapsed model-execution duration is not isolated GPU work. "
            "Macro processor-sharing approximations are NOT native Frontier "
            "vLLM continuous batching or validated capacity predictions."
        ),
        "trace_csv": str(trace_csv),
        "request_metrics_csv": str(request_metrics_csv),
        "samples": samples,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--request-metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    obj = extract(a.trace, a.request_metrics)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(obj['samples'])} low-load proxy samples to {a.output}")
    print("WARNING: This is a macro simulation input, NOT GPU-isolated compute.")


if __name__ == "__main__":
    main()
