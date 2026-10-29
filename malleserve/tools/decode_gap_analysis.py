"""Analyze independently measured Polaris decode timings without mutating Frontier.

Reads Frontier's 15-point KERNEL_ONLY grid and optional Polaris metadata.json.
Reports per-operator decomposition, gaps, and exploratory affine fits. The
Polaris full-engine wall time is NOT a GPU-only profiling ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(rows):
    required = (
        "batch_size", "context_tokens", "measurement_family",
        "frontier_model_ms", "polaris_engine_wall_ms",
        "frontier_attention_ms", "frontier_mlp_up_ms",
        "frontier_mlp_down_ms", "frontier_mlp_act_ms",
        "frontier_other_model_ms",
    )
    if not rows:
        raise ValueError("Empty grid")
    for row in rows:
        missing = [key for key in required if key not in row or row[key] == ""]
        if missing:
            raise ValueError(f"Missing decomposition fields {missing}; rerun updated probe")
        if row["measurement_family"] != "kernel_only":
            raise ValueError("Refusing to mix predictor measurement families")
    results = []
    for row in rows:
        predicted = float(row["frontier_model_ms"])
        observed = float(row["polaris_engine_wall_ms"])
        parts = {key: float(row[key]) for key in (
            "frontier_attention_ms", "frontier_mlp_up_ms",
            "frontier_mlp_down_ms", "frontier_mlp_act_ms",
            "frontier_other_model_ms")}
        if abs(sum(parts.values()) - predicted) > 0.002:
            raise ValueError("Operator decomposition does not sum to model time")
        results.append({
            "batch_size": int(row["batch_size"]),
            "context_tokens": int(row["context_tokens"]),
            "frontier_model_ms": predicted,
            "polaris_engine_wall_ms": observed,
            "gap_ms": predicted - observed,
            "frontier_over_polaris_ratio": predicted / observed,
            **parts,
        })
    return results


def fit_summary(rows):
    # Descriptive only: observations are *full-engine wall* and so not a
    # valid target for recalibrating GPU kernel model parameters directly.
    xs = [r["frontier_model_ms"] for r in rows]
    ys = [r["polaris_engine_wall_ms"] for r in rows]
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator if denominator else None
    intercept = y_mean - slope * x_mean if slope is not None else None
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)] if slope is not None else []
    groups = defaultdict(list)
    for r in rows:
        groups[r["batch_size"]].append(r["gap_ms"])
    return {
        "n": len(rows),
        "gap_min_ms": min(r["gap_ms"] for r in rows),
        "gap_max_ms": max(r["gap_ms"] for r in rows),
        "gap_mean_ms": statistics.mean(r["gap_ms"] for r in rows),
        "gap_by_batch_mean_ms": {str(k): statistics.mean(v) for k, v in sorted(groups.items())},
        "descriptive_affine_wall_from_model": {
            "slope": slope, "intercept_ms": intercept,
            "max_abs_residual_ms": max(map(abs, residuals)) if residuals else None,
            "warning": "Exploratory cross-scope fit ONLY; do not deploy as GPU kernel calibration",
        },
    }


def audit_metadata(data):
    args = data.get("arguments", {})
    return {
        "vllm_version": data.get("vllm_version"),
        "gpu_info": data.get("gpu_info"),
        "model": args.get("model"),
        "backend": args.get("backend"),
        "gpu_memory_utilization": args.get("gpu_memory_utilization"),
        "effective_cudagraph_mode": data.get("effective_cudagraph_mode"),
        "effective_compilation_mode": data.get("effective_compilation_mode"),
        "effective_max_num_batched_tokens": data.get("effective_max_num_batched_tokens"),
        "effective_max_num_seqs": data.get("effective_max_num_seqs"),
        "effective_enable_prefix_caching": data.get("effective_enable_prefix_caching"),
        "notes": (
            "Compare with Frontier resources=fixed4: Qwen3-8B, A100, "
            "CUDA graph FULL synthetic assumption, profiling attention backend "
            "and tensor parallelism. Polaris is a single-GPU offline full-engine "
            "wall-time estimate; do not assume matching online scheduling or "
            "identical measurement boundaries."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--polaris-metadata", type=Path)
    ap.add_argument("--output-prefix", type=Path, default=Path("decode_gap_analysis"))
    args = ap.parse_args()
    with args.grid.open(newline="", encoding="utf-8") as stream:
        rows = analyze(list(csv.DictReader(stream)))
    result = {"fit": fit_summary(rows), "measurement_caveat": (
        "Frontier GPU model time vs Polaris paired full-engine wall; "
        "benchmark averages 128 decode steps at increasing KV contexts."
    )}
    if args.polaris_metadata:
        result["polaris_configuration"] = audit_metadata(json.loads(args.polaris_metadata.read_text()))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail = Path(str(args.output_prefix) + "_operators.csv")
    with detail.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = Path(str(args.output_prefix) + "_summary.json")
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Saved", detail, summary)


if __name__ == "__main__":
    main()
