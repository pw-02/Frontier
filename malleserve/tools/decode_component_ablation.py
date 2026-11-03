"""Fixed4 Qwen3 decode operator accounting and controlled predictor-only ablation.

Does not run simulation, mutate stored profiling tables, fit coefficients, or
change defaults. Treats GPU-worker profiled time as a DIFFERENT measurement
scope from the Frontier model-time prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from malleserve.tools.decode_predictor_reconcile import (
    build_batch, build_predictor, select_live_decode_family,
)

BATCHES = (1, 2, 4, 8, 16)
CONTEXTS = (128, 1024, 4096)

# All of these names are scalar components of one physical layer.
OPS = (
    "attention_pre_proj_time",
    "attention_post_proj_time",
    "attention_decode_execution_time",
    "attention_prefill_execution_time",
    "attention_rope_execution_time",
    "attention_kv_cache_save_execution_time",
    "attn_norm_time",
    "mlp_layer_up_proj_execution_time",
    "mlp_layer_down_proj_execution_time",
    "mlp_layer_act_execution_time",
    "mlp_norm_time",
    "add_time",
)


def predict(predictor, batch, layers):
    from frontier.types import ClusterType
    select_live_decode_family(predictor, batch)
    return predictor.predict_stage_execution_time(
        batch, stage_id=0, cluster_type=ClusterType.MONOLITHIC,
        num_layers=layers, layer_id=0
    )


def finite_table_info(predictor, model_name):
    table = getattr(predictor, "_predictions", {}).get(model_name)
    models = getattr(predictor, "_models", {})
    model = models.get(model_name) if hasattr(models, "get") else None
    if table is None:
        return {"operation": model_name, "found": False}
    info = {
        "operation": model_name, "found": True,
        "table_type": type(table).__name__,
        "table_entries": len(table) if hasattr(table, "__len__") else None,
        "on_demand": bool(table.get("_on_demand_prediction")) if hasattr(table, "get") else None,
        "trained_fallback_type": type(model).__name__ if model is not None else None,
    }
    # Do not serialize models or table contents, which can be very large.
    return info


def run_grid(predictor, layers, *, batching_fraction):
    from frontier.entities.batch import DecodeCudaGraphMetadata
    records = []
    lookups = []
    original = predictor._attention_decode_batching_overhead_fraction
    try:
        for context in CONTEXTS:
            for size in BATCHES:
                batch = build_batch([context] * size)
                batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
                    config_mode="FULL", runtime_mode="FULL", capture_hit=True,
                    is_mixed_batch=False, original_total_tokens=size,
                    padded_total_tokens=size, original_decode_batch_size=size,
                    padded_decode_batch_size=size,
                )
                predictor._attention_decode_batching_overhead_fraction = original
                baseline = predict(predictor, batch, layers)
                if baseline.num_layers != layers:
                    raise RuntimeError("Layer-count mismatch")
                first = baseline.layer_execution_times[0]
                op_values = {key: float(getattr(first, key)) for key in OPS}
                # Physical layers' scalar costs must sum to their stage aggregates.
                for key in OPS:
                    expected = math.fsum(float(getattr(layer, key))
                                         for layer in baseline.layer_execution_times)
                    observed = float(getattr(baseline, key))
                    if not math.isclose(expected, observed, abs_tol=1e-7, rel_tol=1e-8):
                        raise RuntimeError(f"Stage layer aggregation mismatch for {key}: {expected} vs {observed}")
                model = float(baseline.model_time_ms)
                per_layer_sum = math.fsum(op_values.values())
                stage_breakdown_sum = per_layer_sum * layers
                remainder = model - stage_breakdown_sum
                # If these are nonzero they must be investigated, not silently ignored.
                predictor._attention_decode_batching_overhead_fraction = 0.0
                without = predict(predictor, batch, layers)
                delta = model - float(without.model_time_ms)
                expected_delta = (
                    float(baseline.attention_decode_execution_time)
                    * batching_fraction / (1.0 + batching_fraction)
                    if size > 1 else 0.0
                )
                if not math.isclose(delta, expected_delta, abs_tol=1e-6, rel_tol=1e-6):
                    raise RuntimeError(
                        f"Batching ablation inconsistent b={size} c={context}: "
                        f"actual={delta}, expected={expected_delta}; "
                        "prediction caches/config may be state-dependent"
                    )
                record = {
                    "batch_size": size,
                    "context_tokens": context,
                    "measurement_family": "kernel_only",
                    "num_layers": layers,
                    "model_ms": model,
                    "operator_sum_ms": stage_breakdown_sum,
                    "unattributed_model_ms": remainder,
                    "batching_fraction": batching_fraction,
                    "batching_adjustment_ms": delta,
                    "model_ms_no_batching_adjustment": float(without.model_time_ms),
                }
                for name, value in op_values.items():
                    record[name + "_per_layer_ms"] = value
                    record[name + "_stage_ms"] = value * layers
                records.append(record)
    finally:
        predictor._attention_decode_batching_overhead_fraction = original
    names = (
        "attn_pre_proj", "attn_post_proj", "attn_decode",
        "mlp_up_proj", "mlp_down_proj", "mlp_act",
        "input_layernorm", "post_attention_layernorm", "add",
    )
    lookups = [finite_table_info(predictor, name) for name in names]
    return records, lookups


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("outputs/malleserve/decode_component_ablation"))
    parser.add_argument("--frontier-grid", type=Path,
                        help="Optional prior decode_operator_grid.csv to verify totals")
    args = parser.parse_args()
    predictor, layers = build_predictor(resource="fixed4")
    fraction = float(predictor._attention_decode_batching_overhead_fraction)
    records, lookups = run_grid(predictor, layers, batching_fraction=fraction)
    if args.frontier_grid:
        with args.frontier_grid.open(newline="", encoding="utf-8") as stream:
            earlier = {(int(r["batch_size"]), int(r["context_tokens"])): r
                       for r in csv.DictReader(stream)}
        for row in records:
            key = row["batch_size"], row["context_tokens"]
            if key not in earlier:
                raise ValueError(f"Missing prior grid point {key}")
            old = float(earlier[key]["frontier_model_ms"])
            if not math.isclose(row["model_ms"], old, abs_tol=0.001, rel_tol=0):
                raise RuntimeError(f"Predictor baseline drift at {key}: {row['model_ms']} vs {old}")
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_name(prefix.name + "_operators.csv")
    report_path = prefix.with_name(prefix.name + "_report.json")
    write_csv(csv_path, records)
    summary = {
        "num_layers": layers,
        "points": len(records),
        "configured_decode_batching_fraction": fraction,
        "lookup_tables": lookups,
        "largest_batching_adjustment_ms": max(r["batching_adjustment_ms"] for r in records),
        "max_unattributed_model_ms_abs": max(abs(r["unattributed_model_ms"]) for r in records),
        "interpretation": [
            "All per-layer operator times are Frontier predictions, not Polaris measurements.",
            "No batching adjustment is applied at batch size 1.",
            "Ablation is on an in-memory predictor only, restored after each measurement.",
            "Subtracting an operator contribution is accounting, not a measured simulator counterfactual.",
            "Table presence does not identify whether an individual lookup came from measured data or finite trained predictions.",
        ],
    }
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Layer count: {layers}; points: {len(records)}; batching fraction: {fraction:g}")
    print(" batch  ctx  model(ms)  no-batch-extra(ms)  delta(ms)  residual(ms)")
    for row in records:
        print(f"{row['batch_size']:>6} {row['context_tokens']:>5} "
              f"{row['model_ms']:>10.3f} "
              f"{row['model_ms_no_batching_adjustment']:>19.3f} "
              f"{row['batching_adjustment_ms']:>10.3f} "
              f"{row['unattributed_model_ms']:>12.3f}")
    print(f"Saved {csv_path} and {report_path}")


if __name__ == "__main__":
    main()
