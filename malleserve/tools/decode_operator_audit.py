#!/usr/bin/env python3
"""Audit Frontier decode predictions against Polaris worker GPU kernel activity.

Cross-scope diagnosis only: no fitting, no runtime calibration, and
attention-named CUDA kernels are NOT equal to Frontier's attention block.
Accepts either the Polaris job ZIP or an unpacked job directory.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def rows_csv(text):
    return list(csv.DictReader(io.StringIO(text)))


def input_files(path):
    """Read named files from an extracted job folder or uploaded Polaris ZIP."""
    suffixes = ("grid/summary.csv", "grid/kernel_analysis_operators.csv",
                "grid/kernel_analysis_kernels.csv", "grid/metadata.json")
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            mapping = {}
            for suffix in suffixes:
                matches = [x for x in names if x.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {suffix} in ZIP; found {matches}")
                mapping[suffix] = archive.read(matches[0]).decode("utf-8")
        return mapping
    mapping = {}
    for suffix in suffixes:
        file = path / suffix
        if not file.is_file():
            raise FileNotFoundError(file)
        mapping[suffix] = file.read_text(encoding="utf-8")
    return mapping


def number(row, key):
    if key not in row or row[key] in ("", None):
        raise ValueError(f"Missing numeric field {key} in {row}")
    return float(row[key])


def csv_out(path, records):
    if not records:
        raise ValueError("No matched rows to report")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-grid", type=Path, required=True,
                        help="decode_investigate_grid.csv from fixed4 reconciliation")
    parser.add_argument("--polaris", type=Path, required=True,
                        help="Polaris job ZIP or unpacked job directory")
    parser.add_argument("--attention-trace", type=Path,
                        help="Optional fresh fixed4 kernel-only synthetic attention trace")
    parser.add_argument("--output-prefix", type=Path, default=Path("decode_operator_audit"))
    args = parser.parse_args()
    files = input_files(args.polaris)
    polaris = rows_csv(files["grid/summary.csv"])
    operators = rows_csv(files["grid/kernel_analysis_operators.csv"])
    kernels = rows_csv(files["grid/kernel_analysis_kernels.csv"])
    metadata = json.loads(files["grid/metadata.json"])
    grid = rows_csv(args.frontier_grid.read_text(encoding="utf-8-sig"))

    f_index = {(int(r["batch_size"]), int(r["context_tokens"])): r for r in grid}
    p_index = {(int(r["batch_size"]), int(r["prompt_len"])): r for r in polaris}
    op_index = {(int(r["batch_size"]), int(r["prompt_len"])): r for r in operators}
    if len(f_index) != len(grid) or len(p_index) != len(polaris):
        raise ValueError("Duplicate batch/context pairs in grid")
    results = []
    for key in sorted(p_index, key=lambda v:(v[1],v[0])):
        if key not in f_index or key not in op_index:
            raise ValueError(f"Missing Frontier/operator row for {key}")
        f, p, op = f_index[key], p_index[key], op_index[key]
        fm = number(f,"frontier_model_ms")
        gpu = number(p,"gpu_kernel_active_ms_per_decode_step")
        attention = number(f,"frontier_attention_ms")
        mlp = sum(number(f,k) for k in
                  ("frontier_mlp_up_ms","frontier_mlp_down_ms","frontier_mlp_act_ms"))
        other = number(f,"frontier_other_model_ms")
        if abs(attention + mlp + other - fm) > .02:
            raise ValueError(f"Frontier component reconciliation failed for {key}; check units")
        named_attn = number(op,"attention_named_incremental_ms_per_step")
        named_gemm = number(op,"gemm_named_incremental_ms_per_step")
        named_other = number(op,"other_unclassified_incremental_ms_per_step")
        ksum = number(op,"kernel_sum_incremental_ms_per_step")
        if abs(named_attn + named_gemm + named_other - ksum) > .03:
            raise ValueError(f"Polaris kernel category sum mismatch for {key}")
        results.append({
            "batch_size":key[0], "context_tokens":key[1],
            "frontier_gpu_model_ms":round(fm,5),
            "polaris_profiled_gpu_busy_ms":round(gpu,5),
            "cross_scope_gap_ms":round(fm-gpu,5),
            "frontier_over_polaris_ratio":round(fm/gpu,4),
            "frontier_attention_block_ms":round(attention,5),
            "frontier_mlp_ms":round(mlp,5),
            "frontier_other_model_ms":round(other,5),
            "polaris_attention_named_ms":round(named_attn,5),
            "polaris_gemm_named_ms":round(named_gemm,5),
            "polaris_other_named_ms":round(named_other,5),
            "polaris_kernel_sum_ms":round(ksum,5),
            "gpu_kernel_sum_minus_busy_ms":round(ksum-gpu,5),
            "scope_warning":"Frontier attention block includes projections and other ops; named-kernel groups are not equivalent",
        })
    suffix = args.output_prefix
    suffix.parent.mkdir(parents=True, exist_ok=True)
    csv_out(suffix.with_name(suffix.name+"_comparison.csv"),results)

    # Rank kernel contributions in the real workload: GEMM types and
    # attention kernels should be inspected together, not equated with MLP.
    top = {}
    for r in kernels:
        key = (int(r["batch_size"]), int(r["prompt_len"]))
        if key not in p_index:
            continue
        top.setdefault(key, []).append(r)
    selected = []
    for key in sorted(top, key=lambda v:(v[1],v[0])):
        for rank, r in enumerate(sorted(top[key],
                         key=lambda r: -float(r["incremental_kernel_duration_ms_per_step"]))[:12],1):
            selected.append({
                "batch_size":key[0], "context_tokens":key[1],"rank":rank,
                "category":r["heuristic_category"],"kernel_name":r["kernel_name"],
                "incremental_duration_ms":r["incremental_kernel_duration_ms_per_step"],
            })
    csv_out(suffix.with_name(suffix.name+"_top_kernels.csv"),selected)

    trace_summary = {}
    if args.attention_trace:
        family, source, origins, n = Counter(), Counter(), Counter(), 0
        scales, per_layer = [], []
        with args.attention_trace.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                ev=json.loads(line)
                lookup=ev.get("lookup") or {}
                family[str(lookup.get("measurement_family","unknown"))] += 1
                source[str(lookup.get("prediction_source","unknown"))] += 1
                origins[str(ev.get("origin","unknown"))] += 1
                scales.append(float(ev.get("calibration_scale", 1)))
                if ev.get("scaled_attention_decode_ms_per_layer") is not None:
                    per_layer.append(float(ev["scaled_attention_decode_ms_per_layer"]))
                n += 1
        trace_summary = {
            "trace_records":n, "measurement_families":dict(family),
            "lookup_sources":dict(source), "origins":dict(origins),
            "calibration_scales":sorted(set(round(s,8) for s in scales)),
            "median_attention_decode_ms_per_layer":statistics.median(per_layer) if per_layer else None,
        }
        if set(family) != {"kernel_only"}:
            trace_summary["warning"] = "NON-KERNEL_ONLY family present; do not use trace for matched graph analysis"

    report={
        "matching_grid_points":len(results),
        "gap_mean_ms":statistics.mean(r["cross_scope_gap_ms"] for r in results),
        "gap_min_ms":min(r["cross_scope_gap_ms"] for r in results),
        "gap_max_ms":max(r["cross_scope_gap_ms"] for r in results),
        "polaris_metadata":{k:v for k,v in metadata.items()
                            if k in ("vllm_version","attention_backend","dtype",
                                     "effective_cudagraph_mode","effective_compilation_mode",
                                     "measurement","warning")},
        "frontier_attention_includes":"attention decode+prefill, pre/post projections, RoPE, KV save, norms",
        "interpretation_limits":[
            "Frontier attention block cannot be equated with attention-named FlashAttention kernel time.",
            "GEMM-named CUDA kernels include QKV and attention output projections, not just MLP.",
            "GPU kernel-active union includes sampling and non-model GPU work; profiler perturbs execution.",
            "Paired 129-token vLLM benchmark averages growing contexts; Frontier grid uses fixed context.",
            "No timing coefficient or causal attribution is established by this cross-scope comparison.",
        ],
        "attention_trace":trace_summary,
    }
    summary = suffix.with_name(suffix.name+"_report.json")
    summary.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"Matched {len(results)} grid points; mean cross-scope gap {report['gap_mean_ms']:.3f} ms")
    print(f"Outputs: {suffix.name}_comparison.csv, _top_kernels.csv, _report.json")
    print("CAUTION: For true operator attribution, match GPU projection kernels to Frontier")
    print("attention pre/post projection plus MLP and check layer counts, not kernel names alone.")


if __name__=="__main__":
    main()
