"""Predictor-only decode sensitivity check for opt-in RoPE and norm multipliers.

Uses one fixed4 Qwen3 A100 predictor. Does not alter training artifacts or
benchmark coefficients. No full simulator run is performed.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from malleserve.tools.decode_predictor_reconcile import (
    build_batch, build_predictor, select_live_decode_family,
)


SCALES = (
    "decode_phase_attn_rope_calibration_scale",
    "decode_phase_input_layernorm_calibration_scale",
    "decode_phase_post_attention_layernorm_calibration_scale",
)
CASES = {
    "baseline": (None, None, None),
    "rope_half": (0.5, None, None),
    "norm_half": (None, 0.5, 0.5),
    "combined_half": (0.5, 0.5, 0.5),
}


def run(predictor, layers):
    from frontier.entities.batch import DecodeCudaGraphMetadata
    from frontier.types import ClusterType

    original = tuple(getattr(predictor._config, name) for name in SCALES)
    records = []
    try:
        for context in (128, 1024, 4096):
            for batch_size in (1, 2, 4, 8, 16):
                batch = build_batch([context] * batch_size)
                batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
                    config_mode="FULL", runtime_mode="FULL", capture_hit=True,
                    is_mixed_batch=False, original_total_tokens=batch_size,
                    padded_total_tokens=batch_size,
                    original_decode_batch_size=batch_size,
                    padded_decode_batch_size=batch_size,
                )
                values = {}
                for case_name, scales in CASES.items():
                    for name, scale in zip(SCALES, scales):
                        setattr(predictor._config, name, scale)
                    select_live_decode_family(predictor, batch)
                    stage = predictor.predict_stage_execution_time(
                        batch, stage_id=0, cluster_type=ClusterType.MONOLITHIC,
                        num_layers=layers, layer_id=0,
                    )
                    values[case_name] = stage
                baseline = values["baseline"]
                for case_name, scales in CASES.items():
                    value = values[case_name]
                    expected = float(baseline.model_time_ms)
                    for scale, attr in zip(scales, (
                        "attention_rope_execution_time",
                        "attn_norm_time", "mlp_norm_time",
                    )):
                        if scale is not None:
                            expected += (scale - 1) * float(getattr(baseline, attr))
                    if not math.isclose(
                        float(value.model_time_ms), expected,
                        abs_tol=1e-6, rel_tol=1e-8,
                    ):
                        raise RuntimeError(
                            f"Sensitivity mismatch b={batch_size} c={context} "
                            f"case={case_name}: actual={value.model_time_ms} "
                            f"expected={expected}"
                        )
                    records.append({
                        "case": case_name,
                        "batch_size": batch_size,
                        "context_tokens": context,
                        "num_layers": layers,
                        "model_ms": float(value.model_time_ms),
                        "baseline_model_ms": float(baseline.model_time_ms),
                        "change_ms": float(value.model_time_ms - baseline.model_time_ms),
                        "attention_rope_ms_baseline": float(baseline.attention_rope_execution_time),
                        "input_norm_ms_baseline": float(baseline.attn_norm_time),
                        "post_attention_norm_ms_baseline": float(baseline.mlp_norm_time),
                        **dict(zip(SCALES, scales)),
                    })
    finally:
        for name, previous in zip(SCALES, original):
            setattr(predictor._config, name, previous)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/malleserve/decode_calibration_sensitivity.csv"))
    args = parser.parse_args()
    predictor, layers = build_predictor(resource="fixed4")
    records = run(predictor, layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for row in records:
        if row["batch_size"] == 4 and row["context_tokens"] == 1024:
            print(f"{row['case']:14} {row['model_ms']:8.3f}ms "
                  f"delta={row['change_ms']:+.3f}ms")
    print(f"Saved {len(records)} rows: {args.output}")


if __name__ == "__main__":
    main()
