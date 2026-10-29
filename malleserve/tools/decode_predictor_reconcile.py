"""Reconcile captured live decode-batch predictions and the synthetic Polaris grid.

Runs with one fixed-4 MalleServe predictor instance, exactly as configured
by the regular native simulation. Input is the JSONL from the opt-in
FRONTIER_DECODE_REPLAY_PATH hook. No benchmark coefficients are modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from malleserve.tools.decode_predictor_matched import BATCH_SIZES, CONTEXTS, probe
from malleserve.tools.decode_predictor_diagnostic import read_polaris


def build_batch(contexts, *, prompts=None, num_tokens=None, decoding=None):
    from frontier.entities import Batch, Request
    contexts = [int(value) for value in contexts]
    if not contexts or any(value <= 0 for value in contexts):
        raise ValueError("Decode contexts must be positive")
    size = len(contexts)
    prompts = list(contexts if prompts is None else prompts)
    num_tokens = list([1] * size if num_tokens is None else num_tokens)
    decoding = list([True] * size if decoding is None else decoding)
    if not (len(prompts) == len(num_tokens) == len(decoding) == size):
        raise ValueError("Batch fields must have equal lengths")
    if any(value < 1 or value > context for value, context in zip(prompts, contexts)):
        raise ValueError("Invalid prompt length for context")
    requests = [
        Request(arrived_at=0.0, num_prefill_tokens=int(prompt),
                num_decode_tokens=max(32, context - prompt + 16),
                num_processed_tokens=context + 1)
        for prompt, context in zip(prompts, contexts)
    ]
    for request, is_decode in zip(requests, decoding):
        request._is_prefill_complete = bool(is_decode)
    return Batch(replica_id=0, requests=requests,
                 num_tokens=[int(n) for n in num_tokens],
                 is_moe=False, num_context_tokens=contexts,
                 request_is_decoding=[bool(v) for v in decoding])


def select_live_decode_family(predictor, batch):
    """Select exactly the operator family used by a live CUDA-graph decode step.

    Use the predictor's batch-aware selector, not a private table assignment;
    fail loudly if the synthetic batch is missing graph-mode metadata.
    """
    from frontier.types import MeasurementType
    selected = predictor._activate_measurement_type_for_batch(batch)
    family = predictor._measurement_family_name(selected)
    if selected != MeasurementType.KERNEL_ONLY or family != "kernel_only":
        raise RuntimeError(
            "Synthetic decode batch selected timing family "
            f"{selected!s} ({family}), expected KERNEL_ONLY (kernel_only). "
            "Check CUDA-graph batch metadata and scheduler configuration."
        )
    return family


def components(value):
    names = {
        "model_ms": "model_time_ms",
        "attention_ms": "attention_time",
        "mlp_up_ms": "mlp_layer_up_proj_execution_time",
        "mlp_down_ms": "mlp_layer_down_proj_execution_time",
        "mlp_act_ms": "mlp_layer_act_execution_time",
        "forward_launch_ms": "forward_launch_time",
    }
    result = {name: float(getattr(value, attr)) for name, attr in names.items()}
    result["total_ms"] = float(value.total_time * 1000.0)
    result["host_ms"] = result["total_ms"] - result["model_ms"]
    return result


def reconcile(predictor, captures, *, limit=30, tolerance_ms=0.001):
    from frontier.types import ClusterType
    rows = []
    with Path(captures).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            contexts = item["context_tokens"]
            batch = build_batch(contexts, num_tokens=item["num_tokens"],
                                decoding=item["request_is_decoding"])
            family = select_live_decode_family(predictor, batch)
            current = predictor.predict_stage_execution_time(
                batch, stage_id=int(item["stage_id"]),
                cluster_type=ClusterType.MONOLITHIC,
                num_layers=int(item["num_layers"]), layer_id=int(item["layer_id"]))
            reproduced = components(current)
            original = item["original"]
            delta = reproduced["model_ms"] - float(original["model_ms"])
            row = {
                "batch_id": item["batch_id"],
                "measurement_family": family,
                "size": len(contexts),
                "context_tokens": json.dumps(contexts),
                "captured_model_ms": original["model_ms"],
                "reconstructed_model_ms": reproduced["model_ms"],
                "delta_ms": round(delta, 6),
                "matched_within_tolerance": abs(delta) <= tolerance_ms,
            }
            for field in ("attention_ms", "mlp_up_ms", "mlp_down_ms",
                          "mlp_act_ms", "forward_launch_ms", "host_ms"):
                row["original_" + field] = original.get(field)
                row["reconstructed_" + field] = reproduced[field]
                if original.get(field) is not None:
                    row["delta_" + field] = round(
                        reproduced[field] - float(original[field]), 6)
            rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError("No captured decode batches")
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_predictor(*, resource="fixed4"):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from malleserve.scenario import prepare
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.types import ClusterType
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "malleserve/conf"),
                               version_base=None):
        scenario = OmegaConf.to_container(compose(
            config_name="config", overrides=[
                "resources=" + resource,
                "policy=fixed",
                "experiment.name=decode_reconcile_config_" + resource,
            ]), resolve=True)
    arguments, _, _ = prepare(scenario)
    old_argv = sys.argv
    try:
        sys.argv = ["frontier.main", *arguments]
        config = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = old_argv
    simulator = Simulator(config)
    predictor = simulator._predictors[ClusterType.MONOLITHIC]
    num_layers = predictor._num_layers_per_pipeline_stage
    return predictor, int(num_layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured", type=Path, required=True,
                        help="decode_same_batch.jsonl from live scheduler probe")
    parser.add_argument("--polaris-summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("decode_reconcile"))
    parser.add_argument("--attention-trace", type=Path, default=None,
                        help="Trace attention features/scales for reconstructed and grid batches")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--tolerance-ms", type=float, default=0.001)
    args = parser.parse_args()
    if args.limit <= 0 or args.tolerance_ms < 0:
        parser.error("Invalid limit/tolerance")
    import os
    if args.attention_trace is not None:
        os.environ["FRONTIER_ATTENTION_TRACE_PATH"] = str(args.attention_trace)
        os.environ["FRONTIER_ATTENTION_TRACE_ORIGIN"] = "reconstructed"
    predictor, layers = build_predictor(resource="fixed4")
    rows = reconcile(predictor, args.captured, limit=args.limit,
                     tolerance_ms=args.tolerance_ms)
    path = Path(str(args.output_prefix) + "_captured.csv")
    write_csv(path, rows)
    mismatch = sum(not row["matched_within_tolerance"] for row in rows)
    print(f"Captured batch reconstruction: {len(rows)} cases; "
          f"{mismatch} mismatch(es); saved {path}")
    # Same predictor instance, same fixed4 configuration, all 15 grid cases.
    if args.attention_trace is not None:
        os.environ["FRONTIER_ATTENTION_TRACE_ORIGIN"] = "synthetic_grid_fixed4"
    grid = probe(predictor, layers, read_polaris(args.polaris_summary),
                 before_predict=lambda batch: select_live_decode_family(predictor, batch))
    grid_path = Path(str(args.output_prefix) + "_grid.csv")
    write_csv(grid_path, grid)
    print(f"15-point fixed4 synthetic decode grid saved {grid_path}")
    if mismatch:
        print("WARNING: Reconstructed batches differ from captured live batches.")
        print("The captured probe does not contain all scheduler/request metadata; "
              "do not treat synthetic grid as a validated reproduction.")
    else:
        print("Captured model times matched under the reconstructed batch fields. "
              "This tests those fields, not every hidden scheduler state.")


if __name__ == "__main__":
    main()
