"""Direct 15-case Frontier execution predictor probe (no event replay).

Constructs explicit homogeneous *decode* Batches and uses the same trained
predictor/configuration as the native MalleServe simulation. Polaris measures
engine-wall, not GPU-only time; both are retained separately.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

BATCH_SIZES = (1, 2, 4, 8, 16)
CONTEXTS = (128, 1024, 4096)


def probe(predictor, model_layers: int, polaris: dict, before_predict=None):
    from frontier.entities import Batch, Request
    from frontier.types import ClusterType
    rows = []
    for context in CONTEXTS:
        for size in BATCH_SIZES:
            requests = [
                Request(arrived_at=0.0, num_prefill_tokens=context,
                        num_decode_tokens=32, num_processed_tokens=context + 1)
                for _ in range(size)
            ]
            for request in requests:
                request._is_prefill_complete = True
            batch = Batch(replica_id=0, requests=requests,
                          num_tokens=[1] * size, is_moe=False,
                          num_context_tokens=[context] * size,
                          request_is_decoding=[True] * size)
            # Homogeneous, pure-decode synthetic probes represent a captured
            # CUDA-graph step; annotate this assumption explicitly. Real captured
            # metadata is used separately by decode_predictor_reconcile.
            from frontier.entities.batch import DecodeCudaGraphMetadata
            batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
                config_mode="FULL", runtime_mode="FULL", capture_hit=True,
                is_mixed_batch=False, original_total_tokens=size,
                padded_total_tokens=size, original_decode_batch_size=size,
                padded_decode_batch_size=size,
            )
            family = before_predict(batch) if before_predict is not None else (
                predictor._measurement_family_name(
                    predictor._activate_measurement_type_for_batch(batch)
                )
            )
            if family != "kernel_only":
                raise RuntimeError(f"Decode predictor selected {family}, expected kernel_only")
            stage = predictor.predict_stage_execution_time(
                batch, stage_id=0, cluster_type=ClusterType.MONOLITHIC,
                num_layers=model_layers, layer_id=0,
            )
            model_ms = float(stage.model_time_ms)
            total_ms = float(stage.total_time * 1000)
            host_ms = total_ms - model_ms
            rows.append({
                "batch_size": size, "context_tokens": context,
                "measurement_family": family,
                "polaris_engine_wall_ms": polaris[(size, context)],
                "frontier_model_ms": round(model_ms, 6),
                "frontier_modelled_host_ms": round(host_ms, 6),
                "frontier_total_stage_ms": round(total_ms, 6),
                "frontier_minus_polaris_wall_ms": round(total_ms - polaris[(size, context)], 6),
                "comparison_scope": "GPU_MODEL_VS_FULL_ENGINE_WALL_NOT_IDENTICAL",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polaris-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("decode_predictor_matched.csv"))
    parser.add_argument("--config-name", default="config")
    args = parser.parse_args()
    from malleserve.tools.decode_predictor_diagnostic import read_polaris
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from malleserve.scenario import prepare
    from frontier.config import SimulationConfig
    from frontier.simulator import Simulator
    from frontier.types import ClusterType
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "malleserve/conf"), version_base=None):
        scenario = OmegaConf.to_container(compose(config_name=args.config_name,
            overrides=["resources=fixed8", "policy=fixed"]), resolve=True)
    # Reuse exactly the normal MalleServe vLLM/Qwen3/A100 predictor
    # configuration; do not run simulation events.
    frontier_args, _, _ = prepare(scenario)
    prior = sys.argv
    try:
        sys.argv = ["frontier.main", *frontier_args]
        config = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = prior
    simulator = Simulator(config)
    predictor = simulator._predictors[ClusterType.MONOLITHIC]
    cluster = config.get_clusters()[ClusterType.MONOLITHIC]
    layers = cluster.replica_config.model_config.num_layers
    polaris = read_polaris(args.polaris_summary)
    rows = probe(predictor, layers, polaris)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"batch={row['batch_size']:2} context={row['context_tokens']:4} "
              f"model={row['frontier_model_ms']:.3f}ms "
              f"host={row['frontier_modelled_host_ms']:.3f}ms "
              f"total={row['frontier_total_stage_ms']:.3f}ms "
              f"Polaris-wall={row['polaris_engine_wall_ms']:.3f}ms")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
