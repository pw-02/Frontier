"""Native Frontier command configuration for MalleServe experiments.

Shares calibrated vLLM v1 / Qwen3-8B / A100 settings with existing runner.
The main difference is execution IN PROCESS for debugging.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_scenario(path: Path) -> dict:
    from examples.malleserve.transient.run_native_scenario import load_scenario as validate
    return validate(path)


def prepare(config: dict) -> tuple[list[str], Path, Path]:
    from examples.malleserve.transient.run_native_scenario import load_scenario
    from examples.malleserve.transient.generate_native_schedule import generate
    # Scenario already validated on file load, but overrides may be applied.
    name = config["experiment"]["name"]
    if not isinstance(name, str) or not name or any(not(c.isalnum() or c in "-_") for c in name):
        raise ValueError("Invalid scenario name")
    resource = config["resources"]
    trace = Path(config["workload"]["trace"])
    if not trace.is_absolute():
        trace = ROOT / trace
    manifest = trace.with_suffix(".manifest.json")
    if not trace.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Missing trace/manifest: {trace}")
    max_tokens = json.loads(manifest.read_text(encoding="utf-8"))["required_frontier_max_tokens"]
    out = ROOT / "outputs/examples/malleserve" / name
    out.mkdir(parents=True, exist_ok=True)
    if resource["mode"] == "fixed":
        count = int(resource["gpus"])
        transient_args = []
    elif resource["mode"] == "transient":
        if int(resource["candidate_nodes"]) != 2:
            raise ValueError("Current rotation generator supports 2 candidate nodes")
        count = int(resource["persistent_gpus"]) + (
            int(resource["candidate_nodes"]) * int(resource["transient_gpus_per_node"]))
        schedule = out / "availability.jsonl"
        events = generate(float(resource["duration_seconds"]), float(resource["node_lifetime_seconds"]))
        schedule.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        transient_args = ["--transient_schedule_file", str(schedule),
             "--transient_persistent_replicas", str(resource["persistent_gpus"]),
             "--transient_replicas_per_node", str(resource["transient_gpus_per_node"]),
             "--transient_startup_seconds", str(resource["startup_seconds"])]
    else:
        raise ValueError("Invalid resource mode")
    args = [
        "--simulation_mode", "online",
        "--sys_arch", "co-location",
        "--cc_backend_config_type", "analytical",
        "--cluster_config_num_replicas", str(count),
        "--cluster_scheduler_config_type", "round_robin",
        "--replica_config_model_name", "Qwen/Qwen3-8B",
        "--replica_config_device", "a100",
        "--replica_config_attn_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--replica_scheduler_config_type", "vllm_v1",
        "--decode_cuda_graph_mode", "piecewise",
        "--vllm_v1_scheduler_config_max_model_len", "17000",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", "2048",
        "--vllm_v1_scheduler_config_block_size", "16",
        "--vllm_v1_scheduler_config_num_blocks_mode", "explicit",
        "--vllm_v1_scheduler_config_num_blocks", "8192",
        "--vllm_v1_scheduler_config_enable_prefix_caching",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", str(trace),
        "--trace_request_generator_config_max_tokens", str(max_tokens),
        *transient_args,
        "--no-random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_atten_input_file",
        "data/profiling/compute/a100/Qwen/Qwen3-8B/attention_calibrated.csv",
        "--random_forrest_execution_time_predictor_config_linear_op_kernel_only_input_file",
        "data/profiling/compute/a100/Qwen/Qwen3-8B/linear_op_kernel_only.csv",
        "--random_forrest_execution_time_predictor_config_atten_kernel_only_input_file",
        "data/profiling/compute/a100/Qwen/Qwen3-8B/attention_kernel_only.csv",
        "--metrics_config_output_dir", str(out),
        "--metrics_config_run_id", name,
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--metrics_config_enable_metrics_ground_truth_trace",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]
    return args, out, manifest
