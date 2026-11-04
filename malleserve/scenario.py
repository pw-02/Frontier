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


def resolve_routing_policy(sched: dict) -> tuple[str, float]:
    """Readable routing names; legacy native settings work when alias is null."""
    alias = sched.get("routing_policy")
    if alias is None:
        return str(sched["cluster"]), float(sched.get("cache_weight", 1.0))
    policies = {
        "rr": ("round_robin", 1.0),
        "wa": ("kv_aware", 0.0),
        "cwa": ("kv_aware", 1.0),
    }
    if alias not in policies:
        raise ValueError(f"Invalid routing_policy {alias!r}; choose rr, wa or cwa")
    return policies[alias]


def prepare(config: dict) -> tuple[list[str], Path, Path]:
    from malleserve.tools.generate_native_schedule import generate
    from malleserve.tools.generate_cluster_availability import generate as generate_cluster
    from frontier.transient_lifecycle import read_availability
    expected_policy = "fixed" if config["resources"]["mode"] == "fixed" else "transient"
    if config["policy"]["name"] not in ((expected_policy,) if expected_policy == "fixed" else ("transient", "naive")):
        raise ValueError("Policy and resource mode mismatch")
    # Native Frontier handles final validation of its own CLI dataclasses.
    name = config["experiment"]["name"]
    if not isinstance(name, str) or not name or any(not(c.isalnum() or c in "-_") for c in name):
        raise ValueError("Invalid scenario name")
    resource = config["resources"]
    model = config["model"]
    sched = config["scheduler"]
    trace = Path(config["workload"]["trace"])
    if not trace.is_absolute():
        trace = ROOT / trace
    manifest = trace.with_suffix(".manifest.json")
    if not trace.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Missing trace/manifest: {trace}")
    max_tokens = json.loads(manifest.read_text(encoding="utf-8"))["required_frontier_max_tokens"]
    from malleserve.run_state import experiment_output
    out = experiment_output(config)
    out.mkdir(parents=True, exist_ok=True)
    if resource["mode"] == "fixed":
        count = int(resource["gpus"])
        transient_args = []
    elif resource["mode"] == "transient":
        lifecycle_mode = str(resource.get("lifecycle_mode", "baseline_recovery"))
        schedule = out / "availability.jsonl"
        lifetime = float(resource["node_lifetime_seconds"])
        duration = float(resource["duration_seconds"])
        schedule_kind = str(resource.get("schedule_kind", "rotation"))
        external_schedule = resource.get("schedule_file")
        if external_schedule:
            source = Path(str(external_schedule))
            if not source.is_absolute():
                source = ROOT / source
            if not source.is_file():
                raise FileNotFoundError(f"Missing external availability trace: {source}")
            import shutil
            if source.resolve() == schedule.resolve():
                raise ValueError("External source schedule must not point to the experiment output")
            shutil.copyfile(source, schedule)
            events = None
        elif schedule_kind == "synthetic_overlapping":
            events = generate_cluster(
                duration=duration,
                nodes=int(resource["candidate_nodes"]),
                seed=int(resource.get("schedule_seed", 7)),
                min_lifetime=float(resource.get("min_lifetime_seconds", 90)),
                max_lifetime=float(resource.get("max_lifetime_seconds", 240)),
                min_downtime=float(resource.get("min_downtime_seconds", 20)),
                max_downtime=float(resource.get("max_downtime_seconds", 90)),
                initial_spread=float(resource.get("initial_spread_seconds", 90)),
            )
        elif schedule_kind != "rotation":
            raise ValueError(f"Unsupported schedule_kind: {schedule_kind}")
        elif int(resource["candidate_nodes"]) != 2 and lifecycle_mode not in (
                "no_reclamation", "observed_one_reclaim"):
            raise ValueError("Rotation requires candidate_nodes=2; use synthetic_overlapping or schedule_file")
        elif lifecycle_mode == "no_reclamation":
            # No terminal reclaim: the candidate remains active through
            # the complete request-drain period as well as measurement.
            events = [
                {"timestamp": 0.0, "node_id": "transient-0", "action": "available"},
            ]
        elif lifecycle_mode == "observed_one_reclaim":
            if float(resource["startup_seconds"]) != 0:
                raise ValueError("Observed one-reclaim timeline specifies ready times; startup_seconds must be 0")
            reclaim_at = float(resource["reclaim_at_seconds"])
            ready_again_at = float(resource["ready_again_at_seconds"])
            if not (0 < reclaim_at < ready_again_at <= duration):
                raise ValueError("Expected 0 < reclaim_at < ready_again_at <= duration")
            events = [
                {"timestamp": 0.0, "node_id": "transient-0", "action": "available"},
                {"timestamp": reclaim_at, "node_id": "transient-0", "action": "reclaim"},
                {"timestamp": ready_again_at, "node_id": "transient-0", "action": "available"},
            ]
        elif lifecycle_mode == "single_reclaim":
            # Two distinct candidate nodes: a single abrupt reclaim with
            # a replacement node becoming available immediately afterward.
            # Normal startup delay applies to each availability event.
            reclaim_at = float(resource["reclaim_at_seconds"])
            if not 0 < reclaim_at < duration:
                raise ValueError("Expected 0 < reclaim_at_seconds < duration")
            events = [
                {"timestamp": 0.0, "node_id": "transient-0", "action": "available"},
                {"timestamp": reclaim_at, "node_id": "transient-0", "action": "reclaim"},
                {"timestamp": reclaim_at, "node_id": "transient-1", "action": "available"},
            ]
        elif lifecycle_mode in ("baseline_recovery", "naive"):
            events = generate(duration, lifetime)
        elif lifecycle_mode == "ideal_recovery":
            events = generate(duration, lifetime)
        else:
            raise ValueError(f"Unknown lifecycle_mode: {lifecycle_mode}")
        if events is not None:
            schedule.write_text("".join(json.dumps(e) + "\\n" for e in events), encoding="utf-8")
        parsed = read_availability(str(schedule))
        candidate_count = len({event.node for event in parsed})
        configured = int(resource["candidate_nodes"])
        if schedule_kind == "synthetic_overlapping" and not external_schedule and candidate_count != configured:
            raise ValueError(f"Synthetic schedule has {candidate_count} nodes, expected {configured}")
        count = int(resource["persistent_gpus"]) + (
            candidate_count * int(resource["transient_gpus_per_node"]))
        recovery_mode = (str(resource["recovery_mode"])
                         if lifecycle_mode == "single_reclaim"
                         else lifecycle_mode)
        if recovery_mode not in ("baseline_recovery", "ideal_recovery",
                                 "no_reclamation", "observed_one_reclaim", "naive"):
            raise ValueError(f"Unsupported recovery_mode: {recovery_mode}")
        transient_args = ["--transient_schedule_file", str(schedule),
             "--transient_persistent_replicas", str(resource["persistent_gpus"]),
             "--transient_replicas_per_node", str(resource["transient_gpus_per_node"]),
             "--transient_startup_seconds", str(resource["startup_seconds"]),
             "--transient_recovery_mode", (
                 "ideal_recovery" if recovery_mode == "ideal_recovery" else "baseline_recovery")]
    else:
        raise ValueError("Invalid resource mode")
    calibration = config.get("calibration", {})
    sensitivity_names = (
        "decode_phase_attn_rope_calibration_scale",
        "decode_phase_input_layernorm_calibration_scale",
        "decode_phase_post_attention_layernorm_calibration_scale",
    )
    import math
    sensitivity_args = []
    for key in sensitivity_names:
        value = calibration.get(key)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid experimental decode calibration {key}={value!r}")
        sensitivity_args.extend([
            "--random_forrest_execution_time_predictor_config_" + key, str(value)
        ])
    cluster_scheduler, cache_weight = resolve_routing_policy(sched)
    kv_routing_args = []
    if cluster_scheduler == "kv_aware":
        for key, default in (("load_weight", 1.0), ("cache_weight", 1.0), ("decode_token_weight", 16.0)):
            value = cache_weight if key == "cache_weight" else float(sched.get(key, default))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid KV-aware routing {key}={value!r}")
            kv_routing_args.extend([
                "--kv_aware_cluster_scheduler_config_" + key, str(value)
            ])
    args = [
        "--simulation_mode", "online",
        "--sys_arch", "co-location",
        "--cc_backend_config_type", "analytical",
        "--cluster_config_num_replicas", str(count),
        "--cluster_scheduler_config_type", cluster_scheduler,
        *kv_routing_args,
        "--replica_config_model_name", str(model["name"]),
        "--replica_config_device", str(model["device"]),
        "--replica_config_attn_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--replica_scheduler_config_type", str(sched["name"]),
        "--decode_cuda_graph_mode", str(sched["decode_cuda_graph_mode"]),
        "--vllm_v1_scheduler_config_max_model_len", str(sched["max_model_len"]),
        "--vllm_v1_scheduler_config_max_tokens_in_batch", str(sched["max_tokens_in_batch"]),
        "--vllm_v1_scheduler_config_block_size", str(sched["block_size"]),
        "--vllm_v1_scheduler_config_num_blocks_mode", "explicit",
        "--vllm_v1_scheduler_config_num_blocks", str(sched["num_blocks"]),
        ("--vllm_v1_scheduler_config_enable_prefix_caching" if sched["enable_prefix_caching"] else "--no-vllm_v1_scheduler_config_enable_prefix_caching"),
        ("--vllm_v1_scheduler_config_enable_chunked_prefill" if sched["enable_chunked_prefill"] else "--no-vllm_v1_scheduler_config_enable_chunked_prefill"),
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", str(trace),
        "--trace_request_generator_config_max_tokens", str(max_tokens),
        *transient_args,
        "--no-random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_atten_input_file",
        str(model["attention_profile"]),
        "--random_forrest_execution_time_predictor_config_linear_op_kernel_only_input_file",
        str(model["linear_profile"]),
        "--random_forrest_execution_time_predictor_config_atten_kernel_only_input_file",
        str(model["attention_kernel_profile"]),
        *sensitivity_args,
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
