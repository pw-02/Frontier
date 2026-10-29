#!/usr/bin/env bash
# Native Frontier vLLM continuous-batching transient-worker simulation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1

TRACE_FILE="${1:?Usage: bash examples/malleserve/transient/run_native_transient.sh TRACE_CSV SCHEDULE_JSONL [run_id]}"
AVAILABILITY_FILE="${2:?Missing availability JSONL}"
RUN_ID="${3:-native_transient}"
MANIFEST="${TRACE_FILE%.csv}.manifest.json"
[[ -f "$TRACE_FILE" && -f "$MANIFEST" && ( "${RESOURCE_MODE:-transient}" == "fixed" || -f "$AVAILABILITY_FILE" ) ]] || {
  echo "Missing trace, manifest or transient availability JSONL" >&2
  exit 2
}

PERSISTENT_REPLICAS="${PERSISTENT_REPLICAS:-4}"
GPUS_PER_TRANSIENT_NODE="${GPUS_PER_TRANSIENT_NODE:-4}"
TRANSIENT_CANDIDATE_NODES="${TRANSIENT_CANDIDATE_NODES:-2}"
TRANSIENT_STARTUP_SECONDS="${TRANSIENT_STARTUP_SECONDS:-40}"
RESOURCE_MODE="${RESOURCE_MODE:-transient}"
if [[ "$RESOURCE_MODE" == "fixed" ]]; then
  TOTAL_REPLICAS="${FIXED_REPLICAS:?FIXED_REPLICAS required for fixed mode}"
  TRANSIENT_ARGS=()
elif [[ "$RESOURCE_MODE" == "transient" ]]; then
  TOTAL_REPLICAS=$((PERSISTENT_REPLICAS + GPUS_PER_TRANSIENT_NODE * TRANSIENT_CANDIDATE_NODES))
  TRANSIENT_ARGS=(
    --transient_schedule_file "$AVAILABILITY_FILE"
    --transient_persistent_replicas "$PERSISTENT_REPLICAS"
    --transient_replicas_per_node "$GPUS_PER_TRANSIENT_NODE"
    --transient_startup_seconds "$TRANSIENT_STARTUP_SECONDS"
  )
else
  echo "Unsupported RESOURCE_MODE: $RESOURCE_MODE" >&2; exit 2
fi
TRACE_MAX_TOKENS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["required_frontier_max_tokens"])' "$MANIFEST")"
OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$ROOT/outputs/examples/malleserve/$RUN_ID}"
mkdir -p "$OUTPUT_DIR"

echo "Native Frontier transient simulation"
echo "Trace: $TRACE_FILE"
echo "Schedule: $AVAILABILITY_FILE"
echo "Precreated replicas: $TOTAL_REPLICAS"
echo "Initialization delay: ${TRANSIENT_STARTUP_SECONDS}s (assumption until measured)"

python -m frontier.main \
  --simulation_mode online \
  --sys_arch co-location \
  --cc_backend_config_type analytical \
  --cluster_config_num_replicas "$TOTAL_REPLICAS" \
  --cluster_scheduler_config_type round_robin \
  --replica_config_model_name Qwen/Qwen3-8B \
  --replica_config_device a100 \
  --replica_config_attn_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --replica_scheduler_config_type vllm_v1 \
  --decode_cuda_graph_mode piecewise \
  --vllm_v1_scheduler_config_max_model_len 17000 \
  --vllm_v1_scheduler_config_max_tokens_in_batch 2048 \
  --vllm_v1_scheduler_config_block_size 16 \
  --vllm_v1_scheduler_config_num_blocks_mode explicit \
  --vllm_v1_scheduler_config_num_blocks 8192 \
  --vllm_v1_scheduler_config_enable_prefix_caching \
  --vllm_v1_scheduler_config_enable_chunked_prefill \
  --request_generator_config_type trace_replay \
  --trace_request_generator_config_trace_file "$TRACE_FILE" \
  --trace_request_generator_config_max_tokens "$TRACE_MAX_TOKENS" \
  "${TRANSIENT_ARGS[@]}" \
  --no-random_forrest_execution_time_predictor_config_enable_dummy_mode \
  --random_forrest_execution_time_predictor_config_atten_input_file data/profiling/compute/a100/Qwen/Qwen3-8B/attention_calibrated.csv \
  --random_forrest_execution_time_predictor_config_linear_op_kernel_only_input_file data/profiling/compute/a100/Qwen/Qwen3-8B/linear_op_kernel_only.csv \
  --random_forrest_execution_time_predictor_config_atten_kernel_only_input_file data/profiling/compute/a100/Qwen/Qwen3-8B/attention_kernel_only.csv \
  --metrics_config_output_dir "$OUTPUT_DIR" \
  --metrics_config_run_id "$RUN_ID" \
  --metrics_config_write_metrics \
  --metrics_config_store_request_metrics \
  --metrics_config_enable_metrics_ground_truth_trace \
  --no-metrics_config_store_plots \
  --no-metrics_config_enable_chrome_trace \
  --no-metrics_config_write_json_trace

python examples/malleserve/analyze_mooncake.py \
  --manifest "$MANIFEST" --metrics-dir "$OUTPUT_DIR" --slo-seconds "${SLO_SECONDS:-30}"

echo "Native transient lifecycle diagnostics:"
find "$OUTPUT_DIR" -name transient_lifecycle.json -print
