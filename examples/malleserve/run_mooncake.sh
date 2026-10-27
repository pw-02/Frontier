#!/usr/bin/env bash
# Functional Mooncake trace smoke run; DUMMY timing, not Polaris calibration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1

TRACE_FILE="${1:?Usage: bash examples/malleserve/run_mooncake.sh trace.csv [run_id]}"
RUN_ID="${2:-mooncake_smoke}"
MANIFEST="${TRACE_FILE%.csv}.manifest.json"
if [[ ! -f "$TRACE_FILE" || ! -f "$MANIFEST" ]]; then
  echo "Missing trace CSV or adjacent manifest; run convert_mooncake.py first." >&2
  exit 2
fi
TRACE_MAX_TOKENS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["required_frontier_max_tokens"])' "$MANIFEST")"
OUTPUT_DIR="${METRICS_OUTPUT_DIR:-$ROOT/outputs/examples/malleserve/$RUN_ID}"
mkdir -p "$OUTPUT_DIR"
echo "Frontier trace: $TRACE_FILE"
echo "Metrics: $OUTPUT_DIR"
echo "Timing model: DUMMY (do not compare numerical values with Polaris)"
python -m frontier.main \
  --simulation_mode online \
  --sys_arch co-location \
  --cc_backend_config_type analytical \
  --cluster_config_num_replicas 1 \
  --cluster_scheduler_config_type sticky_round_robin \
  --replica_config_model_name llama3.1-8b \
  --replica_config_device a100 \
  --replica_config_attn_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --replica_scheduler_config_type vllm_v1 \
  --decode_cuda_graph_mode none \
  --vllm_v1_scheduler_config_max_model_len 17000 \
  --vllm_v1_scheduler_config_max_tokens_in_batch 16384 \
  --vllm_v1_scheduler_config_block_size 16 \
  --vllm_v1_scheduler_config_num_blocks_mode explicit \
  --vllm_v1_scheduler_config_num_blocks 8192 \
  --vllm_v1_scheduler_config_enable_prefix_caching \
  --vllm_v1_scheduler_config_enable_chunked_prefill \
  --request_generator_config_type trace_replay \
  --trace_request_generator_config_trace_file "$TRACE_FILE" \
  --trace_request_generator_config_max_tokens "$TRACE_MAX_TOKENS" \
  --random_forrest_execution_time_predictor_config_enable_dummy_mode \
  --random_forrest_execution_time_predictor_config_dummy_execution_time_ms 1.0 \
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
