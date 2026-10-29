#!/usr/bin/env bash
# Run the fixed-capacity Polaris validation matrix against a calibrated Frontier runner.
# Usage:
#   ONLY=4:10 bash validation/run_frontier_matrix.sh   # one sanity check
#   bash validation/run_frontier_matrix.sh             # all 27 cells
# Optional: SOURCE_RUNNER=..., DATASET=..., KV_BLOCKS=8115
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
SOURCE_RUNNER="${SOURCE_RUNNER:-examples/malleserve/run_mooncake_calibrated_rr_piecewise_2048.sh}"
DATASET="${DATASET:-data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json}"
KV_BLOCKS="${KV_BLOCKS:-8115}"  # FlashAttention 2 Polaris baseline; not FlashInfer diagnostic

[[ -f "$SOURCE_RUNNER" ]] || { echo "Missing calibrated runner: $SOURCE_RUNNER" >&2; exit 2; }
[[ -f "$DATASET" ]] || { echo "Missing prepared dataset: $DATASET" >&2; exit 2; }
[[ "$KV_BLOCKS" =~ ^[0-9]+$ ]] || { echo "KV_BLOCKS must be an integer" >&2; exit 2; }

mkdir -p data/malleserve
# Convert one trace per rate. Within a rate, *all worker counts replay the same requests*.
for rate in 2 4 6 8 10 12; do
  file="data/malleserve/mooncake_r${rate}_330s.csv"
  manifest="data/malleserve/mooncake_r${rate}_330s.manifest.json"
  if [[ ! -f "$file" || ! -f "$manifest" ]]; then
    python examples/malleserve/convert_mooncake.py \
      --dataset "$DATASET" \
      --rate "$rate" \
      --warmup-seconds 30 --measurement-seconds 300 \
      --max-output-tokens 128 \
      --output "$file"
  fi
done

# The calibrated runner resolves repo ROOT relative to its script's location.
# Write temporary siblings in examples/malleserve/, not under /tmp.
GENERATED=()
cleanup() { for file in "${GENERATED[@]}"; do rm -f "$file"; done; }
trap cleanup EXIT

for workers in 1 2 4 6 8; do
  generated="$(mktemp "examples/malleserve/.frontier_validation_${workers}.XXXXXX.sh")"
  GENERATED+=("$generated")
  python - "$SOURCE_RUNNER" "$generated" "$workers" "$KV_BLOCKS" <<'PY'
import pathlib, re, sys
src, dest, workers, blocks = sys.argv[1:]
script = pathlib.Path(src).read_text(encoding="utf-8")
for check in (
    "--cluster_scheduler_config_type round_robin",
    "--decode_cuda_graph_mode piecewise",
    "--vllm_v1_scheduler_config_max_tokens_in_batch 2048",
    "--vllm_v1_scheduler_config_num_blocks_mode explicit",
):
    if check not in script:
        raise SystemExit(f"The source runner is not the expected RR/piecewise/2048/explicit-block baseline: {check!r} missing")
for flag, value in (("--cluster_config_num_replicas", workers),
                    ("--vllm_v1_scheduler_config_num_blocks", blocks)):
    pattern = r"(" + re.escape(flag) + r"[ \t]+)[0-9]+(?=[ \t]|$)"
    script, count = re.subn(pattern, lambda m: m.group(1) + value, script)
    if count != 1:
        raise SystemExit(f"Expected exactly one occurrence of {flag}; found {count}")
pathlib.Path(dest).write_text(script, encoding="utf-8")
PY

  for rate in 2 4 6 8 10 12; do
    # Real matrix: worker counts 1/2/4 have r2..r10; 6/8 have r2..r12.
    if (( workers < 6 && rate == 12 )); then continue; fi
    if [[ -n "${ONLY:-}" && "$ONLY" != "${workers}:${rate}" ]]; then continue; fi

    run_id="validation_w${workers}_r${rate}"
    trace="data/malleserve/mooncake_r${rate}_330s.csv"
    output="outputs/examples/malleserve/$run_id"
    if [[ -d "$output" ]] && find "$output" -name malleserve_summary.json -print -quit | grep -q .; then
      echo "SKIP completed: $run_id"
      continue
    fi
    echo "===== Frontier: workers=${workers} rate=${rate}/s KV=${KV_BLOCKS} ====="
    bash "$generated" "$trace" "$run_id"
  done
done

echo "All selected simulations completed. Run comparison script next."
