#!/usr/bin/env bash
# Controlled 10-RPS routing-policy sweep: 4/8 fixed GPUs and two transient recovery modes.
# Usage: bash malleserve/tools/run_routing_policy_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

group="routing_policy_sweep"
for resource in fixed4 fixed8 transient; do
  for routing in rr wa cwa; do
    if [[ "$resource" == transient ]]; then
      recoveries=(baseline_recovery ideal_recovery)
      policy=transient
    else
      recoveries=(fixed)
      policy=fixed
    fi
    for recovery in "${recoveries[@]}"; do
      if [[ "$recovery" == fixed ]]; then
        name="${resource}_${routing}_r10"
        extra=()
      else
        name="${resource}_${recovery}_${routing}_r10"
        extra=("resources.lifecycle_mode=${recovery}")
      fi
      echo "=== ${name} ==="
      python -m malleserve.debug_run \
        "resources=${resource}" "policy=${policy}" \
        workload=mooncake_r10 \
        "scheduler.routing_policy=${routing}" \
        scheduler.load_weight=1.0 \
        scheduler.decode_token_weight=16.0 \
        calibration=baseline \
        "experiment.group=${group}" \
        "experiment.name=${name}" \
        "${extra[@]}"
    done
  done
done
