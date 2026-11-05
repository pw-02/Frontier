#!/usr/bin/env bash
# Canonical MalleServe E1–E4 sweep, 34 RPS. Fixed E1: 8 or 16 GPU baseline.
# E2/E3/E4 share each seed's overlapping node activation trace.
# SEEDS="7 19 41" ROUTINGS="rr wa cwa" bash malleserve/tools/run_e1_e4.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
read -r -a seeds <<< "${SEEDS:-7}"
read -r -a routes <<< "${ROUTINGS:-rr cwa}"
group="e1_e4_hpc_r34"
for route in "${routes[@]}"; do
  case "$route" in rr|wa|cwa) ;; *) echo "Invalid router: $route" >&2; exit 2;; esac
  for gpus in 8 16 32; do
    name="e1_fixed${gpus}_${route}_r34"
    echo "=== ${name} ==="
    python -m malleserve.debug_run \
      case=e1 resources.gpus="${gpus}" workload=mooncake_r34 \
      scheduler.routing_policy="${route}" \
      experiment.group="${group}" experiment.name="${name}"
  done
done
for seed in "${seeds[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then echo "Invalid seed: $seed" >&2; exit 2; fi
  for experiment in e2 e3 e4; do
    for route in "${routes[@]}"; do
      name="${experiment}_s${seed}_${route}_r34"
      echo "=== ${name} ==="
      python -m malleserve.debug_run \
        case="${experiment}" workload=mooncake_r34 \
        resources.schedule_seed="${seed}" \
        scheduler.routing_policy="${route}" \
        experiment.group="${group}" experiment.name="${name}"
    done
  done
done
python -m malleserve.tools.summarize_e1_e4
