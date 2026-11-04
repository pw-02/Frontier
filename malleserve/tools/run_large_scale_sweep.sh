#!/usr/bin/env bash
# Medium-scale, calibrated synthetic HPC availability sweep (NOT measured HPC arrivals).
# Default: 2 routers (rr,cwa), one seed, 8 persistent + 16x4 transient candidates.
# SEEDS="7 19 41" ROUTINGS="rr wa cwa" bash malleserve/tools/run_large_scale_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
read -r -a seeds <<< "${SEEDS:-7}"
read -r -a routings <<< "${ROUTINGS:-rr cwa}"
group="large_scale_hpc_r34"
for route in "${routings[@]}"; do
  case "$route" in rr|wa|cwa) ;; *) echo "Invalid route: $route" >&2; exit 2 ;; esac
  for gpus in 8 16 32; do
    echo "=== fixed${gpus} ${route} r34 ==="
    python -m malleserve.debug_run \
      "resources=fixed${gpus}" policy=fixed workload=mooncake_r34 \
      "scheduler.routing_policy=${route}" calibration=baseline \
      "experiment.group=${group}" \
      "experiment.name=fixed${gpus}_${route}_r34"
  done
done
for seed in "${seeds[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "SEEDS must be non-negative integer seeds" >&2; exit 2
  fi
  for recovery in baseline_recovery ideal_recovery; do
    for route in "${routings[@]}"; do
      name="large_s${seed}_${recovery}_${route}_r34"
      echo "=== ${name} ==="
      python -m malleserve.debug_run \
        resources=transient_large policy=transient workload=mooncake_r34 \
        "resources.schedule_seed=${seed}" "resources.lifecycle_mode=${recovery}" \
        "scheduler.routing_policy=${route}" calibration=baseline \
        "experiment.group=${group}" "experiment.name=${name}"
    done
  done
done
python -m malleserve.tools.summarize_large_scale_sweep
