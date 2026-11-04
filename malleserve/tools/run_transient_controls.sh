#!/usr/bin/env bash
# Extra controls for the already-completed 12-RPS lifetime sensitivity study.
# Adds 3 no-reclamation runs and 6 single-reclaim runs to the SAME output group.
set -euo pipefail
cd "$(dirname "$0")/../.."
group="transient_lifetime_sweep_r12"
for routing in rr wa cwa; do
  name="transient_no_reclamation_s40_${routing}_r12"
  echo "=== ${name} ==="
  python -m malleserve.debug_run \
    resources=transient policy=transient workload=mooncake_r12 \
    resources.lifecycle_mode=no_reclamation resources.startup_seconds=40 \
    scheduler.routing_policy="${routing}" \
    scheduler.load_weight=1.0 scheduler.decode_token_weight=16.0 \
    calibration=baseline \
    experiment.group="${group}" experiment.name="${name}"
  for recovery in baseline_recovery ideal_recovery; do
    name="transient_single_reclaim_t180_s40_${recovery}_${routing}_r12"
    echo "=== ${name} ==="
    python -m malleserve.debug_run \
      resources=transient policy=transient workload=mooncake_r12 \
      resources.lifecycle_mode=single_reclaim \
      +resources.reclaim_at_seconds=180 \
      resources.startup_seconds=40 \
      scheduler.routing_policy="${routing}" \
      scheduler.load_weight=1.0 scheduler.decode_token_weight=16.0 \
      calibration=baseline \
      experiment.group="${group}" experiment.name="${name}" \
      "+resources.recovery_mode=${recovery}"
  done
done
python -m malleserve.tools.summarize_transient_lifetime_sweep
