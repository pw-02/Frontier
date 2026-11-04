#!/usr/bin/env bash
# Capacity-exposure sensitivity: same 12-RPS requests, variable node lifetimes.
# Default: 18 runs = 6 fixed (4/8 GPUs) + 12 transient (60/180s, 2 recoveries, 3 routers).
# Optional: LIFETIMES="60 180 300" ROUTINGS="rr wa cwa" bash ...
set -euo pipefail
cd "$(dirname "$0")/../.."

group="transient_lifetime_sweep_r12"
read -r -a lifetimes <<< "${LIFETIMES:-60 180}"
read -r -a routings <<< "${ROUTINGS:-rr wa cwa}"
startup="${STARTUP_SECONDS:-40}"

if [[ "${#lifetimes[@]}" -eq 0 || "${#routings[@]}" -eq 0 ]]; then
  echo "Expected non-empty LIFETIMES and ROUTINGS" >&2
  exit 2
fi

for routing in "${routings[@]}"; do
  case "$routing" in rr|wa|cwa) ;; *) echo "Invalid routing: $routing" >&2; exit 2 ;; esac
done
for lifetime in "${lifetimes[@]}"; do
  if ! [[ "$lifetime" =~ ^[0-9]+$ ]] || (( lifetime <= 0 )); then
    echo "LIFETIMES must contain positive integer seconds" >&2; exit 2
  fi
done

# Fixed controls: use the identical workload and scheduler calibration.
for workers in 4 8; do
  for routing in "${routings[@]}"; do
    name="fixed${workers}_${routing}_r12"
    echo "=== ${name} ==="
    python -m malleserve.debug_run \
      "resources=fixed${workers}" policy=fixed workload=mooncake_r12 \
      "scheduler.routing_policy=${routing}" \
      scheduler.load_weight=1.0 scheduler.decode_token_weight=16.0 \
      calibration=baseline \
      "experiment.group=${group}" "experiment.name=${name}"
  done
done

# Same worker-lifecycle events for all router/recovery pairings at each lifetime.
# Do not change persistent GPUs, arrivals, cache or predictor calibration.
for lifetime in "${lifetimes[@]}"; do
  for recovery in baseline_recovery ideal_recovery; do
    for routing in "${routings[@]}"; do
      name="transient_l${lifetime}_s${startup}_${recovery}_${routing}_r12"
      echo "=== ${name} ==="
      python -m malleserve.debug_run \
        resources=transient policy=transient workload=mooncake_r12 \
        "resources.node_lifetime_seconds=${lifetime}" \
        "resources.startup_seconds=${startup}" \
        "resources.lifecycle_mode=${recovery}" \
        "scheduler.routing_policy=${routing}" \
        scheduler.load_weight=1.0 scheduler.decode_token_weight=16.0 \
        calibration=baseline \
        "experiment.group=${group}" "experiment.name=${name}"
    done
  done
done

echo "Summarize: python -m malleserve.tools.summarize_transient_lifetime_sweep"
