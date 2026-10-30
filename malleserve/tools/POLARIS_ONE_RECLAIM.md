# Frontier vs Polaris one-reclaim evidence experiment

This experiment deliberately retains **native naïve round-robin routing**;
it does not implement a proposed MalleServe policy.

## Exact service-capacity schedule modeled

From the supplied one-reclaim Polaris run (relative to measured phase):

- t=0: 8 ready workers (4 persistent + 4 transient)
- t≈180: 4 transient workers reclaimed, 4 ready remain
- t≈240: Polaris initiates restore; **not a serving transition**
- t≈299: restored workers become ready; 8 workers again

The Frontier lifecycle uses `startup_seconds: 0` because we encode
*observed service readiness* directly at t=0 and t=299.
The 240-second restore initiation is not represented by an additional
serving event. This avoids incorrectly delaying initial readiness.

This is **capacity timeline matching**, not a claim that Frontier reproduces
the exact supervisor startup/retry protocol, HTTP routing, or worker setup.
The simulator's native reclaim/retry semantics are unchanged. In
particular, the Polaris run saw 7 logical retries, whereas Frontier
may retry a different number of requests.

## Run

```bash
python -m malleserve.debug_run \\
  resources=polaris_one_reclaim policy=naive workload=mooncake_r10 \\
  experiment.name=polaris_one_reclaim_r10
```

Frontier output:
`outputs/malleserve/runs/polaris_one_reclaim_r10`.

## Compare arrival-time latency bins

Unzip the uploaded Polaris bundle and locate
`transient-one-reclaim-8-r10/<run>/data/requests.jsonl`.
Obtain the exact measured-phase epoch from
`data/events.jsonl` before invoking:

```bash
python -m malleserve.tools.compare_one_reclaim \\
  --frontier-run outputs/malleserve/runs/polaris_one_reclaim_r10 \\
  --frontier-manifest data/malleserve/mooncake_r10_330s.manifest.json \\
  --polaris-requests /path/to/data/requests.jsonl \\
  --polaris-zero-unix 1793323433.066342 \\
  --output one_reclaim_latency_by_arrival.csv
```

The comparison bins arrivals [0,180), [180,240), [240,300) seconds.
Different trace sampling, per-request input/output token sequences,
timing predictor bias, and retry semantics remain potential confounders.
Use windowed curves as evidence, not exact per-request pairing.
\nThe supplied Polaris event log records `schedule_started` at Unix 1793323433.066342; the requests use `phase=measured`. Use this timestamp as the observed measured-phase origin.\n