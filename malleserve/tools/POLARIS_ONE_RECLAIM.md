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
## Request accounting (updated comparator)

The CSV now includes `arrived`, `completed_eventually`, `no_completion_event`,
`completed_after_window_end`, and `completion_fraction`. The former `n`
remains a count of requests with completion records. Percentiles are conditional
on completion and **must not** be interpreted as full-arrival percentiles when
`no_completion_event` is nonzero. Results need no simulation rerun.

The comparator cannot attribute interruptions without separately recorded
reclamation diagnostics; do not infer retry counts from missing completions.

## Aligned-window and recovery analysis

Use the same comparator with `--aligned-windows` to compare [30,180),
[180,240), [240,299) and [299,330) seconds. The first window excludes
Frontier's first 30 seconds absent from measured Polaris data; the final
window isolates arrivals after restored transient workers become ready.
This does **not** pair requests across systems or eliminate trace differences.
All percentiles are grouped by arrival time (not completion time).

```bash
python -m malleserve.tools.compare_one_reclaim \\
  --frontier-run outputs/malleserve/runs/polaris_one_reclaim_r10_stagefix \\
  --frontier-manifest data/malleserve/mooncake_r10_330s.manifest.json \\
  --polaris-requests /path/to/data/requests.jsonl \\
  --polaris-zero-unix 1793323433.066342 \\
  --aligned-windows --output one_reclaim_aligned_stagefix.csv
```
