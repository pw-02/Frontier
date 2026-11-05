# MalleServe — canonical E1–E4 HPC-serving experiments

MalleServe extends native Frontier's monolithic vLLM V1 simulator with
node-granular, potentially overlapping HPC GPU availability, startup delays,
abrupt reclamation, and failure-aware request accounting. The recommended
experimental setup uses **four numbered cases**. A case describes resource
and recovery semantics, not the routing policy.

| Case | Configuration | Research purpose |
| --- | --- | --- |
| **E1** | Stable GPUs only | Baseline serving without transient capacity |
| **E2** | Stable + idle HPC GPUs, no reclamation | Potential capacity benefit (optimistic bound) |
| **E3** | Stable + idle HPC GPUs, immediate reclamation, no recovery | Failure/disruption under abrupt resource loss |
| **E4** | Stable + idle HPC GPUs, immediate reclamation, full-request restart | Recovery benefit and recomputation overhead |

All cases use identical workload inputs when compared. E2 uses the
**same first availability times** as E3/E4 but never reclaims candidate nodes.
Consequently E2 has more GPU-seconds than E3/E4: it is an optimistic upper
bound, **not an equal-capacity control**. E3 and E4 do use matching availability
for the same seed, and differ only in recovery.

## How to run

Install dependencies according to Frontier's root README. From the repo root:

```bash
python -m unittest tests.unit.test_malleserve_hydra tests.unit.test_malleserve_e_cases -v

# First smoke-test E3 request failure handling with a small candidate pool.
python -m malleserve.debug_run \
  case=e3 workload=mooncake_r34 resources.candidate_nodes=2 \
  scheduler.routing_policy=cwa \
  experiment.group=e1_e4_hpc_r34 experiment.name=e3_smoke_n2_cwa_r34

# Full reproducible E1–E4 suite (default seed 7, RR + CWA).
bash malleserve/tools/run_e1_e4.sh
python -m malleserve.tools.summarize_e1_e4

# Expand routing and availability repetitions.
SEEDS="7 19 41" ROUTINGS="rr wa cwa" \
  bash malleserve/tools/run_e1_e4.sh
```

The sweep tests E1 on 8/16/32 permanent GPUs and E2/E3/E4 on 8 permanent
GPUs plus 16 candidate nodes, each with 4 GPUs. Candidate availability
is **seeded synthetic** (staggered arrivals, 90–240 second fragment lifetime,
20–90 second downtime); up to 72 GPU replicas may be ready concurrently.
Startup is 40 simulated seconds. The 34-RPS workload uses 30 seconds of
warmup and 300 seconds of measurement with a Mooncake-derived 12,000-request
prepared dataset, constant-rate arrivals and a 128-token output cap.

The default sweep runs **12 simulations** (6 fixed controls and 6 transient).
Three routing policies and three seeds yield 36 simulations (9 fixed controls
and 27 transient). Each run stores its summary in:

```text
outputs/malleserve/runs/e1_e4_hpc_r34/
  fixed/e1_fixed8_rr_r34_qwen_qwen3_8b/
  ...
  transient/e2_s7_rr_r34_qwen_qwen3_8b/
  transient/e3_s7_rr_r34_qwen_qwen3_8b/
  transient/e4_s7_rr_r34_qwen_qwen3_8b/
```

Summary CSV: `outputs/malleserve/runs/e1_e4_hpc_r34/e1_e4_comparison.csv`.

## Behaviour and metrics

- **E1:** stable-only native Frontier; no availability events.
- **E2:** first candidate allocation for each node makes it ready after
  startup, and it remains available through request drain. No reclaims.
- **E3:** abrupt reclamation destroys local scheduling and KV state;
  unfinished requests assigned to that node **fail permanently**. Their
  `request_failure` events are written to `metrics_ground_truth.jsonl`,
  with terminal failure counts in `transient_lifecycle.json`.
- **E4:** identical reclamation schedule, but interrupted requests are
  returned to the global scheduler and fully recompute from the beginning.

Compare **successful completions**, **failed requests**, **unresolved requests**,
SLO attainment/goodput, in-window throughput and outstanding work. Mean,
P50, P95 and P99 latency are **conditional on successful completion**, which
can hide E3's dropped requests. Failures must count as SLO misses, and must
not be silently equated with outstanding requests.

Experiment routing is orthogonal to case:
`scheduler.routing_policy=rr` (Round Robin),
`wa` (Workload-Aware), or `cwa` (Cache + Workload-Aware).
All policies share the vLLM V1 worker and prefix cache implementation.

## External HPC availability traces

For real availability replay set
`resources.schedule_file=/absolute/path/to/availability.jsonl`.
Events require only `timestamp` (non-negative seconds), `node_id` and
`action` (`available` or `reclaim`). Different nodes can overlap, while
each node's events must alternate valid availability/reclaim transitions.

E3/E4 replay the trace unchanged. E2 retains the first `available` event
per node but drops subsequent reclaims, maintaining its no-reclamation
upper-bound semantics. Each distinct `node_id` represents four GPU
replicas in the current preset. The external trace is included in the
experiment fingerprint. Use a unique name for a different availability
trace or benchmark setup.

## Historical compatibility

Previous experiments used names such as `baseline_recovery`,
`ideal_recovery`, `single_reclaim`, `observed_one_reclaim` and
`no_reclamation`. These are **not** canonical E1–E4 configurations.
Underlying native oracle and legacy lifecycle mechanisms may remain for
reading/validating historical simulation results; the old sweep entry
points and prior experiment groups are no longer part of the recommended
research workflow. Existing result files have **not** been moved or deleted.

**Validation:** Unit tests and scenario composition are included, but
native end-to-end E3 failure handling and a complete large-scale E1–E4
sweep must be tested in your Frontier environment before paper claims.
