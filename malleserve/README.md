# MalleServe — Hydra research front-end to native Frontier

All MalleServe-specific code is now organized here, not under `examples/`.

- `conf/` — canonical Hydra defaults and config groups (model, scheduler, workload, resources, policy)
- `debug_run.py` — single-process step-through Frontier runner
- `scenario.py` — maps composed Hydra config to Frontier's native CLI dataclasses
- `tools/` — Mooncake trace conversion, analysis and resource schedule generator

## Start and debug

```bash
pip install -e \".[test]\"
python -m malleserve.debug_run
python -m malleserve.debug_run resources=fixed4 policy=fixed
python -m malleserve.debug_run resources=fixed8 policy=fixed
python -m malleserve.debug_run resources.startup_seconds=0 experiment.name=baseline_s0
```

Set a breakpoint in `frontier/transient_lifecycle.py` or `frontier/simulator.py`, then use **Debug Python File** on `malleserve/debug_run.py`. Hydra and Frontier execute in the same interpreter. For a multi-run sweep, give each scenario a unique output name:

```bash
python -m malleserve.debug_run -m resources.startup_seconds=0,20,40 'experiment.name=baseline_s${resources.startup_seconds}'
```

## Medium-scale overlapping HPC capacity experiments (new)

The **native Frontier** lifecycle now accepts arbitrary overlapping availability
events rather than assuming two alternating nodes. The existing rotation
preset is unchanged. New `resources=transient_large` provides an
**explicitly synthetic**, seeded, heterogeneous node availability model:

- 8 permanent GPU replicas, 16 independent candidate nodes × 4 GPUs each:
  at most 72 ready GPU replicas (all only if candidates overlap).
- Candidates arrive at staggered times; allocations last uniformly 90–240
  simulated seconds, with 20–90 seconds between allocations.
- 40-second startup applies to each allocation. Each node's schedule is
  deterministic for a specified seed; independent candidate availability
  intervals may overlap.
- Fixed 8, Fixed 16 and Fixed 32 serve as always-ready reference capacities.
- `workload=mooncake_r34` replays **34 constant arrivals/sec** over the
  previous 30s warmup + 300s measurement window (11,220 requests). It uses
  the existing prepared 12k-record Mooncake dataset with the existing
  output length cap of 128 tokens; **this is not yet a one-hour or
  trace-timestamp-faithful workload**.

Quick smoke run with fewer candidates first:

```bash
python -m unittest tests.unit.test_malleserve_hydra -v
python -m malleserve.debug_run resources=transient_large policy=transient \
  workload=mooncake_r34 resources.candidate_nodes=4 \
  scheduler.routing_policy=cwa \
  experiment.group=large_scale_hpc_r34 \
  experiment.name=smoke_n4_cwa_r34
```

Full fixed/transient comparison (2 routing policies × 3 fixed capacities,
2 recovery modes × 1 resource seed = **10 simulations** by default):

```bash
bash malleserve/tools/run_large_scale_sweep.sh
python -m malleserve.tools.summarize_large_scale_sweep
```

For three seeds and all routing policies (9 fixed + 18 transient = **27 runs**):

```bash
SEEDS="7 19 41" ROUTINGS="rr wa cwa" \
  bash malleserve/tools/run_large_scale_sweep.sh
```

All results are under `outputs/malleserve/runs/large_scale_hpc_r34/`.
The summary records latency, SLO, in-window completion throughput, outstanding
work, reclamation counts, retries and lost/restored work. Completed matching
runs are skipped when repeated.

To replay **your own HPC availability log**, first convert it to JSONL
with one event per line:
`{"timestamp": 120.0, "node_id": "compute-12", "action": "available"}`
or action `"reclaim"`. Timestamps are simulated seconds from experiment
start, and node IDs represent distinct four-GPU groups in this preset.
Events can overlap across nodes but must alternate available/reclaim
**within** each node. Every named candidate consumes preallocated simulator
replicas. In the Hydra CLI set `resources.schedule_file=/absolute/path/to/trace.jsonl`;
this takes precedence over the synthetic generator. All events are checked
using Frontier's native `read_availability` validator, and the external
trace is included in experiment fingerprints so changes require a new run.
For an external trace use a unique `experiment.name` to avoid overwriting
past results. The reported candidate count derives from distinct node IDs
in the file, not `resources.candidate_nodes`.

These are **synthetic sensitivity benchmarks**, not claims about a measured
HPC system. Before interpreting results as realistic deployment performance,
replace synthetic availability with scheduler allocation logs, validate GPU
startup times, and use actual multi-hour request arrivals and workloads.

## Routing policy comparison sweep (RR / WA / CWA)

MalleServe exposes three clear research-facing names:

- `rr`: Round Robin (native `round_robin`).
- `wa`: Workload-Aware; outstanding prefill/decode work, no incoming-request cache preference (native `kv_aware`, `cache_weight=0`).
- `cwa`: Cache + Workload-Aware; outstanding work plus prompt-prefix cache locality (native `kv_aware`, `cache_weight=1`).

Specify one using `scheduler.routing_policy=rr|wa|cwa`. The original
`scheduler.cluster` and `scheduler.cache_weight` overrides continue to work
when `scheduler.routing_policy=null` (the default). If an alias is selected it
takes precedence over those legacy knobs.

Run all 12 controlled Qwen3-8B, Mooncake 10-RPS scenarios (3 routing policies
× fixed4, fixed8, baseline recovery, ideal recovery):

```bash
bash malleserve/tools/run_routing_policy_sweep.sh
python -m malleserve.tools.summarize_routing_policy_sweep
```

Outputs are organized under
`outputs/malleserve/runs/routing_policy_sweep/{fixed,transient}/<experiment>_qwen_qwen3_8b/`.
Setting `experiment.group=routing_policy_sweep` changes only the result
directory. Without `experiment.group`, older result paths are unchanged.
The script uses distinct names and does not move or overwrite ongoing
experiments in the original `runs/fixed` and `runs/transient` folders.
Runs with the same completed fingerprint are skipped on re-execution.
The summary command writes `routing_policy_comparison.csv` to the group folder
for subsequent P50/P95/P99 plots and flags unfinished runs as MISSING.

## Transient lifetime and workload-pressure study (12 RPS)

This controlled sensitivity study tests the hypothesis that 60-second GPU
fragments with 40-second startup offer too little useful capacity. It uses the
*same Mooncake 12-RPS trace* for all policies. It does **not** modify the
previous 10-RPS routing sweep or claim these illustrative node lifetimes
represent measured Polaris availability.

Run the default 18 comparisons (fixed 4/8 with RR/WA/CWA; transient 60s/180s
with Baseline/Ideal Recovery and RR/WA/CWA):

```bash
bash malleserve/tools/run_transient_lifetime_sweep.sh
python -m malleserve.tools.summarize_transient_lifetime_sweep
```

To add a third 300-second fragment case, keeping the same fixed controls:

```bash
LIFETIMES="60 180 300" bash malleserve/tools/run_transient_lifetime_sweep.sh
python -m malleserve.tools.summarize_transient_lifetime_sweep
```

Outputs stay under `outputs/malleserve/runs/transient_lifetime_sweep_r12/`,
separately from earlier experiments. The aggregated
`transient_lifetime_comparison.csv` includes P50/P95/P99, TTFT, completion
throughput, outstanding requests, recovery counters, and the number of
**extra ready GPU-seconds** derived from each run's actual availability trace.

For the 330-second schedule at 40-second startup, the 60-second fragments
provide 100 seconds of 4 extra ready GPUs (average 5.21 ready GPUs total);
180-second fragments provide 250 seconds of 4 extra ready GPUs (average 7.03).
Thus different lifetimes change both GPU exposure and reclamation frequency;
compare RR/WA/CWA **within** a lifetime to isolate routing, and compare
lifetimes to understand capacity viability. Fixed 8 remains an always-ready
capacity reference, not an equal-exposure control. The original 12-RPS trace
has 30 seconds warmup and 300 seconds of measurement arrivals.

This is an experimental sensitivity analysis, not a cherry-picked replacement
for the existing baseline; report both 60-second and 180-second outcomes,
including cases where the longer lifetime does not help. In particular,
check `transient_lifecycle.json` for interrupted requests and
`measured_still_outstanding_at_window_end` in the summary.

## Additional reclamation controls at 12 RPS

The original `baseline_recovery` means **full recomputation of interrupted
requests following an abrupt transient-node reclaim**, not the Fixed 4
baseline. `ideal_recovery` preserves committed work at zero transfer cost
(an oracle, not a real migration implementation).

To complement the already-completed 60s/180s lifetime experiments, run:

```bash
bash malleserve/tools/run_transient_controls.sh
```

This runs **nine additional cases** (three routing policies each):

- `no_reclamation`: one extra four-GPU node becomes ready at t=40s and
  remains active through the measurement and request-drain period; 0 reclaims.
- `single_reclaim` + `baseline_recovery`: first extra node ready t=40s,
  reclaimed abruptly at t=180s with full recomputation of interrupted work;
  a different four-GPU node becomes ready at t=220s and stays online.
- `single_reclaim` + `ideal_recovery`: identical availability schedule,
  but preserves committed progress and KV context across the one reclaim.

All runs use `mooncake_r12`, 4 persistent GPUs, 40s startup and unchanged
router/predictor settings, and are stored alongside the existing study under
`outputs/malleserve/runs/transient_lifetime_sweep_r12/`. The script
automatically regenerates `transient_lifetime_comparison.csv` with these
new controls. Existing fixed and rotating experiments are **not** rerun.

These controls isolate the incremental effect of one reclamation under
identical availability and routing settings. Do **not** treat no-reclamation
as equal-capacity to the one-reclaim case: it has no 40s replacement gap and
retains a warm KV cache. The earlier `observed_one_reclaim` debug scenario
is retained separately; it uses an externally specified ready-time timeline
and requires zero startup delay, so we do not mix it into this controlled
40-second-startup comparison.

## KV-aware cluster routing (experimental)

Use `scheduler.cluster=kv_aware` to place requests using both local prefix
cache overlap and the worker's outstanding token work. This reuses the
existing vLLM V1 replica scheduler and per-worker KV block pool. It does
**not** move KV state between workers or change simulated GPU execution.

```bash
python -m malleserve.debug_run \
  resources=fixed4 policy=fixed workload=mooncake_r12 \
  scheduler.cluster=kv_aware \
  scheduler.load_weight=1.0 scheduler.decode_token_weight=16.0 \
  experiment.name=fixed4_r12_kv_aware
```

The routing heuristic minimises (in prefill-equivalent tokens):

```text
new_request_uncached_prefill_tokens +
  load_weight * (outstanding_prefill_tokens +
                 decode_token_weight * outstanding_decode_tokens)
```

Outstanding work includes both **waiting and running** requests. Estimates
come from current request progress and cache-block identities; they are
not a calibrated latency or service-time predictor. The read-only cache
probe does not update cache-hit statistics, allocate blocks, or imply
that KV state is shared. When multiple arrivals are assigned in one
event, projected work is updated between placements. Ties use replica
ID order. Under native transient lifecycle only ready workers are eligible;
cache state on reclaimed workers is reset by the existing lifecycle.

Keep experiment names distinct for `round_robin`, `lor`, and
`kv_aware` comparisons. Because their routing differs, cache hits and
batch composition may differ even with identical source traces.

Run targeted regression tests:

```bash
python -m unittest tests.unit.test_kv_aware_cluster_scheduler -v
```

## Workloads

Convert prepared Mooncake JSON to a request trace CSV and matching manifest:

```bash
python -m malleserve.tools.convert_mooncake \\
  --dataset data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json \\
  --rate 10 --warmup-seconds 30 --measurement-seconds 300 \\
  --output data/malleserve/mooncake_r10_330s.csv
```

The CSV captures fixed-rate arrival times, request prompt/output sizes and prefix identities, not original Mooncake arrival timestamps. Frontier replays these requests identically under each resource policy. Transient availability is a separate JSONL event schedule, generated by `malleserve/tools/generate_native_schedule.py` for transient rotation scenarios. The calibrated A100/Qwen3-8B prediction data remains under `data/profiling/`.

Run the Hydra config checks with `python -m unittest discover -s tests/unit -p test_malleserve_hydra.py -v`.

The legacy macro approximation is retained for historical reproducibility, not native performance estimation. The integration has not yet undergone a full end-to-end execution and fixed-capacity parity check in your Frontier environment.

## Polaris validation (CSV only)

```bash
python -m malleserve.validate validation.polaris=data/malleserve/validation/polaris_summary.csv
```

The only supported experiment execution is `python -m malleserve.debug_run` (including Hydra `-m`).
The only supported validation reference is the consolidated Polaris summary CSV. See `VALIDATION.md`.

## Full fixed-capacity sweep

The sole experiment runner is Hydra's `malleserve.debug_run`. From the Frontier repo root:

```bash
python -m malleserve.debug_run -m \\
  resources=fixed1,fixed2,fixed4,fixed6,fixed8 \\
  policy=fixed \\
  workload=mooncake_r2,mooncake_r4,mooncake_r6,mooncake_r8,mooncake_r10,mooncake_r12,mooncake_r14,mooncake_r16
```

This runs a Cartesian grid of **40** Frontier simulations (five worker counts by eight rates), including cells the original Polaris sweep did not measure. Each writes to a concise directory such as `outputs/malleserve/runs/fixed/fixed4_r10_qwen_qwen3_8b/`, containing `malleserve_summary.json`, `request_metrics.csv`, etc. Frontier's internal `<model>/online_serving/<run-id>` taxonomy is flattened **after** the run. Fixed results live under `runs/fixed/<experiment>_<model>/`, and transient results under `runs/transient/<experiment>_<model>/`. Existing old-format runs are not moved or deleted. Hydra's own metadata stays under `outputs/hydra/`.

You must first generate each missing workload CSV and adjacent manifest with `python -m malleserve.tools.convert_mooncake`. If you only want Polaris-measured cells, run the relevant subsets using the same Hydra interface; the validation CSV retains missing result flags. Do not launch multiple runs concurrently under the same `experiment.name`. Existing output folders may need cleaning before re-running a case because Frontier's own metrics files can otherwise be overwritten.

Validation (once the simulations have finished):

```bash
python -m malleserve.validate validation.polaris=data/malleserve/validation/polaris_summary.csv
```

## Controlled transient recovery modes

The native transient lifecycle supports three controls through
`resources.lifecycle_mode` with `resources=transient policy=transient`; routing is selected independently:

- `baseline_recovery`: physical node rotations, full request recomputation after reclamation.
- `no_reclamation`: one four-GPU candidate joins once and stays online.
- `ideal_recovery`: identical rotation to `baseline_recovery`; zero-time restoration
  of **committed token progress and KV context** with destination KV block
  capacity enforced. This is an experimental **oracle abstraction**, not
  an implemented KV transport protocol. In-flight work not committed at
  the reclaim event is not preserved, and no migration/transfer time is
  charged. Requests can wait for destination memory admission.

For an apples-to-apples 10 RPS comparison:

```bash
python -m malleserve.debug_run \
  resources=transient policy=transient workload=mooncake_r10 \
  resources.lifecycle_mode=ideal_recovery \
  resources.startup_seconds=40 \
  experiment.name=oracle_r10_l60_s40
```

For the six-run routing sweep, use `scheduler.cluster=round_robin` or `kv_aware`;
set `scheduler.cache_weight=0.0` for token-only and `1.0` for KV-aware.
Use `resources.lifecycle_mode=baseline_recovery` or `ideal_recovery`.
Legacy `resources=naive`, `policy=naive`, and `resources.lifecycle_mode=naive` remain available for old runs.

The output `transient_lifecycle.json` includes
`recovery_mode`, `oracle_restored_requests`, `oracle_restored_tokens`
and existing per-replica reclamation diagnostics. `retries` counts
interrupted request assignments in both modes; `lost_processed_tokens`
only counts discarded progress for `baseline_recovery`.

**Validation status:** Unit tests cover core reclamation state accounting,
but full native end-to-end oracle replay against the reference configuration
must be exercised before treating the oracle results as calibrated performance
claims. In particular, review queue admission, KV block occupancy, P95,
logical completion counts and multi-reclamation cases.
