# Exploratory transient-capacity simulator for MalleServe

## Status: experimental macro model — NOT native Frontier elastic serving

This first implementation tests **the lifecycle and experiment design** for
the MalleServe research question: how much *dependable fixed-equivalent*
serving capacity can be obtained from continuously rotating transient GPUs?

**Important fidelity warning:** This is an event-driven **macro proxy**
that runs inside the Frontier repository, but NOT through Frontier's native
GPU-stage / vLLM continuous-batching scheduler. It uses per-request execution
durations extracted from a calibrated, low-load Frontier simulation, then
models each worker as a processor-sharing server. Such durations **are not
isolated GPU work**. Consequently, neither absolute latency nor capacity
nor the numerical benefit of a policy is currently validated. Treat output
as engineering smoke tests and exploratory hypotheses only.

We deliberately keep the core Frontier simulator unchanged rather than
silently break its fixed-replica event invariants. True native dynamic
replica lifecycle, batch cancellation and KV-state semantics are still
required before producing defensible paper-scale conclusions.

## Policies and comparison

All modes replay the **same Mooncake request content and fixed arrival times**.
For the transient modes, **two separate candidate nodes each provide four
GPUs**, but the default synthetic schedule makes only one node available
at a time. At each lifetime boundary, the current node is reclaimed and
the next node becomes available.

| Mode | Capacity | Reclamation | Start/recovery |
| --- | --- | --- | --- |
| fixed_persistent | Four permanent GPUs | None | Always ready |
| fixed_stable | Eight permanent GPUs | None | Always ready |
| naive | Four permanent + one available four-GPU node | Kill active work, retry from scratch; lose local state | Start each new candidate on arrival; pay startup delay; activate all ready workers |
| ideal | Same physical availability as naive | Preserve remaining work and instantly migrate | Zero startup and migration overhead |

The naïve routing policy is deterministic round-robin over *ready workers*,
including workers already processing requests. Once routed, a request stays
on its worker until completed or reclaimed. There is no risk awareness,
standby withholding or future-lifetime knowledge.

This version has **no KV-cache working-set model**: optional
--cold-start-work-seconds charges a simple first-request proxy penalty.
Do not interpret this as genuine LMCache or vLLM prefix-cache behavior.

Reclamation is instantaneous at the specified timestamp. There is no
preemption warning, no server grace/drain and no scheduler reacquisition
delay beyond the configured initialization assumption.

## 1. Build a service proxy from an existing calibrated Frontier run

From your Frontier checkout on the malleserve-development branch:

    cd ~/projects/Frontier
    conda activate frontier

Use a low-load native Frontier run with *complete* request-level metrics
(e.g. your four-worker / two-rps validation):

    METRICS=$(find outputs/examples/malleserve/validation_w4_r2 \
      -name request_metrics.csv -print -quit)

    python examples/malleserve/transient/service_profile.py \
      --trace data/malleserve/mooncake_r2_330s.csv \
      --request-metrics "$METRICS" \
      --output data/malleserve/transient_proxy_w4r2.json

This tool validates the token lengths and IDs, then exports
request_model_execution_time (milliseconds) as a **proxy** in seconds.
It deliberately does not claim that this is additive independent GPU work.

## 2. Build an offered-rate trace

Reuse the existing Mooncake converter; it preserves prompt/output lengths,
prefix block identity and disjoint 30s warmup / 300s measurement slices:

    python examples/malleserve/convert_mooncake.py \
      --dataset data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json \
      --rate 10 --warmup-seconds 30 --measurement-seconds 300 \
      --output data/malleserve/mooncake_r10_330s.csv

## 3. Compare four systems under rotating node availability

    python examples/malleserve/transient/simulate.py \
      --trace data/malleserve/mooncake_r10_330s.csv \
      --service-profile data/malleserve/transient_proxy_w4r2.json \
      --startup-seconds 40 \
      --lifetime-seconds 60 \
      --output-dir outputs/examples/malleserve/transient_r10_l60

**40 seconds is an illustrative startup assumption, NOT a Polaris
measurement.** Replace it with your actual model startup/readiness timing
when available. Try 0, 20, 40 and 80 seconds as a sensitivity study.

Outputs:

    outputs/examples/malleserve/transient_r10_l60/
      comparison.csv
      manifest.json
      fixed_persistent/
        summary.json
        requests.csv
        worker_events.jsonl
      fixed_stable/
        ...
      naive/
        ...
      ideal/
        ...

The summary distinguishes completed-window throughput from SLO goodput
that allows completion during drain, and includes failed-to-complete
arrivals in the SLO denominator. It also records startup counts, worker
reclaims, interrupted attempts and wasted proxy work.

If a workload overloads the macro model, some requests remain outstanding
at the end of the configured drain; these are counted as not meeting SLO.
Increase --drain-seconds if necessary and report the difference.

## 4. Sweep workload rates and transient lifetimes

    python examples/malleserve/transient/run_sweep.py \
      --dataset data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json \
      --service-profile data/malleserve/transient_proxy_w4r2.json \
      --rates 8 10 12 14 16 \
      --lifetimes 30 60 120 \
      --startup-seconds 40 \
      --output-dir outputs/examples/malleserve/transient_sweep

For 5 rates x 3 lifetimes, this produces 15 scenarios x 4 policy modes,
plus comparison.csv and the plots in each scenario when Matplotlib
is installed.

## 5. Custom resource-availability trace

Use --schedule-jsonl PATH instead of the synthetic lifetime pattern.
Each line must be strict JSON, e.g.:

    {"timestamp":0,"node_id":"transient-0","action":"available"}
    {"timestamp":90,"node_id":"transient-0","action":"reclaim"}
    {"timestamp":90,"node_id":"transient-1","action":"available"}
    {"timestamp":150,"node_id":"transient-1","action":"reclaim"}

Each candidate node maps to --gpus-per-transient-node logical GPUs.
The mode fixed_stable intentionally ignores this schedule. The other
two modes use the exact same schedule, including short availability
windows that expire before a worker finishes initializing.

**Availability is an exogenous trace**: unlike ordinary cloud
autoscaling, the serving policy cannot extend a GPU's lifetime.

## Plot outputs and test against native Frontier

Plot the four-policy comparison and request-arrival-window timelines:

    python examples/malleserve/transient/plot.py \
      --scenario outputs/examples/malleserve/transient_r10_l60

Plot the offered-rate sweep at each resource lifetime:

    python examples/malleserve/transient/plot.py \
      --sweep outputs/examples/malleserve/transient_sweep

Requires Matplotlib. The plots prominently say macro model.

**Mandatory validation gate before relying on numerical predictions:** compare
the macro's fixed-worker controls with the exact same native Frontier trace:

    NATIVE4=$(find outputs/examples/malleserve/validation_w4_r10 \
      -name malleserve_summary.json -print -quit)
    NATIVE8=$(find outputs/examples/malleserve/validation_w8_r10 \
      -name malleserve_summary.json -print -quit)

    python examples/malleserve/transient/validate_baselines.py \
      --macro-scenario outputs/examples/malleserve/transient_r10_l60 \
      --native-fixed-4 "$NATIVE4" \
      --native-fixed-8 "$NATIVE8" \
      --max-relative-error 0.20

The gate returns nonzero on material differences in fixed-worker p95/p99
or SLO goodput. **Failure means the macro model cannot be used for
quantitative transient claims.** Passing is only a prerequisite, not
validation of the native transient lifecycle.

## 6. Test the state machine

    python -m unittest discover -s tests/unit \
      -p test_malleserve_transient_macro.py -v

The tests cover worker rotations, cancelled startup, immediate reclaim,
lost work and retries, perfect continuation, and denominator correctness.

## Next research step: native Frontier fidelity

Before using these numbers in a paper, the important next engineering phase
is integrating availability into Frontier's native monolithic scheduler.
At minimum it needs:

- Event-driven ready masks for precreated replica schedulers, not just a
  different fixed cluster size per simulation.
- Safe cancellation of in-flight per-replica GPU stage events, including
  request epochs and KV allocator teardown.
- Request restart with correct metrics accounting (once per client request)
  and requeued work that consumes genuine native batch service time.
- An ideal reference preserving interrupted request progress and cache state.
- A prefix-cache cold/warm state model, measured Polaris startup timing,
  and a test showing fixed modes reproduce the existing Frontier baselines
  before any dynamic-churn claims.

The current macro simulation is **not** a substitute for these features,
but provides reusable event schedules, metrics definitions and policy tests.
