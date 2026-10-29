## Preferred input: Polaris summary CSV

The `fixed_capacity` analysis CSV is sufficient for **sweep-level**
validation; you do not need to copy the full `runs/` tree just to compare
latency, throughput, SLO and backlog.

```bash
python -m malleserve.validate \
  validation.polaris=/path/to/polaris_validation_summary.csv \
  validation.frontier=outputs/examples/malleserve \
  validation.output=outputs/malleserve/validation/fixed_capacity_comparison.csv
```

Each source row is preserved, including repeated runs at the same
GPU/RPS combination (e.g. two 6-GPU/12-RPS runs). A missing or failed
Polaris run is not treated as a valid accuracy target. The validity flag
requires `validity_status=completed`, `submission_pct>=99`, zero
failures and explicitly zero dropped iterations. Empty dropped-iteration
cells mean **unknown**, not zero. Frontier results are matched by the
resource count and RPS using `fixed{workers}_mooncake_r{rate}`.
Duplicate Polaris repeats share that Frontier result and are separately
preserved in the output.

This is a **summary-level** comparison only. For request-level latency
distributions, prefill/decode attribution, matching actual output lengths,
or debugging missing arrivals, retain the full logs from Polaris.

# Fixed-capacity Polaris ↔ Frontier validation

Run Frontier with **fixed** Hydra resources and replay the *same* Mooncake CSV for
the corresponding rate. Import Polaris data as ZIP archives or as its
`experiments/fixed_capacity/runs/` tree; do not store large telemetry logs in git.

## First cell: fixed workers 4, rate 10

```bash
# From Frontier repository root, with environment installed:
python -m malleserve.debug_run resources=fixed4 policy=fixed workload=mooncake_r10
python -m malleserve.validate \
  validation.polaris='/path/to/fixed-workers-4-r10.zip' \
  validation.frontier=outputs/examples/malleserve
```

Run the other cells by changing `resources=fixed4` to `fixed8` or
`resources=fixed4 resources.gpus=2 experiment.name=fixed2_mooncake_r10`
and `workload=mooncake_r10` to the matching trace. Hydra currently has
workload groups for rates 2/10/12; add groups or override
`workload.trace=data/malleserve/mooncake_r6_330s.csv` plus unique
`experiment.name` for other rates. Always generate the trace and adjacent
manifest first using `malleserve.tools.convert_mooncake`.

For the entire Polaris sweep, transfer `runs/` to a local folder (or one ZIP
at a time), simulate all matching fixed-capacity cells, then:

```bash
python -m malleserve.validate \
  validation.polaris=/path/to/fixed_capacity/runs \
  validation.frontier=outputs/examples/malleserve \
  validation.output=outputs/malleserve/validation/full_sweep.csv
```

The comparison CSV records validity flags, measured request counts,
Polaris mean and P95 **client** latency, Frontier mean and P95 simulated
**engine** latency, SLO attainment, SLO goodput, and backlog. Missing
Frontier cells are explicit and not silently fabricated.

## Critical fidelity checks

1. Same model/quantization, GPU SKU, 1-GPU workers and routing semantics.
2. Same trace **request sequence, arrival rate, warmup/measurement window**,
   prompt lengths and capped completion lengths; compare manifests.
3. Same vLLM prefix caching behavior, available GPU KV blocks (real GPU
   utilization setting of 0.85 does **not** imply 8192 blocks), batching and
   CUDA graph effects.
4. Compare client latency and simulated engine latency as **different**
   measurements. Diagnose queue wait, prefills, decode, caching and
   networking before treating residual as execution-time calibration.
5. Compare SLO goodput, p50/p95/p99, request conservation and saturation
   across RPS/worker counts. Do not tune against only one cell.

The supplied fixed-workers-4-r10 ZIP reports 3300 total requests, 3000
successful measured checks, zero failed measured requests, and 8.903 seconds
P95 client latency. That run cannot by itself establish simulator accuracy.

The new Hydra validation adapter has not yet been executed on the full sweep.
