# Native Frontier MalleServe experiment configurations

Use the same Frontier engine for **fixed-4**, **fixed-8**, and **naive transient** comparisons. These presets inherit common settings instead of duplicating long CLI argument lists.

## Quick start

From the repository root, after updating `malleserve-development` and activating the Frontier environment:

```bash
python examples/malleserve/transient/run_native_scenario.py --config examples/malleserve/transient/scenarios/fixed4.yaml --dry-run
python examples/malleserve/transient/run_native_scenario.py --config examples/malleserve/transient/scenarios/fixed4.yaml
python examples/malleserve/transient/run_native_scenario.py --config examples/malleserve/transient/scenarios/fixed8.yaml
python examples/malleserve/transient/run_native_scenario.py --config examples/malleserve/transient/scenarios/naive.yaml
```

The fixed cases omit `--transient_schedule_file` completely and create 4 or 8 always-ready native schedulers. The naive case precreates 4 persistent and 2x4 candidate replica slots; only currently ready candidates can accept traffic. Every case retains the same vLLM v1 scheduling and Qwen3-8B/A100 calibrated execution-time predictor in `run_native_transient.sh`. The source of truth for model, scheduling, prefix-cache and predictor defaults remains that shell runner; the YAML currently controls **scenario/workload/resources/SLO**, not all Frontier CLI flags.

`common.yaml` defines the trace and SLO. `fixed4.yaml`, `fixed8.yaml`, and `naive.yaml` inherit it using `extends`. Relative `extends` paths resolve from the YAML file itself; relative workload paths resolve from the Frontier repository root. Copy a scenario file, give it a unique experiment name, and override only the keys that differ.

Results appear at `outputs/examples/malleserve/<experiment.name>/`, with per-request metrics and `malleserve_summary.json`; transient runs additionally emit `transient_lifecycle.json`.

## Where the workload trace comes from

The input is a **prepared Mooncake JSON dataset** (individual request lengths and 512-token content-block identities). The checked-in `examples/malleserve/convert_mooncake.py` converts it into a Frontier CSV plus adjacent manifest. Example for 10 requests/second:

```bash
python examples/malleserve/convert_mooncake.py \
  --dataset data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json \
  --rate 10 --warmup-seconds 30 --measurement-seconds 300 \
  --output data/malleserve/mooncake_r10_330s.csv
```

The converter generates 300 warmup and 3000 measured arrivals (3300 total), with deterministic timestamps spaced 0.1 seconds apart. Original Mooncake timestamps are **not** preserved: this is open-loop constant-rate load generation to match the fixed-duration k6 experiment. It uses disjoint dataset slices with a two-record gap; output length is capped at 128 tokens by default. The trace CSV columns include `arrived_at`, `num_prefill_tokens`, `num_decode_tokens`, `session_id`, `block_hash_ids`, `phase`, and source provenance. SHA-derived 16-token subblock IDs represent possible shared prefixes; they are not measured cache hits.

A matching `mooncake_r10_330s.manifest.json` stores warmup/measurement request counts and `required_frontier_max_tokens`. The native shell runner reads this field to prevent silently clipping large prompts. Frontier's `TraceReplayRequestGenerator` uses the CSV arrivals and token counts to create simulator requests; 16-token block IDs allow its prefix cache to recognize repeated prefixes, if enabled. A resource availability schedule is a **separate JSONL**: events `available` and `reclaim` for named transient nodes. The scenario runner generates this automatically for naive transient cases; fixed cases don't use it.

For 2 or 12 RPS, generate separate CSVs using `--rate 2` or `--rate 12`; update the YAML workload trace path. Changing resource policy should **not** change the trace file. Do not compare scenarios with different request datasets/timestamps and attribute the difference purely to resource management.

## Validation caveats

The runner integration needs execution in your Frontier environment. Validate fixed-4/8 parity against known native runs and request conservation before reporting performance. The naive policy discards per-replica state on reclaim and retries requests; native ideal continuation and clairvoyant policies are not implemented. The 40-second startup delay is an assumption rather than measured Polaris startup data. Existing decode bias versus Polaris remains a calibration limitation.
