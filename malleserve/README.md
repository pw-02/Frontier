# MalleServe — native Frontier experiments with Hydra

The primary experiment entry point is `malleserve/debug_run.py`. It executes
`frontier.main.main()` **in the same Python interpreter**, so IDE breakpoints
work inside Frontier's simulator, cluster scheduler, replica scheduler and
transient lifecycle. The older `examples/malleserve/` scripts remain for
compatibility and Mooncake trace conversion, not as the primary interface.

## Install and run

```bash
git switch malleserve-development
git pull --ff-only origin malleserve-development
python -m pip install -e ".[test]"
python -m malleserve.debug_run
```

The default `malleserve/conf/config.yaml` composes:

- `workload/mooncake_r10.yaml` (Mooncake 10 RPS; replay an existing CSV and manifest)
- `resources/naive.yaml` (four persistent, rotating four-GPU node candidates)
- `model/qwen3_8b_a100.yaml` (calibrated Qwen3-8B/A100 predictor paths)
- `scheduler/vllm_v1.yaml` (continuous batching, 2048-token batch budget, KV cache)
- `policy/naive.yaml` (activate ready transient nodes; retry on reclamation)

Common Hydra commands:

```bash
# Fixed-4 or fixed-8 baselines; BOTH resource and policy must agree
python -m malleserve.debug_run resources=fixed4 policy=fixed
python -m malleserve.debug_run resources=fixed8 policy=fixed

# Naive, but with zero startup delay
python -m malleserve.debug_run resources.startup_seconds=0

# Change workload (CSV and adjacent manifest must exist)
python -m malleserve.debug_run workload=mooncake_r2

# Change native vLLM batch budget
python -m malleserve.debug_run scheduler.max_tokens_in_batch=1024

# Multirun sensitivity study; each job executes native Frontier independently
python -m malleserve.debug_run -m resources.startup_seconds=0,20,40
```

Results are written under `outputs/examples/malleserve/<experiment.name>/`.
By default the name is `<resource-label>_<workload-name>`. For multiruns
**override `experiment.name` or include the swept parameter in its name**,
for example:

```bash
python -m malleserve.debug_run -m \
  resources.startup_seconds=0,20,40 \
  'experiment.name=naive_r10_s${resources.startup_seconds}'
```

Do not run separate jobs with identical output names concurrently: Frontier
writes to its named metrics root. Hydra writes its own sweep metadata to
`outputs/hydra`, whereas Frontier metrics are in the path above.

## IDE debugging

Open `malleserve/debug_run.py` and choose **Debug Python File**. Set a
breakpoint in `frontier/transient_lifecycle.py` at `NativeTransientLifecycle.change`
or `frontier/simulator.py` in `Simulator.run`. Run with no CLI overrides
for the default naive scenario, or configure IDE launch arguments such as
`resources=fixed4 policy=fixed`. Unlike the historical Bash wrappers, the
simulation is not launched as a child process.

## Trace creation and resource availability

Create a 10 RPS workload CSV from the prepared Mooncake JSON:

```bash
python examples/malleserve/convert_mooncake.py \
  --dataset data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json \
  --rate 10 --warmup-seconds 30 --measurement-seconds 300 \
  --output data/malleserve/mooncake_r10_330s.csv
```

This produces the CSV of arrival timestamps, prompt/output lengths and
prefix-block hashes, together with an adjacent `.manifest.json` recording
counts and max token length. The converter generates *constant-rate* arrivals;
the timestamps are not original Mooncake arrival times. Frontier's trace
replay uses the CSV to schedule every request in its native DES.

For transient scenarios a *separate* JSONL of node availability and
reclamations is generated from `resources.node_lifetime_seconds` and
`resources.duration_seconds`. Fixed scenarios do not use availability
events. Keep workload and model constant when comparing resource policies.

## Notes / limitations

Hydra is now the primary MalleServe configuration interface, **not** the
global configuration system for all of Frontier. The Python entry point
translates selected Hydra groups to Frontier's existing CLI dataclasses
and calls the native engine in-process. No separate timing simulator is
introduced. Only fixed and naive transient policies are implemented; native
ideal recovery and clairvoyant policies still require work.

Full environment-level CPU/GPU simulation and fixed-parity validation remain
to be completed. The illustrative 40-second startup delay has not been
validated against Polaris.
