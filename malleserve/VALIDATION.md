# Polaris vs Frontier: fixed-capacity validation

The only simulation entry point is **Hydra** (`python -m malleserve.debug_run`).
The only Polaris comparison input is the consolidated CSV summary.

## 1. Prepare your environment

```bash
git switch malleserve-development
git pull --ff-only origin malleserve-development
pip install -e ".[test]"
```

The Hydra workloads default to
`data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json` as the **prepared**
Mooncake dataset. If the matching trace CSV and `.manifest.json` exist,
they are reused and the source JSON is not needed. Otherwise the Hydra
runner automatically generates both from the prepared JSON using the
configured rate, 30-second warmup, 300-second measurement period and
128-token output cap. Override `workload.dataset=/absolute/path.json`
if your dataset is elsewhere.

Trace generation fails rather than overwriting a partial CSV/manifest
pair, or reusing a trace whose rate/block size differs from the YAML.
The generator is part of the Hydra experiment workflow; no preliminary
trace-conversion command is required.

## 2. Run one experiment and debug

```bash
python -m malleserve.debug_run resources=fixed4 policy=fixed workload=mooncake_r10
```

For IDE breakpoints, open `malleserve/debug_run.py`; the real Frontier
simulation executes inside that Python interpreter.

## 3. Full sweep (5 GPU counts × 8 arrival rates = 40 cells)

```bash
python -m malleserve.debug_run -m \
  resources=fixed1,fixed2,fixed4,fixed6,fixed8 \
  policy=fixed \
  workload=mooncake_r2,mooncake_r4,mooncake_r6,mooncake_r8,mooncake_r10,mooncake_r12,mooncake_r14,mooncake_r16
```

Hydra's basic launcher executes these sequentially. Results are under
`outputs/malleserve/runs/fixed4_r10/`, etc. A completed run contains
`malleserve_summary.json`, `metrics_ground_truth.jsonl` and a
`run_complete.json` marker holding a SHA-256 fingerprint of the composed
configuration and request trace. **Rerunning the same sweep skips only
matching completed runs** (`experiment.skip_completed=true` by default).
Changed parameters or input files do **not** silently reuse an old run.

An incomplete run or an old output directory without the matching
completion marker is *not* automatically overwritten: remove/rename
that specific run directory, or choose a new `experiment.name`.
This deliberately prevents accidental mixing of results. To rerun a
previously completed experiment, remove its output directory first.
Don't run concurrent jobs with the same experiment name.

## 4. Validate against your Polaris CSV

Save the consolidated Polaris CSV as
`data/malleserve/validation/polaris_summary.csv` or pass an absolute path:

```bash
python -m malleserve.validate \
  validation.polaris=data/malleserve/validation/polaris_summary.csv
```

The comparison appears in
`outputs/malleserve/validation/fixed_capacity_comparison.csv`.
Duplicate Polaris repeats remain separate rows and missing Frontier
cells are flagged. A valid Polaris load needs completed status, at
least 99% submitted requests, zero HTTP failures and explicitly zero
dropped iterations. Empty dropped-iteration fields are unknown.

**Interpretation:** Polaris is HTTP client end-to-end latency; Frontier
reports modeled engine-side completion latency. These are not strictly
the same boundary. Examine differences in P50/P95/P99, SLO goodput,
throughput and backlog across the matrix, not just one cell. Request
output-length variation, GPU KV allocation and decode model bias need
separate fidelity checks before claiming predictive accuracy.
