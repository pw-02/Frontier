# MalleServe × Frontier: Mooncake trace bridge

This prototype lives in your Frontier fork's malleserve-development branch.
It does not change Frontier's simulation engine or the real MalleServe runtime.

## 1. Install Frontier locally

From the Frontier repository root:

    conda env create -f environment.yml     # once
    conda activate frontier
    python -m pip install -e ".[test]"

A GPU is not needed for dummy or measured-profile simulations. GPU operator profiling was performed on Polaris; prediction training and trace replay run on a CPU-only machine.

## 2. Obtain the prepared Mooncake dataset

Use the prepared JSON from MalleServe:

    datasets/prepared/mooncake_toolagent_qwen3_8b_n12000.json

If the prepared file is only on Polaris, copy it to your local Frontier
checkout. Replace POLARIS_SSH_HOST with your usual Polaris SSH hostname
or alias:

    mkdir -p data/malleserve
    scp pwatters@POLARIS_SSH_HOST:/lus/grand/projects/BFTrainer/pwatters/malle-serve/datasets/prepared/mooncake_toolagent_qwen3_8b_n12000.json data/malleserve/

Do not commit this possibly large generated JSON file to the fork.

## 3. Create a short functional test trace

    python examples/malleserve/convert_mooncake.py       --dataset data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json       --rate 2 --warmup-seconds 2 --measurement-seconds 5       --output data/malleserve/mooncake_r2_smoke.csv

The converter writes a Frontier trace CSV and an adjacent
mooncake_r2_smoke.manifest.json. The CSV includes arrival time,
input/output lengths, session_id and block_hash_ids; the extra
provenance columns are ignored by Frontier.

This matches the fixed_capacity k6 workload's open-loop fixed-rate
arrivals and disjoint dataset slices (including the two-record gap).
It ignores original Mooncake arrival timestamps, caps output length
at 128 tokens, and does not cycle requests.

Each 512-token Mooncake block ID expands into 32 deterministic
16-token subblock hashes, preserving prefix equality. Partial terminal
blocks are not considered fully cacheable. This describes prefix
opportunity, not actual cache hits.

## 4. Run the simulation

    bash examples/malleserve/run_mooncake.sh       data/malleserve/mooncake_r2_smoke.csv smoke_r2

The existing runner uses the Qwen3-8B model configuration, a simulated
A100, one replica / TP=1, vLLM V1 prefix caching and **DUMMY** 1 ms
execution timings. Its 8192-block KV capacity is an arbitrary functional
smoke-test setting, **not** a measured Polaris capacity.

Inspect:

    find outputs/examples/malleserve/smoke_r2 -name request_metrics.csv

Frontier writes into a nested model/workload/run-id directory. For
the included dummy online-serving example, the files are under:

    outputs/examples/malleserve/smoke_r2/llama3_1_8b/online_serving/smoke_r2/

The analyzer automatically discovers that single nested run directory
when you provide the outer metrics root. It writes malleserve_summary.json
next to Frontier's request_metrics.csv.

The runner records:
- metrics_ground_truth.jsonl: simulated request arrivals/completions, in seconds
- request_metrics.csv: latency/queue/TTFT and cache metrics; exported times in milliseconds
- system_metrics.json: Frontier's aggregate metrics
- malleserve_summary.json: window-aware MalleServe analysis

You can rerun analysis without the simulator:

    python examples/malleserve/analyze_mooncake.py       --manifest data/malleserve/mooncake_r2_smoke.manifest.json       --metrics-dir outputs/examples/malleserve/smoke_r2       --slo-seconds 30

SLO goodput allows requests to finish during drain after the
measurement boundary; completion throughput inside the window does
not. The analyzer also reports p95 queue wait and backlog at the
measurement boundary.

## 5. Full fixed-duration rehearsal

When the short smoke passes:

    python examples/malleserve/convert_mooncake.py       --dataset data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json       --rate 2 --warmup-seconds 30 --measurement-seconds 300       --output data/malleserve/mooncake_r2_330s.csv

    bash examples/malleserve/run_mooncake.sh       data/malleserve/mooncake_r2_330s.csv mooncake_r2_330s

Start with one **simulated replica** for the functional check, then use
four simulated replicas for calibration. No physical GPU is needed to run
the simulator. The stable session_id is a heuristic derived from the first
512-token block; it is not validated against MalleServe's multi-replica
routing policy.

## 6. Qwen3-8B / A100: Polaris-to-Frontier calibration

**Goal:** replace the 1 ms dummy execution times with measured A100 CUDA
operator timings, replay the same Mooncake workload on four simulated
replicas, and compare predictions with the measured MalleServe baseline.

The `malleserve-development` branch contains the Polaris Qwen3-8B,
TP=1, BF16 profiles (FlashInfer attention, `VLLM_KV_CACHE_LAYOUT=NHD`):

| Dataset | File | Coverage |
| --- | --- | --- |
| Linear operators | `data/profiling/compute/a100/Qwen/Qwen3-8B/linear_op.csv` | 20 token-count configurations |
| Standard attention | `data/profiling/compute/a100/Qwen/Qwen3-8B/attention.csv` | 188 shapes (143 prefill, 45 decode) |
| Mixed-length prefill | `data/profiling_mixed/compute/a100/Qwen/Qwen3-8B/attention_mixed.csv` | Variable-length prefill batches |
| Prefill + decode mixed | `data/profiling_mixed/compute/a100/Qwen/Qwen3-8B/attention_true_mixed.csv` | Co-located mixed batches |

These are **operator-level GPU measurements**, not complete-model serving
latencies. No Polaris allocation or CUDA stack is needed for the steps below.

### 6.1 Pull the branch locally

From a local Frontier clone (substitute your own checkout path):

~~~bash
cd /path/to/your/Frontier
git switch malleserve-development
git pull --ff-only origin malleserve-development
conda activate frontier
export PYTHONPATH="$PWD"
python -c "import frontier, pandas, sklearn; print('Frontier imports OK')"
ls data/profiling/compute/a100/Qwen/Qwen3-8B/
ls data/profiling_mixed/compute/a100/Qwen/Qwen3-8B/
~~~

Follow [Section 1](#1-install-frontier-locally) if the local environment is
not installed. The prepared Mooncake JSON must also exist locally; see
[Section 2](#2-obtain-the-prepared-mooncake-dataset).

### 6.2 Combine all attention profiles

The standard and mixed campaigns were written under different output
roots. Their respective `attention_combined.csv` files do **not**
automatically include rows from the other campaign. Merge the three
source tables explicitly; preserve the originals.

~~~bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("data/profiling/compute/a100/Qwen/Qwen3-8B")
mixed = Path("data/profiling_mixed/compute/a100/Qwen/Qwen3-8B")
files = [
    root / "attention.csv",
    mixed / "attention_mixed.csv",
    mixed / "attention_true_mixed.csv",
]
frames = []
for f in files:
    df = pd.read_csv(f)
    assert "measurement_type" in df, f
    assert set(df["measurement_type"].dropna()) == {"CUDA_EVENT"}, f
    print(f"{f}: {len(df)} rows")
    frames.append(df)

combined = pd.concat(frames, ignore_index=True, sort=False)
output = root / "attention_calibrated.csv"
combined.to_csv(output, index=False)
print(f"Saved {len(combined)} rows to {output}")
print("True-mixed rows:",
      combined.get("is_true_mixed_batch",
                   pd.Series(dtype=bool)).fillna(False).sum())
PY
~~~

Use `attention_calibrated.csv` to train both standard and mixed-batch
attention models. True-mixed rows support the
`attn_decode_in_mixed` predictor required for co-located
prefill+decode batches.

### 6.3 Generate a non-dummy, four-replica runner

Preserve `examples/malleserve/run_mooncake.sh` as the functional
dummy-timing smoke test. Create a **separate local runner** by replacing
the following exact arguments:

~~~bash
python - <<'PY'
from pathlib import Path

src = Path("examples/malleserve/run_mooncake.sh")
dst = Path("examples/malleserve/run_mooncake_calibrated.sh")
script = src.read_text()
replacements = {
    "--cluster_config_num_replicas 1":
        "--cluster_config_num_replicas 4",
    "--random_forrest_execution_time_predictor_config_enable_dummy_mode":
        "--no-random_forrest_execution_time_predictor_config_enable_dummy_mode",
    "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms 1.0":
        "--random_forrest_execution_time_predictor_config_atten_input_file "
        "data/profiling/compute/a100/Qwen/Qwen3-8B/attention_calibrated.csv",
    "Timing model: DUMMY (do not compare numerical values with Polaris)":
        "Timing model: measured A100 GPU; CPU overhead not calibrated",
}
for old, new in replacements.items():
    if old not in script:
        raise RuntimeError(f"Original runner changed; missing: {old}")
    script = script.replace(old, new)
dst.write_text(script)
print(f"Created {dst}")
PY

grep -E 'num_replicas|enable_dummy_mode|atten_input_file|num_blocks' \
  examples/malleserve/run_mooncake_calibrated.sh
~~~

This uses the default Qwen3-8B/A100 `linear_op.csv`, selects the
new combined attention dataset, disables dummy timing, and configures
four one-A100 simulated replicas. The first simulation automatically
trains missing random-forest predictors and caches them; training can
be CPU-intensive. The optional standalone training commands are:

~~~bash
python -m frontier.training.cli linear_op \
  --dataset_path data/profiling/compute/a100/Qwen/Qwen3-8B/linear_op.csv \
  --model_name Qwen/Qwen3-8B --device a100 --tensor_parallel_size 1 \
  --measurement_type CUDA_EVENT --output_dir cache \
  --num_training_job_threads 4 --k_fold_cv_splits 5

python -m frontier.training.cli attention \
  --layer_dataset_path data/profiling/compute/a100/Qwen/Qwen3-8B/attention_calibrated.csv \
  --compute_dataset_path data/profiling/compute/a100/Qwen/Qwen3-8B/linear_op.csv \
  --model_name Qwen/Qwen3-8B --device a100 --tensor_parallel_size 1 \
  --measurement_type CUDA_EVENT --output_dir cache \
  --num_training_job_threads 4 --k_fold_cv_splits 5
~~~

You may skip standalone training and allow the simulator to train
on demand. **Do not compare dummy-mode results with Polaris.**

**Uncalibrated runner assumptions:** the existing 8192 KV blocks
**per replica** are a placeholder; `sticky_round_robin` routing might
not match MalleServe's cache-aware dispatching; `decode_cuda_graph_mode
none` might differ from the serving runtime; and CPU scheduler /
kernel-launch overhead is still skipped by default. Thus this is the
**first calibration pass**, not validated performance reproduction.

### 6.4 Short trace test

Confirm the prepared dataset from Section 2 is present, then:

~~~bash
python examples/malleserve/convert_mooncake.py \
  --dataset data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json \
  --rate 10 --warmup-seconds 2 --measurement-seconds 10 \
  --output data/malleserve/mooncake_r10_test.csv

bash examples/malleserve/run_mooncake_calibrated.sh \
  data/malleserve/mooncake_r10_test.csv calibrated_r10_test

find outputs/examples/malleserve/calibrated_r10_test \
  -name malleserve_summary.json -o -name request_metrics.csv
~~~

The converter generates both the trace CSV and the adjacent
`mooncake_r10_test.manifest.json` required by the runner.

### 6.5 Full 30 + 300 s replay at 10 requests/s

Once the short non-dummy run completes:

~~~bash
python examples/malleserve/convert_mooncake.py \
  --dataset data/malleserve/mooncake_toolagent_qwen3_8b_n12000.json \
  --rate 10 --warmup-seconds 30 --measurement-seconds 300 \
  --output data/malleserve/mooncake_r10_330s.csv

bash examples/malleserve/run_mooncake_calibrated.sh \
  data/malleserve/mooncake_r10_330s.csv calibrated_r10_330s
~~~

The runner invokes `examples/malleserve/analyze_mooncake.py`;
find `malleserve_summary.json`, `request_metrics.csv` and
`system_metrics.json` below
`outputs/examples/malleserve/calibrated_r10_330s/`.
Frontier simulates all planned arrivals; a real k6 run can drop or
delay requests, so compare equivalent measurement windows.

### 6.6 Compare to the Polaris baseline

The initial target is the MalleServe **four-A100, 10-request/s**
fixed-worker experiment, with 30 s warmup, 300 s measurement and a
30 s end-to-end SLO:

| Metric | Measured Polaris baseline |
| --- | ---: |
| Measured submitted requests | 3,000 |
| p95 end-to-end latency | 8.81 s |
| p99 end-to-end latency | 13.76 s |
| Requests meeting the 30 s SLO | 100% |
| SLO goodput | 10 req/s |
| Token-weighted prefix-cache hit rate | ~58.5% |

Compare TTFT, latency percentiles, completion throughput, SLO goodput,
queue delay, backlog, worker load and prefix-cache utilization—not
throughput alone. Investigate **scheduling and KV-cache fidelity**
before expanding the operator profiling grid. Additional CPU overhead
or targeted decode/prefill profiling may be needed once specific
prediction errors are identified.

## 7. Deferred fidelity work

Calibrate true KV-cache capacity, memory headroom, CPU launch and
scheduler overhead, runtime CUDA-graph policy, cache-aware multiworker
routing and any model-kernel discrepancies. Once the fixed-worker
baseline matches, validate held-out input rates. Only then introduce
node loss, startup delays, request interruption and cache recovery.

Dummy execution timings are **not** valid Polaris performance predictions.

## Tests

    python -m unittest discover -s tests/unit -p 'test_malleserve_mooncake_adapter.py'
