# MalleServe × Frontier: Mooncake trace bridge

This prototype lives in your Frontier fork's malleserve-development branch.
It does not change Frontier's simulation engine or the real MalleServe runtime.

## 1. Install Frontier locally

From the Frontier repository root:

    conda env create -f environment.yml     # once
    conda activate frontier
    python -m pip install -e ".[test]"

A GPU is not needed to execute dummy-timing simulations.

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

The initial script uses a built-in Llama3.1-8B model description,
a simulated A100, one replica / TP=1, vLLM V1 prefix caching and
DUMMY execution-time prediction. Its 8192-block KV capacity is an
arbitrary smoke-test setting, NOT calibrated to Polaris.

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

One GPU only until multiworker scheduling and cache-aware routing are
validated. The stable session_id is a heuristic derived from the
first 512-token block; it is not validated against MalleServe's
multi-replica routing policy.

## 6. Deferred work once Polaris baseline finishes

Calibrate true Qwen3-8B/A100 execution, overhead and physical KV capacity.
Then compare held-out input-rate experiments, request latencies, cache
behavior, completion throughput, SLO goodput and backlog. A k6
experiment can drop or delay scheduled requests; this deterministic
CSV simulates all expected arrivals. Only after stable-worker
validation should we introduce node loss, startup delays, request
interruption and cache-state recovery.

Dummy simulation timings are not valid Polaris performance predictions.

## Tests

    python -m unittest discover -s tests/unit -p 'test_malleserve_mooncake_adapter.py'
