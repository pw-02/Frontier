# Real scheduler-batch decode predictor replay

This is an **opt-in diagnosis**, not a validated performance timing run.
It captures the *actual* vLLM-v1 scheduler-generated MONOLITHIC batches
with exactly one scheduled decode token per request, including their
actual per-request computed contexts and decoding flags. It calls the
same trained execution predictor again on the **same Batch object**
immediately after the first prediction, and saves both predicted timings.

No synthetic context is guessed or substituted. The two calls should
agree if prediction is deterministic for a fixed batch and configuration.
The output breaks model time into aggregate attention, MLP components,
forward launch and modelled host total. This does not independently
measure host runtime on Polaris.

Example (from Frontier repository root, use a fresh experiment name):

```bash
FRONTIER_DECODE_REPLAY_PATH=outputs/malleserve/decode_same_batch.jsonl \\
FRONTIER_DECODE_REPLAY_LIMIT=30 \\
python -m malleserve.debug_run \\
  resources=fixed4 policy=fixed workload=mooncake_r10 \\
  experiment.name=decode_replay_probe_r10
```

**Critical:** This duplicates predictor calls *during simulation*.
Use the resulting run solely for diagnostics, not performance baselines.
Compare original and same_batch_replay from JSONL; look for any nonzero
`delta_model_ms`. Then compare original batch timings/context vectors
with the synthetic-grid output. Differences with synthetic predictions
can reflect fundamentally different context distributions, CUDA-graph
capture sizes, model configuration or runtime state. The earlier
synthetic grid initializes fixed8 while this run initializes fixed4.

Remove the environment variable for all official runs.
