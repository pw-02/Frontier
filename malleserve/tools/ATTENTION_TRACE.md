# Decode-attention feature and calibration trace

Enable this **during a new diagnostic-only simulation**, alongside the same-batch
capture. It logs actual attention decode feature vectors **from the predictor's
existing call**, without making an extra prediction for the trace.

```bash
FRONTIER_DECODE_REPLAY_PATH=outputs/malleserve/decode_same_batch_v2.jsonl \\
FRONTIER_ATTENTION_TRACE_PATH=outputs/malleserve/decode_attention_live.jsonl \\
FRONTIER_ATTENTION_TRACE_ORIGIN=live \\
FRONTIER_DECODE_REPLAY_LIMIT=30 \\
python -m malleserve.debug_run resources=fixed4 policy=fixed \\
  workload=mooncake_r10 experiment.name=decode_attention_trace_r10
```

To trace the **reconstructed batches and 15-point grid**, without re-running
the live simulation:

```bash
python -m malleserve.tools.decode_predictor_reconcile \\
  --captured outputs/malleserve/decode_same_batch_v2.jsonl \\
  --polaris-summary data/7796796.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/summary.csv \\
  --attention-trace outputs/malleserve/decode_attention_reconstructed.jsonl \\
  --output-prefix decode_attention_reconcile
```

Compare `decode_attention_live.jsonl` and
`decode_attention_reconstructed.jsonl`. Per record:
- `feature_batch_size` and `feature_rounded_kv_cache_size` are the
  **actual prediction features** passed to the attention model.
- `raw_attention_decode_ms_per_layer`, `calibration_branch`,
  `calibration_scale` and `scaled_attention_decode_ms_per_layer`
  reveal whether the difference arises before or after calibration.
- Per request, `num_processed_tokens`, `num_processed_decode_tokens`,
  `num_emitted_decode_tokens`, and `effective_attention_context_tokens`
  make differences in decode progression visible.

**Important:** The existing capture only stored schedule-time context, not
processed/emitted decode counters. The reconstruction creates fresh requests,
which can select a different late-decode calibration branch. This diagnostic
is intended to test that hypothesis, **not** to justify calibration changes.

The tracing environment variables must be unset for official experiments.
