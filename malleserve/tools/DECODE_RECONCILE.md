# Live-batch reconstruction vs fixed-4 predictor

Run after the live same-batch capture probe and the Polaris decode
benchmark. Reconstruct each captured decode batch using its **actual
batch size, current-context vector, decoding flags and token counts**.
Initialize the predictor using the native fixed-4 MalleServe configuration
instead of the earlier synthetic benchmark's fixed-8 configuration.
Compare aggregate and component predictions and mark mismatches. Then
run the 15-point homogeneous Polaris grid **with the same predictor
configuration**.

```bash
python -m malleserve.tools.decode_predictor_reconcile \\
  --captured outputs/malleserve/decode_same_batch.jsonl \\
  --polaris-summary data/7796796.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/summary.csv \\
  --output-prefix decode_reconcile
```

Outputs:
- `decode_reconcile_captured.csv`: each original vs reconstructed
  batch's model, attention and MLP component predictions and deltas.
- `decode_reconcile_grid.csv`: full 15-point synthetic predictor grid
  using the *fixed-4* predictor rather than fixed-8.

**Interpretation:** The previous capture does not include original
per-request prompt lengths, request IDs, CUDA graph / shape metadata, or
all scheduler state. Reconstruction assigns each prompt the current
context length. Matching is useful evidence, but does not establish
that every input is reproduced. A **nonmatch is not automatically a
predictor-calibration failure**; inspect which per-layer/operator
components differ. Compare `frontier_model_ms` to Polaris full-engine
wall time only with that scope distinction. No calibration changes.
The predictor's training artifacts must be locally present.

The diagnostic constructs a Simulator but does not run requests, and
normal MalleServe preparation may create a named output directory.
