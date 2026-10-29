# Polaris vs Frontier decode timing diagnostic

Reads Polaris `summary.csv` from the standalone decode benchmark and
Frontier's existing `frontier_stage_batch_ledger.jsonl`, grouping *observed*
decode-only stages by batch size. Outputs all **15 Polaris grid points**
(1/2/4/8/16 × 128/1024/4096) alongside observed Frontier model, total
stage and explicitly-modelled host-component medians.

**Critical limitation:** These Frontier stage medians do NOT match Polaris
context lengths. The current stage ledger does not store the current
per-request decode context. Columns containing context-matched Frontier
predictions remain empty and comparisons are marked NOT_CONTEXT_MATCHED.
Do not interpret the values as a calibrated head-to-head benchmark.
Polaris numbers are full-engine wall time, while Frontier numbers are
modelled execution times.

```bash
python -m malleserve.tools.decode_predictor_diagnostic \
  --frontier-ledger outputs/malleserve/runs/fixed4_r10/frontier_stage_batch_ledger.jsonl \
  --polaris-summary /path/to/polaris/summary.csv \
  --output outputs/malleserve/decode_diagnostic.csv
python -m unittest tests.unit.test_decode_predictor_diagnostic -v
```

A true exact synthetic predictor comparison requires constructing 15
scheduler-valid synthetic decode batches with explicit current-context
lengths, then invoking the actual stage predictor. This diagnostic
intentionally does not pretend that the existing ledger provides that.
