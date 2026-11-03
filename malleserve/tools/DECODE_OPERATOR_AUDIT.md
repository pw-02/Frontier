# Decode discrepancy: operator audit without new Polaris jobs

This diagnoses why Frontier's Qwen3-8B/A100 decode stage predictions are
larger than Polaris GPU-worker kernel-active measurements. It does **not**
assume the timing scopes are identical, and does not change predictor values.

## 1. In Frontier: refresh exact synthetic grid and attention lookup provenance

From the Frontier root on `malleserve-development`:

```bash
python -m malleserve.tools.decode_predictor_reconcile \
  --captured outputs/malleserve/decode_same_batch_v4.jsonl \
  --polaris-summary data/7796796.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/summary.csv \
  --attention-trace outputs/malleserve/decode_operator_attention_trace.jsonl \
  --output-prefix outputs/malleserve/decode_operator
```

If your captured JSONL or Polaris wall-time `summary.csv` is elsewhere,
supply the actual paths from the original decode experiment.

This produces `decode_operator_grid.csv`, `decode_operator_captured.csv`,
and a new attention lookup trace. The live-captured reconstruction must still
match the original predictor before interpreting synthetic operator values.
Attention trace must use `measurement_family=kernel_only`; do not use old
CUDA_EVENT/eager traces.

## 2. Compare against Polaris GPU profiling archive

Copy the existing Polaris job ZIP to this machine (the archive may have been
downloaded from Polaris as `7810629.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.zip`).
Then:

```bash
python -m malleserve.tools.decode_operator_audit \
  --frontier-grid outputs/malleserve/decode_operator_grid.csv \
  --polaris /path/to/7810629.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.zip \
  --attention-trace outputs/malleserve/decode_operator_attention_trace.jsonl \
  --output-prefix outputs/malleserve/decode_operator_audit
```

The `--polaris` option also accepts an extracted job folder containing
`grid/summary.csv` and `grid/kernel_analysis_{operators,kernels}.csv`.

Outputs:
- `decode_operator_audit_comparison.csv`: matched 9-point GPU-active/model
  comparison with **explicitly different** accounting boundaries
- `decode_operator_audit_top_kernels.csv`: top 12 Polaris CUDA kernels for
  every batch/context point, including their actual mangled kernel names
- `decode_operator_audit_report.json`: gap summaries, CUDA graph
  metadata, attention lookup family, provenance and calibration scale

## 3. How to use the results

Inspect the comparison at context=1024, batch=4 and context=4096,
batch=16. Review actual GEMM kernel names and Frontier attention suboperators,
not merely aggregate `frontier_attention_ms` versus
`polaris_attention_named_ms`. Frontier's attention time includes attention
pre/post projections and other operations, while Polaris GEMM-named kernels
include work outside MLP.

Check for (a) unexpected attention measurement family or lookup source,
(b) incorrect layer counts/double counting in the operator aggregation,
(c) attention backend or CUDA graph differences, and (d) deviations
that change with context and batch size.

If necessary, follow up with an isolated GPU model-runner forward timing
probe or operation-range instrumented profiler on Polaris. The available
paired generation tracer does not assign GEMM kernels to QKV vs MLP
with certainty. No correction factor should be deployed based only
on the cross-scope CSV.
