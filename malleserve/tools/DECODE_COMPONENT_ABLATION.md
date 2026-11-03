# Fixed4 decoder: operator accounting and reversible batching ablation

This is a **predictor-only** probe. It does not run another 330 s simulation,
does not alter the runtime scheduler, does not modify predictor coefficients
or profiling files, and does not provide a scope-matched GPU-forward benchmark.

Run from Frontier root on branch `malleserve-development`:

```bash
python -m malleserve.tools.decode_component_ablation \
  --frontier-grid outputs/malleserve/decode_operator_grid.csv \
  --output-prefix outputs/malleserve/decode_component_ablation
```

If the previously produced grid was moved, omit `--frontier-grid`; the
comparison to earlier results will not be verified.

Outputs:

- `decode_component_ablation_operators.csv`: for each batch 1/2/4/8/16
  and context 128/1024/4096, predicted **one physical layer** and
  **stage aggregate** times for QKV/attention pre-proj, output post-proj,
  FlashAttention decode, RoPE, KV save, both layer norms, MLP up/down/
  activation and residual add. Includes model aggregate, model-minus-operator
  residual, and no-batching-overhead counterfactual.
- `decode_component_ablation_report.json`: layer count, lookup table
  presence/types/sizes, configured batching overhead and consistency summary.

The operator columns provide an accounting view, not an estimate of how much
a run's P95 would change if the operation were eliminated. Stage aggregate
is computed by summing physical-layer costs, with an explicit residual for
operations not in the selected set. It checks per-layer sums against
StageExecutionTime aggregation and fails if a mismatch is found.

The ablation uses a **temporary in-memory** value of zero for
`_attention_decode_batching_overhead_fraction` and restores its initial
value in a finally block. The expected delta is verified analytically
against the predicted attention decode component. This tests the contribution
of the adjustment; not the physical validity of either configuration.

For context, Polaris's prior `grid/kernel_analysis_operators.csv` groups CUDA
kernel names heuristically; its GEMM category includes both QKV/output
projections and MLP. The two operators categories are not scope-matched.
The original GPU-worker profiler measures union of CUDA kernel intervals
in 1-token vs 129-token paired traces, which also averages growing context.

Recommended upload:
```bash
zip -j decode_component_ablation_results.zip \
  outputs/malleserve/decode_component_ablation_operators.csv \
  outputs/malleserve/decode_component_ablation_report.json
```

Next decision: if predictor projections/MLP appear to drive the fixed
~6.5 ms floor, obtain operation-labeled GPU timing using NVTX ranges or
isolated model-forward timing before changing calibration.
