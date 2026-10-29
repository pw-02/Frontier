# Decode timing discrepancy: controlled investigation

The original synthetic 47-53 ms estimates were caused by using EAGER
instead of KERNEL_ONLY. The corrected reconstruction matches 30/30
live batches. The remaining ~6.4-8.8 ms gap compares different scopes:
Frontier GPU model stage vs Polaris offline full-engine paired wall time.

The updated `decode_predictor_reconcile` now writes the stage operator
breakdown to `*_grid.csv` (attention, MLP up/down/activation,
forward launch, and model residual). No calibrations are applied.

```bash
python -m malleserve.tools.decode_predictor_reconcile \\
  --captured outputs/malleserve/decode_same_batch_v4.jsonl \\
  --polaris-summary data/7796796.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/summary.csv \\
  --output-prefix decode_investigate

python -m malleserve.tools.decode_gap_analysis \\
  --grid decode_investigate_grid.csv \\
  --polaris-metadata data/7796796.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/metadata.json \\
  --output-prefix decode_investigate
```

The metadata flag may be omitted if the local Polaris metadata file is
not available. Outputs are `decode_investigate_operators.csv` and
`decode_investigate_summary.json`.

The fitted affine relationship is *descriptive only*, not a calibrated
predictor. A physically meaningful operator correction requires matched
GPU-side profiling on Polaris. Do not alter default predictor coefficients
or report the full-engine wall fit as an operator model.
