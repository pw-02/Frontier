# Experimental decode-only operator calibration (MalleServe)

These controls are **sensitivity knobs, not a new calibrated hardware model**.
They are specifically for the Qwen3-8B/A100 mismatch investigation.
The default `calibration=baseline` retains the original Frontier timings,
prediction tables, and prior result-folder naming.

## Code changes

- `frontier/config/execution_time_predictor_config.py` declares three
  optional positive scales (default `None`).
- `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`
  applies them **only in pure-decode batches** (no prefill tokens and at least
  one decode token) after the normal profiling lookup.
- `malleserve/scenario.py` passes requested scales as native Frontier
  predictor CLI options. No runtime environment variable or altered CSV is
  needed.
- Hydra `calibration` group has `baseline`, `rope_half`,
  `norm_half`, and `combined_half` modes.

The three fields are:
```
decode_phase_attn_rope_calibration_scale
decode_phase_input_layernorm_calibration_scale
decode_phase_post_attention_layernorm_calibration_scale
```
Values: `null` disables the experimental override, `0.5` halves the
corresponding **decode-only** prediction, and `1.0` leaves it unchanged.
All non-null values must be finite and > 0.

The half-scaled configs are *not fitted* to the current Polaris trace,
which measures eager kernels instead of the CUDA-graph model-forward scope.

## Fast check before any full simulation

```bash
cd ~/projects/Frontier
git switch malleserve-development
git pull --ff-only origin malleserve-development
python -m malleserve.tools.decode_calibration_sensitivity
```

This writes
`outputs/malleserve/decode_calibration_sensitivity.csv`.
It checks analytical accounting for 15 contexts/batch combinations and
four calibration variants, without running workload events.

## Full simulations (use unique output names)

Use the existing native `debug_run` runner and same workload, schedule and
randomization configuration. The example is fixed4 with the current
`workload=mooncake_r10` default:

```bash
python -m malleserve.debug_run resources=fixed4 policy=fixed \
  calibration=baseline experiment.name=fixed4_r10
python -m malleserve.debug_run resources=fixed4 policy=fixed \
  calibration=rope_half experiment.name=fixed4_r10_rope_half
python -m malleserve.debug_run resources=fixed4 policy=fixed \
  calibration=norm_half experiment.name=fixed4_r10_norm_half
python -m malleserve.debug_run resources=fixed4 policy=fixed \
  calibration=combined_half experiment.name=fixed4_r10_combined_half
```

If the original `fixed4_r10` already exists and is complete, the runner
may skip it automatically. **Do not reuse that output name for a modified
calibration**: `debug_run` deliberately prevents overwriting mismatched
results.

For transient experiments, use `resources=naive policy=naive`, different
experiment names, and the **same calibration choice** as the fixed run.
Do not compare a calibrated transient policy to an uncalibrated fixed
baseline as evidence of a scheduling improvement.

## Interpretation

A 0.5 operator scale probes model sensitivity. It cannot establish that
Polaris's actual CUDA-graph forward executes that operator in half the time.
Qwen3 also has Q/K head normalization, not separately represented by these
three generic layer norm scales. Before promoting any coefficients, obtain
matched GPU forward intervals / operator mapping and verify decode, TTFT,
throughput and latency distributions against Polaris.
