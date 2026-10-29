# Polaris → Frontier fixed-capacity validation

Run native Frontier through the Hydra entry point, then compare its results with the **consolidated Polaris summary CSV**. The legacy ZIP and per-run folder readers have been removed.

```bash
python -m malleserve.debug_run resources=fixed4 policy=fixed workload=mooncake_r10
python -m malleserve.validate \\
  validation.polaris=data/malleserve/validation/polaris_summary.csv
```

Output: `outputs/malleserve/validation/fixed_capacity_comparison.csv`.

Use `python -m malleserve.debug_run -m resources=fixed4,fixed8 policy=fixed workload=mooncake_r10` for Hydra sweeps, ensuring each `experiment.name` is unique. Repeat source rows, e.g. 6 GPUs at 12 RPS, remain distinct. Only completed runs with at least 99% submitted requests, zero failures and explicitly zero dropped iterations count as valid-load evidence; blanks are unknown. Source client latency and simulated engine latency do not have identical boundaries. Preserve the same request traces and GPU model across comparisons.

To create request traces, use `python -m malleserve.tools.convert_mooncake`; it is a data preparation tool, not another experiment runner. Full Polaris logs are only needed for investigating deeper latency causes, not this initial sweep-level validation.
