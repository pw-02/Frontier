# Exact synthetic decode predictor grid

This script constructs **15 real Frontier Batch objects** with batch sizes
1, 2, 4, 8, 16 and current KV decode-context lengths 128, 1024, 4096.
Every request is in decode phase and schedules exactly one token.

It invokes Frontier's actual fitted predictor initialized with MalleServe's
fixed-8 Qwen3-8B/A100 configuration; no previously observed batch medians
are used. Reports model time, separately modelled host overhead, total
predicted stage and matching Polaris full-engine measured wall time.

```bash
python -m malleserve.tools.decode_predictor_matched \
  --polaris-summary data/7796796.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/summary.csv \
  --output decode_predictor_matched.csv
```

**Limitations:** Polaris measures full-engine paired-generation incremental
wall time (CPU+GPU, potentially CUDA graph, host launch and sampling).
Frontier model time is predicted. The total-stage estimate is **not**
identical to engine wall time, even if context and batch size match.
No coefficients are changed. The script builds a full Frontier Simulator
without calling run(), to load the same trained model. It may create the
normal MalleServe output directory as a side effect of scenario preparation.
Inspect model/configuration identity before interpreting calibration claims.
