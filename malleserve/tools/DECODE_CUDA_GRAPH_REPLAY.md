# CUDA graph metadata in decode reconciliation

The legacy v2 same-batch capture omitted CUDA-graph runtime metadata.
That made the standalone reconstructed batch select eager/CUDA_EVENT even
though the real scheduler had selected KERNEL_ONLY. We now capture
`batch.decode_cuda_graph_metadata` and restore it onto reconstructed
batches before invoking the predictor.

A fresh live diagnostic run is required to get the enriched capture:

```bash
FRONTIER_DECODE_REPLAY_PATH=outputs/malleserve/decode_same_batch_v4.jsonl \\
FRONTIER_DECODE_REPLAY_LIMIT=30 \\
python -m malleserve.debug_run resources=fixed4 policy=fixed \\
  workload=mooncake_r10 experiment.name=decode_metadata_capture_v4
```

Then:

```bash
python -m malleserve.tools.decode_predictor_reconcile \\
  --captured outputs/malleserve/decode_same_batch_v4.jsonl \\
  --polaris-summary data/7796796.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/summary.csv \\
  --output-prefix decode_kernel_only_v4
```

The synthetic 15-point grid assumes a FULL CUDA-graph capture for each
homogeneous decode step; this is an explicit **assumption**, not a
reconstruction of real graph padding, shape selection or capture misses.
The replayed real scheduler batches instead use actual serialized
metadata, and the family-selection guard remains active. No calibration
constants are modified.
