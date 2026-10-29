# Attention lookup provenance

Set FRONTIER_ATTENTION_TRACE_PATH as in the earlier diagnostic.
The JSONL now includes `lookup` showing:
- `prediction_source`: exact measured/finite table, runtime cache, or trained model fallback.
- `active_measurement_type` and `measurement_family`.
- `trained_model_class` / `trained_model_params`, when available.
- `training_file_paths`, where stored on the predictor.

This metadata is observational. It does **not** invoke an alternative
prediction path, since doing so could mutate caching and cause unintended
performance changes. Exact measured-row and finite-table lookups currently
share the same final lookup branch and cannot be separated without
tracking which mapping matched.

Run a new MalleServe experiment with
`FRONTIER_ATTENTION_TRACE_PATH=outputs/malleserve/decode_attention_live_source.jsonl`
and `FRONTIER_ATTENTION_TRACE_ORIGIN=live`.
Then run `decode_predictor_reconcile` with `--attention-trace` to produce
a reconstructed trace for comparison. Do not use this run for published
latency baselines.
