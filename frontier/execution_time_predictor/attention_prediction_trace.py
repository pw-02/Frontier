"""Optional decode-attention input/scale tracing without extra predictions."""
from __future__ import annotations

import json
import os
from pathlib import Path


def record_decode_attention(*, path, predictor, batch, operation,
                            decode_batch_size, rounded_kv_cache_size,
                            raw_time_ms, scale, branch, lookup=None):
    requests = []
    for request, is_decoding, scheduled_context in zip(
            batch.requests, batch.request_is_decoding, batch.num_context_tokens):
        requests.append({
            "is_decoding": bool(is_decoding),
            "batch_scheduled_context_tokens": int(scheduled_context),
            "num_prefill_tokens": int(request.num_prefill_tokens),
            "num_processed_tokens": int(request.num_processed_tokens),
            "num_processed_decode_tokens": int(request.num_processed_decode_tokens),
            "num_emitted_decode_tokens": int(request.num_emitted_decode_tokens),
            "effective_attention_context_tokens": (
                int(predictor._get_decode_attention_context_tokens(request))
                if is_decoding else None
            ),
        })
    record = {
        "origin": os.environ.get("FRONTIER_ATTENTION_TRACE_ORIGIN", "live"),
        "batch_id": int(batch.id),
        "batch_size": len(batch.requests),
        "batch_num_tokens": list(batch.num_tokens),
        "num_prefill_tokens": int(batch.num_prefill_tokens),
        "operation": operation,
        "lookup": lookup,
        "feature_batch_size": int(decode_batch_size),
        "feature_rounded_kv_cache_size": int(rounded_kv_cache_size),
        "kv_cache_prediction_granularity": int(
            predictor._config.kv_cache_prediction_granularity
        ),
        "calibration_branch": branch,
        "raw_attention_decode_ms_per_layer": float(raw_time_ms),
        "calibration_scale": float(scale),
        "scaled_attention_decode_ms_per_layer": float(raw_time_ms * scale),
        "requests": requests,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
