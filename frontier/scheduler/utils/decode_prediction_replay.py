"""Opt-in reproducibility probe for *actual* scheduler decode batches.

When FRONTIER_DECODE_REPLAY_PATH is set, replay the very same Batch through
the same predictor, without constructing or approximating its context.
Never enabled in production; diagnostic prediction can mutate predictor state.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

_seen: set[tuple[int, tuple[int, ...]]] = set()


def _timing(value):
    return {
        "model_ms": float(value.model_time_ms),
        "total_ms": float(value.total_time * 1000),
        "host_ms": float(value.total_time * 1000 - value.model_time_ms),
        "attention_ms": float(value.attention_time),
        "mlp_up_ms": float(value.mlp_layer_up_proj_execution_time),
        "mlp_down_ms": float(value.mlp_layer_down_proj_execution_time),
        "mlp_act_ms": float(value.mlp_layer_act_execution_time),
        "forward_launch_ms": float(value.forward_launch_time),
    }


def maybe_probe(batch, original, predictor, stage_id, cluster_type, num_layers, layer_id):
    path = os.environ.get("FRONTIER_DECODE_REPLAY_PATH")
    if not path or cluster_type.name != "MONOLITHIC":
        return
    tokens = list(batch.num_tokens)
    contexts = list(batch.num_context_tokens)
    decoding = list(batch.request_is_decoding)
    if not tokens or any(t != 1 for t in tokens) or not all(decoding):
        return
    key = (len(tokens), tuple(contexts))
    if key in _seen or len(_seen) >= int(os.environ.get("FRONTIER_DECODE_REPLAY_LIMIT", "20")):
        return
    _seen.add(key)
    replay = predictor.predict_stage_execution_time(
        batch, stage_id, cluster_type=cluster_type, num_layers=num_layers,
        layer_id=layer_id,
    )
    row = {
        "batch_id": batch.id,
        "batch_size": len(tokens),
        "context_tokens": contexts,
        "request_is_decoding": decoding,
        "num_tokens": tokens,
        "num_layers": num_layers,
        "layer_id": layer_id,
        "stage_id": stage_id,
        "original": _timing(original),
        "same_batch_replay": _timing(replay),
        "delta_model_ms": float(replay.model_time_ms - original.model_time_ms),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
