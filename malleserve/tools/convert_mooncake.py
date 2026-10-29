#!/usr/bin/env python3
"""Convert a prepared MalleServe Mooncake JSON workload to a Frontier trace CSV.

Matches fixed_capacity/replay_fixed_duration.js:
  * fixed-rate, open-loop arrivals, never the original Mooncake timestamps;
  * disjoint warmup/measurement dataset slices (including k6's +2 safety gap);
  * output lengths clipped to 1..max_output_tokens;
  * deterministic 512-token Mooncake block IDs expanded to 16-token cache IDs.
No Frontier internals are modified. This is a workload adapter, not calibration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def stable_id(*parts: object) -> int:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=False).encode()
    # Positive signed 63-bit integers survive Pandas/Frontier int conversion.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def frontier_block_ids(request: dict, source_block_size: int, frontier_block_size: int) -> list[int]:
    """Encode each full Frontier cache block with stable, reusable content identity.

    For block_size=16 and source_block_size=512, each source block becomes 32
    distinct sub-block hashes; identical source blocks get identical sub-block
    sequences. Partial terminal blocks are not counted as full/cacheable blocks.
    """
    if source_block_size % frontier_block_size:
        raise ValueError("source block size must be a multiple of Frontier block size")
    length = int(request["input_length"])
    source_ids = request["hash_ids"]
    if length <= 0 or len(source_ids) < math.ceil(length / source_block_size):
        raise ValueError("Missing source block hashes or invalid input length")
    count = length // frontier_block_size
    per_source = source_block_size // frontier_block_size
    result = [
        stable_id("mooncake-block", str(source_ids[i // per_source]), i % per_source)
        for i in range(count)
    ]
    # Frontier requires a nonempty metadata cell even for very short prompts.
    if not result:
        result = [stable_id("mooncake-block", str(source_ids[0]), 0)]
    return result


def convert(
    dataset: dict,
    rate: int,
    warmup_seconds: int,
    measurement_seconds: int,
    output_cap: int,
    frontier_block_size: int,
) -> tuple[list[dict], dict]:
    if rate <= 0 or warmup_seconds < 0 or measurement_seconds <= 0 or output_cap <= 0:
        raise ValueError("Invalid positive rate/durations/output cap")
    if dataset.get("schema_version") != 1 or not isinstance(dataset.get("requests"), list):
        raise ValueError("Expected prepared Mooncake JSON schema_version=1 with requests[]")
    source_block_size = int(dataset["block_size"])
    if frontier_block_size <= 0 or source_block_size % frontier_block_size:
        raise ValueError("Frontier block size must divide the source block size")

    requests = dataset["requests"]
    warmup_count = rate * warmup_seconds
    measurement_count = rate * measurement_seconds
    warmup_budget = warmup_count + 2   # Mirrors replay_fixed_duration.js
    required = warmup_budget + measurement_count + 2
    if len(requests) < required:
        raise ValueError(
            f"Not enough prepared requests: need {required}, have {len(requests)}. "
            "Regenerate Mooncake with more records; no wraparound is permitted."
        )

    rows: list[dict] = []
    max_total = 0
    for phase, first_index, count, start_at in (
        ("warmup", 0, warmup_count, 0),
        ("measured", warmup_budget, measurement_count, warmup_seconds),
    ):
        for ordinal in range(count):
            record_index = first_index + ordinal
            record = requests[record_index]
            prompt_tokens = int(record["input_length"])
            decode_tokens = max(1, min(int(record.get("output_length") or 1), output_cap))
            original_blocks = record["hash_ids"]
            block_hashes = frontier_block_ids(record, source_block_size, frontier_block_size)
            max_total = max(max_total, prompt_tokens + decode_tokens)
            rows.append({
                "arrived_at": f"{start_at + ordinal / rate:.9f}",
                "num_prefill_tokens": prompt_tokens,
                "num_decode_tokens": decode_tokens,
                # Heuristic for sticky routing: shared first 512-token prefix.
                # Only one replica should be used for initial calibration.
                "session_id": stable_id("mooncake-session-first-block", str(original_blocks[0])),
                "block_hash_ids": "|".join(str(value) for value in block_hashes),
                "phase": phase,
                "source_index": record_index,
                "source_trace_index": record.get("trace_index", ""),
                "original_request_id": record.get("request_id", ""),
            })

    metadata = {
        "schema_version": 1,
        "source_workload": dataset.get("workload_name", "mooncake"),
        "rate_per_second": rate,
        "warmup_seconds": warmup_seconds,
        "measurement_seconds": measurement_seconds,
        "warmup_count": warmup_count,
        "measurement_count": measurement_count,
        "total_requests": len(rows),
        "source_block_size": source_block_size,
        "frontier_block_size": frontier_block_size,
        "max_output_tokens": output_cap,
        "required_frontier_max_tokens": max_total,
        "dataset_index_gap_after_warmup": 2,
        "arrival_model": "constant_rate_not_original_trace_timestamps",
        "notes": "Prefix hashes describe possible equality, not observed GPU KV hits.",
    }
    return rows, metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True, help="Prepared Mooncake schema-v1 JSON")
    p.add_argument("--output", type=Path, default=Path("malleserve/tools/mooncake_r2.csv"))
    p.add_argument("--rate", type=int, default=2, help="Integer arrivals per second, as in k6")
    p.add_argument("--warmup-seconds", type=int, default=30)
    p.add_argument("--measurement-seconds", type=int, default=300)
    p.add_argument("--max-output-tokens", type=int, default=128)
    p.add_argument("--frontier-block-size", type=int, default=16)
    args = p.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows, metadata = convert(
        dataset, args.rate, args.warmup_seconds, args.measurement_seconds,
        args.max_output_tokens, args.frontier_block_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = args.output.with_suffix(".manifest.json")
    metadata["trace_csv"] = str(args.output.resolve())
    metadata["prepared_dataset"] = str(args.dataset.resolve())
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows):,} requests: {args.output}")
    print(f"  Warmup: {metadata['warmup_count']:,}; measured: {metadata['measurement_count']:,}")
    print(f"  Arrivals: {args.rate}/s, measured interval [{args.warmup_seconds}, "
          f"{args.warmup_seconds + args.measurement_seconds}) s")
    print(f"  Block identifiers: {metadata['source_block_size']} -> "
          f"{metadata['frontier_block_size']} tokens")
    print(f"  Minimum Frontier trace max_tokens: {metadata['required_frontier_max_tokens']}")
    print(f"  Manifest: {manifest_path}")
    print("  This preserves workload shape/prefix identity, not cache hit rates.")


if __name__ == "__main__":
    main()
