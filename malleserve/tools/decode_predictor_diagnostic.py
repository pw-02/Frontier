#!/usr/bin/env python3
"""Compare observed Frontier decode stages with the Polaris full-engine decode grid.

This is a diagnostic of Frontier's *actual predicted batches*, not a synthetic
predictor invocation. Only exact batch size and prompt-length matches are
labelled exact. Host overhead is not inferred from engine-wall minus GPU model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BATCHES = (1, 2, 4, 8, 16)
CONTEXTS = (128, 1024, 4096)


def read_polaris(path: Path) -> dict[tuple[int, int], float]:
    values = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = int(row["batch_size"]), int(row["prompt_len"])
            values[key] = float(row["median_wall_ms_per_decode_step"])
    missing = {(b, c) for b in BATCHES for c in CONTEXTS} - values.keys()
    if missing:
        raise ValueError(f"Polaris data missing grid points: {sorted(missing)}")
    return values


def stages(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cluster_type") != "MONOLITHIC":
                continue
            tokens = row.get("request_num_tokens") or []
            if not tokens or any(n != 1 for n in tokens):
                continue
            batch_size = len(tokens)
            if batch_size not in BATCHES:
                continue
            timing = row["execution_time"]
            model = float(timing["model_time_ms"])
            total = float(timing["total_time_ms"])
            components = timing.get("component_ledger_ms", {})
            # These stage times are predicted quantities, NOT measured GPU
            # wall-clock durations.
            overhead = sum(float(components.get(k, 0)) for k in (
                "schedule_time", "sampler_e2e_time",
                "prepare_inputs_e2e_time", "process_model_outputs_time",
                "forward_launch_time",
            ))
            prompts = row.get("request_num_prefill_tokens")
            # Prompt length is not current decode context. Expose the
            # distinction explicitly: no guessing from prompt lengths.
            yield {
                "batch_size": batch_size,
                "model_ms": model,
                "stage_total_ms": total,
                "modelled_host_components_ms": overhead,
                "prompt_lengths": prompts,
                "batch_id": row.get("batch_id"),
            }


def compare(ledger: Path, polaris: Path, output: Path) -> list[dict]:
    real = read_polaris(polaris)
    groups = defaultdict(list)
    for stage in stages(ledger):
        groups[stage["batch_size"]].append(stage)
    rows = []
    for batch in BATCHES:
        samples = groups[batch]
        def median(k):
            return round(statistics.median(s[k] for s in samples), 5) if samples else None
        for context in CONTEXTS:
            # No synthetic batch has been predicted for this cell; explicitly
            # leave the comparable value blank instead of inventing one.
            rows.append({
                "batch_size": batch,
                "context_tokens": context,
                "polaris_engine_wall_median_ms": real[(batch, context)],
                "frontier_exact_grid_model_ms": None,
                "frontier_observed_batch_model_median_ms": median("model_ms"),
                "frontier_observed_batch_total_median_ms": median("stage_total_ms"),
                "frontier_modelled_host_component_median_ms": median("modelled_host_components_ms"),
                "frontier_observed_decode_stages": len(samples),
                "comparison_status": "NOT_CONTEXT_MATCHED",
                "context_note": "Frontier ledger lacks per-request current decode context; observed batch medians pool contexts",
                "host_note": "Polaris full-engine wall includes host overhead; Frontier host components are modelled, not independently measured",
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-ledger", type=Path, required=True)
    parser.add_argument("--polaris-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("decode_diagnostic.csv"))
    args = parser.parse_args()
    rows = compare(args.frontier_ledger, args.polaris_summary, args.output)
    print("Batch  Frontier model (observed)  Frontier stage (observed)  Samples")
    for batch in BATCHES:
        r = next(row for row in rows if row["batch_size"] == batch)
        print(f'{batch:>5}  {str(r["frontier_observed_batch_model_median_ms"]):>25}  '
              f'{str(r["frontier_observed_batch_total_median_ms"]):>25}  '
              f'{r["frontier_observed_decode_stages"]:>7}')
    print(f"Grid diagnostic written to {args.output}")
    print("NOTE: Exact context-matched predictor values not available from existing ledgers.")


if __name__ == "__main__":
    main()
