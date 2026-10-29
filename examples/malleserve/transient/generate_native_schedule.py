#!/usr/bin/env python3
"""Generate a rotating pool of node-granular transient GPU availability.

The physical resource trace is external to the inference scheduler.
For this first controlled baseline, only ONE four-GPU candidate node is
available at a time. A and B alternate so replacements have different KV
state. Startup/readiness latency is modeled natively by Frontier, not here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def generate(duration: float, lifetime: float,
             nodes: tuple[str, ...] = ("transient-0", "transient-1")) -> list[dict]:
    if duration <= 0 or lifetime <= 0:
        raise ValueError("Duration and lifetime must be positive")
    if not all(math.isfinite(x) for x in (duration, lifetime)):
        raise ValueError("Duration and lifetime must be finite")
    if len(nodes) < 2 or len(set(nodes)) != len(nodes):
        raise ValueError("At least two distinct candidate nodes required")
    result = []
    t, i = 0.0, 0
    while t < duration:
        node = nodes[i % len(nodes)]
        result.append({"timestamp": round(t, 6),
                       "node_id": node, "action": "available"})
        next_t = min(t + lifetime, duration)
        result.append({"timestamp": round(next_t, 6),
                       "node_id": node, "action": "reclaim"})
        t, i = next_t, i + 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=330)
    parser.add_argument("--lifetime-seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = generate(args.duration_seconds, args.lifetime_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(x, separators=(",", ":")) + "\n" for x in events),
        encoding="utf-8",
    )
    print(f"Wrote {len(events)} lifecycle events to {args.output}")


if __name__ == "__main__":
    main()
