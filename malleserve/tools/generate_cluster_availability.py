"""Seeded, overlapping node-granular HPC availability *synthetic model*.

Independent candidate nodes can be live concurrently. Every candidate is
made available at least once, even if its first allocation begins late.
Times are simulated seconds; this is NOT a replay of measured HPC logs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def generate(
    duration: float, nodes: int, seed: int = 7,
    min_lifetime: float = 90, max_lifetime: float = 240,
    min_downtime: float = 20, max_downtime: float = 90,
    initial_spread: float = 90,
) -> list[dict]:
    if not (duration > 0 and isinstance(nodes, int) and nodes > 0
            and 0 < min_lifetime <= max_lifetime
            and 0 <= min_downtime <= max_downtime
            and 0 <= initial_spread < duration):
        raise ValueError("Invalid duration, node count, lifetime, downtime or spread")
    import math
    if not all(map(math.isfinite, (duration, min_lifetime, max_lifetime,
                                  min_downtime, max_downtime, initial_spread))):
        raise ValueError("All schedule parameters must be finite")
    rng = random.Random(seed)
    events = []
    for i in range(nodes):
        node = f"transient-{i:03d}"
        # Stratified initial activation avoids all nodes joining simultaneously.
        time = (i + rng.random()) / nodes * initial_spread
        while time < duration:
            events.append({"timestamp": round(time, 6),
                           "node_id": node, "action": "available"})
            departure = time + rng.uniform(min_lifetime, max_lifetime)
            if departure >= duration:
                # Allow node to continue into drain: no artificial terminal reclaim.
                break
            events.append({"timestamp": round(departure, 6),
                           "node_id": node, "action": "reclaim"})
            time = departure + rng.uniform(min_downtime, max_downtime)
    events.sort(key=lambda e: (e["timestamp"], 0 if e["action"] == "reclaim" else 1,
                               e["node_id"]))
    return events


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=330)
    p.add_argument("--nodes", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-lifetime", type=float, default=90)
    p.add_argument("--max-lifetime", type=float, default=240)
    p.add_argument("--min-downtime", type=float, default=20)
    p.add_argument("--max-downtime", type=float, default=90)
    p.add_argument("--initial-spread", type=float, default=90)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    events = generate(args.duration, args.nodes, args.seed,
                      args.min_lifetime, args.max_lifetime,
                      args.min_downtime, args.max_downtime, args.initial_spread)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    print(f"Wrote {len(events)} events across {args.nodes} candidate nodes: {args.output}")


if __name__ == "__main__":
    main()
