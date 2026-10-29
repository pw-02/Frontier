"""MalleServe native Frontier in-process runner, suitable for IDE breakpoints.

Edit CONFIG and click 'Debug Python File', or:
    python -m malleserve.debug_run
This calls Frontier's normal main() in THIS Python process, not a subprocess.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from malleserve.scenario import load_scenario, prepare

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/malleserve/transient/scenarios/naive.yaml"

# Optional override for quick experiments without editing a YAML file:
OVERRIDES = {
    # "experiment": {"name": "debug_naive"},
    # "resources": {"startup_seconds": 0},
}


def main(config_path: Path = CONFIG, overrides: dict | None = None) -> None:
    from malleserve.scenario import merge

    scenario = merge(load_scenario(config_path), overrides or OVERRIDES)
    args, output, manifest = prepare(scenario)
    print(f"Debugging scenario {scenario['experiment']['name']}, output={output}", flush=True)
    # This is equivalent to python -m frontier.main [args], except the debugger
    # can step directly into the simulator, lifecycle and scheduler code.
    previous_argv = sys.argv
    try:
        sys.argv = ["frontier.main", *args]
        from frontier.main import main as frontier_main
        frontier_main()
    finally:
        sys.argv = previous_argv

    from examples.malleserve.analyze_mooncake import (
        load_events, resolve_metrics_dir, summary,
        read_request_csv_metrics,
    )
    metrics_dir = resolve_metrics_dir(output)
    data = summary(load_events(metrics_dir / "metrics_ground_truth.jsonl"),
                   json.loads(manifest.read_text(encoding="utf-8")),
                   float(scenario["metrics"]["slo_seconds"]))
    data.update(read_request_csv_metrics(metrics_dir / "request_metrics.csv"))
    destination = metrics_dir / "malleserve_summary.json"
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {destination}")


if __name__ == "__main__":
    main()
