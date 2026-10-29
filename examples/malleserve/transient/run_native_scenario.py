#!/usr/bin/env python3
"""Run one native Frontier MalleServe scenario from a small YAML configuration.

Only exposes knobs supported by the existing native transient runner. This is
an experiment interface, not a replacement for Frontier's SimulationConfig.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def load_scenario(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Scenario YAML must contain a mapping")
    if set(config) != {"experiment", "workload", "resources", "metrics"}:
        raise ValueError("Required sections: experiment, workload, resources, metrics")
    experiment, workload, resources, metrics = (
        config[k] for k in ("experiment", "workload", "resources", "metrics")
    )
    if any(not isinstance(obj, dict) for obj in (experiment, workload, resources, metrics)):
        raise ValueError("Every section must be a YAML mapping")
    if set(experiment) != {"name", "policy"} or experiment["policy"] != "naive":
        raise ValueError("Only experiment.name and policy: naive are currently supported")
    name = experiment["name"]
    if not isinstance(name, str) or not name or not all(
        ch.isalnum() or ch in "-_" for ch in name
    ):
        raise ValueError("experiment.name must use only letters, digits, - and _")
    if set(workload) != {"trace"} or not isinstance(workload["trace"], str):
        raise ValueError("workload.trace must be a CSV path")
    required = {"persistent_gpus", "transient_gpus_per_node",
                "candidate_nodes", "node_lifetime_seconds", "startup_seconds",
                "duration_seconds"}
    if set(resources) != required:
        raise ValueError(f"resources keys must be {sorted(required)}")
    for key in ("persistent_gpus", "transient_gpus_per_node", "candidate_nodes"):
        if type(resources[key]) is not int or resources[key] <= 0:
            raise ValueError(f"resources.{key} must be a positive integer")
    if resources["candidate_nodes"] < 2:
        raise ValueError("candidate_nodes must be >= 2 for the rotation generator")
    for key in ("node_lifetime_seconds", "startup_seconds", "duration_seconds"):
        value = resources[key]
        if type(value) not in (float, int) or not math.isfinite(value) or (
            value < 0 if key == "startup_seconds" else value <= 0
        ):
            raise ValueError(f"resources.{key} must be finite and nonnegative/positive")
    if set(metrics) != {"slo_seconds"} or type(metrics["slo_seconds"]) not in (int, float) or (
        not math.isfinite(metrics["slo_seconds"]) or metrics["slo_seconds"] <= 0
    ):
        raise ValueError("metrics.slo_seconds must be positive and finite")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_scenario(args.config)
    experiment = config["experiment"]
    r = config["resources"]
    name = experiment["name"]
    trace = Path(config["workload"]["trace"])
    if not trace.is_absolute():
        trace = ROOT / trace
    schedule = ROOT / "outputs/examples/malleserve/scenario_schedules" / f"{name}.jsonl"
    output = ROOT / "outputs/examples/malleserve" / name
    if not args.dry_run:
        if not trace.is_file() or not trace.with_suffix(".manifest.json").is_file():
            raise FileNotFoundError(f"Missing trace or adjacent manifest for {trace}")
        schedule.parent.mkdir(parents=True, exist_ok=True)
    generate = [
        "python", str(HERE / "generate_native_schedule.py"),
        "--duration-seconds", str(r["duration_seconds"]),
        "--lifetime-seconds", str(r["node_lifetime_seconds"]),
        "--output", str(schedule),
    ]
    # The existing generator rotates two named candidate nodes.
    if r["candidate_nodes"] != 2:
        raise ValueError("Current generator requires candidate_nodes: 2")
    run = ["bash", str(HERE / "run_native_transient.sh"), str(trace), str(schedule), name]
    env = dict(os.environ, PERSISTENT_REPLICAS=str(r["persistent_gpus"]),
               GPUS_PER_TRANSIENT_NODE=str(r["transient_gpus_per_node"]),
               TRANSIENT_CANDIDATE_NODES=str(r["candidate_nodes"]),
               TRANSIENT_STARTUP_SECONDS=str(r["startup_seconds"]),
               SLO_SECONDS=str(config["metrics"]["slo_seconds"]),
               METRICS_OUTPUT_DIR=str(output))
    print("Generate schedule:", " ".join(generate))
    print("Run Frontier:", " ".join(run))
    print("Configuration:", json.dumps({"replicas": r["persistent_gpus"] +
        r["candidate_nodes"] * r["transient_gpus_per_node"],
        "startup_seconds": r["startup_seconds"], "output": str(output)}, indent=2))
    if args.dry_run:
        return
    subprocess.run(generate, cwd=ROOT, check=True)
    subprocess.run(run, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
