#!/usr/bin/env python3
"""Run fixed or transient native Frontier scenarios from reusable YAML presets.

The same calibrated Frontier engine and trace are used for every scenario.
YAML may specify 'extends' to inherit common settings and override sections.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def deep_merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_config(path: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in ancestors:
        raise ValueError("Circular scenario extends: " + str(path))
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Scenario must be a mapping: {path}")
    inherited = config.pop("extends", None)
    if inherited is None:
        return config
    if not isinstance(inherited, str):
        raise ValueError("extends must be a relative YAML filename")
    return deep_merge(read_config(path.parent / inherited, ancestors + (path,)), config)


def load_scenario(path: Path) -> dict:
    config = read_config(path)
    if set(config) != {"experiment", "workload", "resources", "metrics"}:
        raise ValueError("Required sections: experiment, workload, resources, metrics")
    experiment, workload, resources, metrics = (
        config[k] for k in ("experiment", "workload", "resources", "metrics")
    )
    if any(not isinstance(x, dict) for x in (experiment, workload, resources, metrics)):
        raise ValueError("All scenario sections must be mappings")
    name = experiment.get("name")
    if not isinstance(name, str) or not name or any(
        not (x.isalnum() or x in "-_") for x in name
    ):
        raise ValueError("experiment.name must contain only letters, numbers, - or _")
    if set(workload) != {"trace"} or not isinstance(workload["trace"], str):
        raise ValueError("workload.trace must name a CSV")
    if set(metrics) != {"slo_seconds"} or type(metrics["slo_seconds"]) not in (int, float) or (
        not math.isfinite(metrics["slo_seconds"]) or metrics["slo_seconds"] <= 0
    ):
        raise ValueError("metrics.slo_seconds must be positive")
    policy = experiment.get("policy")
    mode = resources.get("mode")
    if policy not in ("fixed", "naive") or mode not in ("fixed", "transient") or (
        (mode == "fixed") != (policy == "fixed")
    ):
        raise ValueError("Use policy: fixed with mode: fixed, or policy: naive with mode: transient")
    if mode == "fixed":
        if set(resources) != {"mode", "gpus"} or type(resources["gpus"]) is not int or resources["gpus"] <= 0:
            raise ValueError("Fixed resources require mode: fixed and positive integer gpus")
    else:
        expected = {"mode", "persistent_gpus", "transient_gpus_per_node", "candidate_nodes",
                    "node_lifetime_seconds", "startup_seconds", "duration_seconds"}
        if set(resources) != expected:
            raise ValueError(f"Transient resources require keys: {sorted(expected)}")
        for key in ("persistent_gpus", "transient_gpus_per_node", "candidate_nodes"):
            if type(resources[key]) is not int or resources[key] <= 0:
                raise ValueError(f"resources.{key} must be a positive integer")
        if resources["candidate_nodes"] != 2:
            raise ValueError("Current rotation generator supports exactly two candidate nodes")
        for key in ("node_lifetime_seconds", "startup_seconds", "duration_seconds"):
            val = resources[key]
            if type(val) not in (int, float) or not math.isfinite(val) or (
                val < 0 if key == "startup_seconds" else val <= 0
            ):
                raise ValueError(f"Invalid resources.{key}")
    return config


def run_scenario(config: dict, dry_run: bool = False) -> dict:
    name = config["experiment"]["name"]
    resources = config["resources"]
    trace = Path(config["workload"]["trace"])
    if not trace.is_absolute():
        trace = ROOT / trace
    manifest = trace.with_suffix(".manifest.json")
    schedule = ROOT / "outputs/examples/malleserve/scenario_schedules" / f"{name}.jsonl"
    output = ROOT / "outputs/examples/malleserve" / name
    if not dry_run and (not trace.is_file() or not manifest.is_file()):
        raise FileNotFoundError(f"Missing trace and adjacent manifest: {trace}")
    env = dict(os.environ, RESOURCE_MODE=resources["mode"],
               SLO_SECONDS=str(config["metrics"]["slo_seconds"]),
               METRICS_OUTPUT_DIR=str(output))
    generate = None
    if resources["mode"] == "transient":
        generate = ["python", str(HERE / "generate_native_schedule.py"),
                    "--duration-seconds", str(resources["duration_seconds"]),
                    "--lifetime-seconds", str(resources["node_lifetime_seconds"]),
                    "--output", str(schedule)]
        env.update(PERSISTENT_REPLICAS=str(resources["persistent_gpus"]),
                   GPUS_PER_TRANSIENT_NODE=str(resources["transient_gpus_per_node"]),
                   TRANSIENT_CANDIDATE_NODES=str(resources["candidate_nodes"]),
                   TRANSIENT_STARTUP_SECONDS=str(resources["startup_seconds"]))
    else:
        env["FIXED_REPLICAS"] = str(resources["gpus"])
    command = ["bash", str(HERE / "run_native_transient.sh"),
               str(trace), str(schedule), name]
    plan = {"scenario": name, "mode": resources["mode"], "trace": str(trace),
            "metrics_output": str(output), "schedule_command": generate,
            "run_command": command, "resource_environment": {
                key: env[key] for key in ("RESOURCE_MODE", "FIXED_REPLICAS",
                  "PERSISTENT_REPLICAS", "GPUS_PER_TRANSIENT_NODE",
                  "TRANSIENT_CANDIDATE_NODES", "TRANSIENT_STARTUP_SECONDS")
                if key in env}}
    print(json.dumps(plan, indent=2))
    if not dry_run:
        if generate:
            schedule.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(generate, cwd=ROOT, check=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    return plan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run_scenario(load_scenario(args.config), args.dry_run)


if __name__ == "__main__":
    main()
