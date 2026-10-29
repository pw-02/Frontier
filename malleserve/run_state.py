"""Idempotent workload preparation and completed-run detection for Hydra experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUCCESS = "run_complete.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def ensure_trace(workload: dict, block_size: int) -> tuple[Path, Path]:
    """Create missing CSV/manifest as a pair; never silently replace existing traces."""
    from malleserve.tools.convert_mooncake import convert
    import csv

    trace = resolve(workload["trace"])
    manifest = trace.with_suffix(".manifest.json")
    if trace.is_file() and manifest.is_file():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if int(info["rate_per_second"]) != int(workload["rate"]):
            raise ValueError(f"Trace rate does not match workload config: {trace}")
        if int(info["frontier_block_size"]) != block_size:
            raise ValueError(f"Trace block size mismatch: {trace}")
        return trace, manifest
    if trace.exists() != manifest.exists():
        raise FileExistsError(f"Incomplete trace pair (CSV/manifest): {trace}. Inspect before regenerating.")
    dataset_path = resolve(workload["dataset"])
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Missing prepared Mooncake dataset {dataset_path}. "
            "Place it there or set workload.dataset=...; existing trace+manifest also work."
        )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rate = int(workload["rate"])
    rows, info = convert(dataset, rate, int(workload["warmup_seconds"]),
                         int(workload["measurement_seconds"]),
                         int(workload["max_output_tokens"]), block_size)
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info["trace_csv"] = str(trace.resolve())
    info["prepared_dataset"] = str(dataset_path.resolve())
    manifest.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"[MalleServe] Generated {len(rows)} requests: {trace}", flush=True)
    return trace, manifest


def fingerprint(config: dict, trace: Path, manifest: Path) -> str:
    # Content-addressed identity detects changed experiment parameters and traces.
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(payload)
    for path in (trace, manifest):
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def is_complete(output: Path, identity: str) -> bool:
    marker = output / SUCCESS
    summary = output / "malleserve_summary.json"
    events = output / "metrics_ground_truth.jsonl"
    if not (marker.is_file() and summary.is_file() and events.is_file()):
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8"))["fingerprint"] == identity
    except (ValueError, KeyError, OSError):
        return False


def mark_complete(output: Path, identity: str) -> None:
    marker = output / SUCCESS
    tmp = output / (SUCCESS + ".tmp")
    tmp.write_text(json.dumps({"fingerprint": identity, "status": "complete"}, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(marker)
