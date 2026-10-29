#!/usr/bin/env python3
"""Exploratory event-driven transient-GPU capacity simulator.

This is an independently testable MACRO model driven by low-load Frontier
request timings. It is NOT Frontier's vLLM batch scheduler: workers use
processor sharing as a deliberately documented approximation. Do not use
its numerical p95 or sustainable capacity in a paper until it is validated
against native Frontier dynamic-replica simulation and Polaris.

The lifecycle and recovery logic (activation delay, rotating candidates,
interruptions/retries, ideal zero-cost continuation) is exact *within this
model*, making it useful for testing the experiment design and hypotheses.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EPS = 1e-9


@dataclass
class Request:
    id: int
    arrived: float
    input_tokens: int
    output_tokens: int
    source_index: int
    total_work: float
    remaining: float
    completed: float | None = None
    attempts: int = 0
    interruptions: int = 0
    wasted_work: float = 0.0


@dataclass
class Worker:
    id: str
    node: str
    persistent: bool = False
    available: bool = False
    ready: bool = False
    generation: int = 0
    completion_generation: int = 0
    jobs: list[Request] = field(default_factory=list)
    last_update: float = 0.0
    ready_seconds_measured: float = 0.0


@dataclass(frozen=True)
class NodeEvent:
    time: float
    node: str
    action: str


class ProxyWorkModel:
    """Deterministic nearest-neighbor interpolation of Frontier low-load data."""

    def __init__(self, data: dict):
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported profile schema")
        samples = data.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("Empty proxy service profile")
        self.samples = []
        self.by_index = {}
        for row in samples:
            index = int(row["source_index"])
            length = int(row["input_tokens"])
            out = int(row["output_tokens"])
            work = float(row["proxy_service_s"])
            if length <= 0 or out <= 0 or not math.isfinite(work) or work <= 0:
                raise ValueError(f"Invalid profile sample: {row}")
            sample = (length, out, work)
            self.samples.append(sample)
            if index in self.by_index:
                raise ValueError(f"Duplicate source_index {index}")
            self.by_index[index] = sample
        self.direct_matches = 0
        self.estimated_matches = 0

    def estimate(self, source_index: int, prompt: int, output: int) -> float:
        matched = self.by_index.get(source_index)
        if matched and matched[0] == prompt and matched[1] == output:
            self.direct_matches += 1
            return matched[2]
        self.estimated_matches += 1
        # Neighbor distances weight output length heavily because decode is
        # the main observed latency contribution on Qwen3-8B.
        neighbors = sorted(
            self.samples,
            key=lambda s: (
                abs(s[1] - output) / 32
                + abs(math.log1p(s[0]) - math.log1p(prompt)) * 0.4,
                abs(s[1] - output),
                abs(s[0] - prompt),
            ),
        )[: min(9, len(self.samples))]
        return statistics.median(s[2] for s in neighbors)


def read_trace(path: Path, work_model: ProxyWorkModel) -> list[Request]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            t = float(row["arrived_at"])
            prompt = int(row["num_prefill_tokens"])
            output = int(row["num_decode_tokens"])
            index = int(row["source_index"])
            if not math.isfinite(t) or t < 0 or prompt <= 0 or output <= 0:
                raise ValueError(f"Invalid trace at row {i}")
            work = work_model.estimate(index, prompt, output)
            rows.append(Request(
                id=i, arrived=t, input_tokens=prompt, output_tokens=output,
                source_index=index, total_work=work, remaining=work,
            ))
    if not rows:
        raise ValueError("Empty input trace")
    if any(b.arrived < a.arrived for a, b in zip(rows, rows[1:])):
        raise ValueError("Arrival times must be nondecreasing")
    return rows


def rotating_schedule(duration: float, lifetime: float) -> list[NodeEvent]:
    """Rotate between two 4-GPU candidates; never exceed one available node."""
    if lifetime <= 0 or not math.isfinite(lifetime):
        raise ValueError("lifetime must be finite and positive")
    events = []
    t = 0.0
    interval = 0
    while t < duration:
        current = f"transient-{interval % 2}"
        events.append(NodeEvent(t, current, "available"))
        stop = min(t + lifetime, duration)
        events.append(NodeEvent(stop, current, "reclaim"))
        t += lifetime
        interval += 1
    return events


def read_schedule(path: Path) -> list[NodeEvent]:
    events = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if set(item) != {"timestamp", "node_id", "action"}:
                raise ValueError(f"Schedule line {index}: expected timestamp,node_id,action")
            time = float(item["timestamp"])
            node = str(item["node_id"])
            action = item["action"]
            if action not in {"available", "reclaim"} or not node or not math.isfinite(time) or time < 0:
                raise ValueError(f"Invalid schedule line {index}")
            events.append(NodeEvent(time, node, action))
    return sorted(events, key=lambda e: (e.time, 0 if e.action == "reclaim" else 1))


def validate_schedule(events: list[NodeEvent]) -> list[str]:
    states: dict[str, bool] = {}
    for e in sorted(events, key=lambda e: (e.time, 0 if e.action == "reclaim" else 1)):
        present = states.get(e.node, False)
        if e.action == "available":
            if present:
                raise ValueError(f"Node {e.node} made available twice at {e.time}")
            states[e.node] = True
        else:
            if not present:
                raise ValueError(f"Node {e.node} reclaimed before availability at {e.time}")
            states[e.node] = False
    return sorted(states)


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] * (hi - k) + s[hi] * (k - lo)


class Experiment:
    """Event-driven processor-sharing approximation, with exact lifecycle."""

    def __init__(
        self,
        requests: list[Request],
        mode: str,
        events: list[NodeEvent],
        startup_seconds: float,
        warmup_seconds: float,
        measurement_seconds: float,
        slo_seconds: float,
        drain_seconds: float,
        persistent_workers: int = 4,
        stable_workers: int = 8,
        gpus_per_transient_node: int = 4,
        cold_start_work_seconds: float = 0,
    ):
        if mode not in {"fixed_persistent", "fixed_stable", "naive", "ideal"}:
            raise ValueError("Unsupported experiment mode")
        for val in (startup_seconds, warmup_seconds, measurement_seconds,
                    slo_seconds, drain_seconds, cold_start_work_seconds):
            if not math.isfinite(val) or val < 0:
                raise ValueError("All durations must be finite and nonnegative")
        if slo_seconds <= 0 or measurement_seconds <= 0:
            raise ValueError("Measurement and SLO must be positive")
        if min(persistent_workers, stable_workers, gpus_per_transient_node) <= 0:
            raise ValueError("GPU counts must be positive")
        self.mode = mode
        self.requests = requests
        self.events = sorted(events, key=lambda e: (e.time, 0 if e.action == "reclaim" else 1))
        nodes = validate_schedule(events) if mode in {"naive", "ideal"} else []
        self.startup_seconds = 0.0 if mode == "ideal" else startup_seconds
        self.warmup_seconds = warmup_seconds
        self.measurement_seconds = measurement_seconds
        self.end = warmup_seconds + measurement_seconds
        self.stop = self.end + drain_seconds
        self.slo = slo_seconds
        self.cold_start_work_seconds = cold_start_work_seconds if mode == "naive" else 0.0
        self.counter = itertools.count()
        self.heap: list[tuple[float, int, int, str, Any]] = []
        self.workers: dict[str, Worker] = {}
        self.pending: deque[Request] = deque()
        self.round_robin = 0
        self.event_log: list[dict] = []
        self.reclaims = 0
        self.startups = 0
        self.interruptions = 0
        self.lost_work = 0.0
        self.migrations = 0
        self.clock = 0.0
        self._cold_workers: set[str] = set()

        fixed = stable_workers if mode == "fixed_stable" else persistent_workers
        for i in range(fixed):
            ident = f"persistent-{i}"
            self.workers[ident] = Worker(
                id=ident, node="persistent", persistent=True,
                ready=True, available=True,
            )
        for node in nodes:
            for j in range(gpus_per_transient_node):
                ident = f"{node}-gpu{j}"
                if ident in self.workers:
                    raise ValueError("Node IDs overlap with persistent workers")
                self.workers[ident] = Worker(id=ident, node=node)

        for request in requests:
            self.push(request.arrived, 4, "arrival", request)
        if mode in {"naive", "ideal"}:
            for event in self.events:
                kind = "reclaim" if event.action == "reclaim" else "available"
                self.push(event.time, 0 if kind == "reclaim" else 1, kind, event.node)

    def push(self, t: float, priority: int, kind: str, payload: Any) -> None:
        heapq.heappush(self.heap, (t, priority, next(self.counter), kind, payload))

    def _advance(self, now: float) -> None:
        if now + EPS < self.clock:
            raise RuntimeError("Simulation time ran backwards")
        for worker in self.workers.values():
            last = worker.last_update
            if worker.ready:
                overlap = max(0.0, min(now, self.end) - max(last, self.warmup_seconds))
                worker.ready_seconds_measured += overlap
                if worker.jobs:
                    per_request = max(0.0, now - last) / len(worker.jobs)
                    for request in worker.jobs:
                        request.remaining = max(0.0, request.remaining - per_request)
            worker.last_update = now
        self.clock = now

    def _ready_workers(self) -> list[Worker]:
        return [w for w in self.workers.values() if w.ready]

    def _assign(self, request: Request) -> None:
        ready = self._ready_workers()
        if not ready:
            self.pending.append(request)
            return
        worker = ready[self.round_robin % len(ready)]
        self.round_robin += 1
        request.attempts += 1
        if worker.id in self._cold_workers:
            request.remaining += self.cold_start_work_seconds
            self._cold_workers.remove(worker.id)
        worker.jobs.append(request)

    def _flush_pending(self) -> None:
        if not self._ready_workers():
            return
        while self.pending:
            self._assign(self.pending.popleft())

    def _reschedule(self, now: float) -> None:
        for worker in self.workers.values():
            worker.completion_generation += 1
            if worker.ready and worker.jobs:
                # Processor-sharing: each request receives 1/N of worker capacity.
                next_remaining = min(r.remaining for r in worker.jobs)
                finish_at = now + max(0, next_remaining) * len(worker.jobs)
                self.push(finish_at, 3, "complete",
                          (worker.id, worker.completion_generation))

    def _node_workers(self, node: str) -> list[Worker]:
        targets = [w for w in self.workers.values() if w.node == node and not w.persistent]
        if not targets:
            raise ValueError(f"Unknown transient node {node}")
        return targets

    def _handle(self, now: float, kind: str, payload: Any) -> None:
        if kind == "arrival":
            self._assign(payload)
        elif kind == "available":
            for worker in self._node_workers(payload):
                if worker.available:
                    raise RuntimeError(f"Duplicate availability for {worker.id}")
                worker.available = True
                worker.generation += 1
                self.startups += 1
                self.push(now + self.startup_seconds, 2, "ready",
                          (worker.id, worker.generation))
            self.event_log.append({"time": now, "action": "available", "node": payload})
        elif kind == "ready":
            worker_id, generation = payload
            worker = self.workers[worker_id]
            if worker.generation != generation or not worker.available:
                return
            worker.ready = True
            if self.cold_start_work_seconds:
                self._cold_workers.add(worker.id)
            self.event_log.append({"time": now, "action": "ready", "worker": worker_id})
            self._flush_pending()
        elif kind == "reclaim":
            for worker in self._node_workers(payload):
                worker.available = False
                worker.ready = False
                worker.generation += 1
                self._cold_workers.discard(worker.id)
                interrupted = worker.jobs
                worker.jobs = []
                for request in interrupted:
                    if self.mode == "naive":
                        lost = max(0.0, request.total_work - request.remaining)
                        request.wasted_work += lost
                        self.lost_work += lost
                        request.remaining = request.total_work
                        request.interruptions += 1
                        self.interruptions += 1
                    else:
                        # Ideal keeps exactly the remaining useful work, with
                        # zero migration/startup cost. Never adds compute.
                        self.migrations += 1
                    self.pending.append(request)
            self.reclaims += 1
            self._flush_pending()
            self.event_log.append({"time": now, "action": "reclaim", "node": payload})
        elif kind == "complete":
            worker_id, generation = payload
            worker = self.workers[worker_id]
            if not worker.ready or worker.completion_generation != generation:
                return
            completed = [r for r in worker.jobs if r.remaining <= EPS]
            if not completed:
                raise RuntimeError("Completion event had no completed request")
            for r in completed:
                r.completed = now
            worker.jobs = [r for r in worker.jobs if r.remaining > EPS]
        else:
            raise RuntimeError(f"Unknown event {kind}")

    def run(self) -> dict:
        while self.heap:
            now, _, _, kind, payload = heapq.heappop(self.heap)
            if now > self.stop + EPS:
                break
            # Ignore invalidated completion/ready events *before* advancing.
            if kind == "complete":
                worker_id, generation = payload
                worker = self.workers[worker_id]
                if not worker.ready or worker.completion_generation != generation:
                    continue
            if kind == "ready":
                worker_id, generation = payload
                worker = self.workers[worker_id]
                if not worker.available or worker.generation != generation:
                    continue
            self._advance(now)
            self._handle(now, kind, payload)
            self._reschedule(now)
        self._advance(self.end if self.clock < self.end else self.clock)
        measured = [r for r in self.requests if self.warmup_seconds - EPS <= r.arrived < self.end - EPS]
        completed = [r for r in measured if r.completed is not None]
        latencies = [r.completed - r.arrived for r in completed]
        on_time = sum(r.completed - r.arrived <= self.slo + EPS for r in completed)
        window_completions = sum(r.completed is not None and r.completed < self.end
                                 for r in measured)
        outstanding = sum(r.completed is None or r.completed >= self.end for r in measured)
        avg_ready = sum(w.ready_seconds_measured for w in self.workers.values()) / self.measurement_seconds
        return {
            "mode": self.mode,
            "model_kind": "UNVALIDATED MACRO PROCESSOR-SHARING PROXY (not native Frontier)",
            "measurement_window_s": [self.warmup_seconds, self.end],
            "measured_arrivals": len(measured),
            "measured_completed_by_drain": len(completed),
            "measured_outstanding_at_end": outstanding,
            "latency_p50_s": pct(latencies, 0.50),
            "latency_p95_s": pct(latencies, 0.95),
            "latency_p99_s": pct(latencies, 0.99),
            "slo_seconds": self.slo,
            "slo_attainment": on_time / len(measured) if measured else 0.0,
            "slo_goodput_rps": on_time / self.measurement_seconds,
            "measured_completion_throughput_in_window_rps": window_completions / self.measurement_seconds,
            "average_ready_gpus": avg_ready,
            "transient_startups": self.startups,
            "node_reclaims": self.reclaims,
            "interrupted_attempts": self.interruptions,
            "zero_cost_migrations": self.migrations,
            "wasted_proxy_work_seconds": self.lost_work,
            "pending_at_stop": len(self.pending),
            "request_count": len(self.requests),
        }


def run_scenarios(
    trace: Path,
    profile: Path,
    output_dir: Path,
    lifetime_seconds: float,
    startup_seconds: float,
    warmup_seconds: float = 30.0,
    measurement_seconds: float = 300.0,
    slo_seconds: float = 30.0,
    drain_seconds: float = 600.0,
    schedule_file: Path | None = None,
    cold_start_work_seconds: float = 0.0,
    persistent_workers: int = 4,
    stable_workers: int = 8,
    gpus_per_transient_node: int = 4,
) -> list[dict]:
    profile_data = json.loads(profile.read_text(encoding="utf-8"))
    work = ProxyWorkModel(profile_data)
    end = warmup_seconds + measurement_seconds
    schedule = (read_schedule(schedule_file) if schedule_file
                else rotating_schedule(end, lifetime_seconds))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in ("fixed_persistent", "fixed_stable", "naive", "ideal"):
        # New Request state for every mode; identical source arrivals and work.
        requests = read_trace(trace, work)
        sim = Experiment(
            requests, mode, schedule, startup_seconds, warmup_seconds,
            measurement_seconds, slo_seconds, drain_seconds,
            persistent_workers=persistent_workers,
            stable_workers=stable_workers,
            gpus_per_transient_node=gpus_per_transient_node,
            cold_start_work_seconds=cold_start_work_seconds,
        )
        summary = sim.run()
        summary.update({
            "startup_seconds_assumed": startup_seconds,
            "transient_lifetime_seconds": lifetime_seconds if not schedule_file else None,
            "cold_start_work_seconds_assumed": cold_start_work_seconds,
            "source_trace": str(trace),
            "service_profile": str(profile),
            "profile_direct_matches": work.direct_matches,
            "profile_interpolated_matches": work.estimated_matches,
        })
        directory = output_dir / mode
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (directory / "requests.csv").open("w", newline="") as out:
            fields = ["request_id", "source_index", "arrival_s", "completion_s",
                      "latency_s", "input_tokens", "output_tokens", "attempts",
                      "interruptions", "wasted_proxy_work_s", "total_proxy_work_s"]
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for r in requests:
                writer.writerow({
                    "request_id": r.id, "source_index": r.source_index,
                    "arrival_s": r.arrived, "completion_s": r.completed,
                    "latency_s": r.completed - r.arrived if r.completed is not None else "",
                    "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                    "attempts": r.attempts, "interruptions": r.interruptions,
                    "wasted_proxy_work_s": r.wasted_work,
                    "total_proxy_work_s": r.total_work,
                })
        with (directory / "worker_events.jsonl").open("w") as out:
            for event in sim.event_log:
                out.write(json.dumps(event, sort_keys=True) + "\n")
        results.append(summary)
        print(f"{mode:18} arrivals={summary['measured_arrivals']:5} "
              f"slo={summary['slo_attainment']:.1%} "
              f"goodput={summary['slo_goodput_rps']:.3f} "
              f"p95={summary['latency_p95_s']} "
              f"ready_gpus={summary['average_ready_gpus']:.2f}")
    fields = ["mode", "measured_arrivals", "measured_completed_by_drain",
              "slo_attainment", "slo_goodput_rps", "latency_p50_s",
              "latency_p95_s", "latency_p99_s", "average_ready_gpus",
              "interrupted_attempts", "zero_cost_migrations",
              "wasted_proxy_work_seconds", "measured_outstanding_at_end"]
    with (output_dir / "comparison.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    manifest = {
        "warning": (
            "EXPLORATORY PROXY ONLY. Per-request Frontier low-load elapsed "
            "execution times do not model native vLLM continuous batching. "
            "Do NOT report numerical latency, capacity or policy advantage "
            "as calibrated Frontier results."
        ),
        "lifecycle": "event-driven; 4 candidate GPUs alternate between transient nodes",
        "modes": [r["mode"] for r in results],
        "startup_seconds": startup_seconds,
        "slo_seconds": slo_seconds,
        "schedule_file": str(schedule_file) if schedule_file else None,
        "schedule": [vars(e) for e in schedule],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True,
                   help="CSV from examples/malleserve/convert_mooncake.py")
    p.add_argument("--service-profile", type=Path, required=True,
                   help="JSON from transient/service_profile.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lifetime-seconds", type=float, default=60.0,
                   help="Synthetic rotation period; ignored for supplied schedule")
    p.add_argument("--schedule-jsonl", type=Path,
                   help="Optional actual/synthetic availability event schedule")
    p.add_argument("--startup-seconds", type=float, required=True,
                   help="Explicit assumed/measured startup duration for naive mode")
    p.add_argument("--cold-start-work-seconds", type=float, default=0.0,
                   help="OPTIONAL proxy penalty for first request on each returned GPU")
    p.add_argument("--warmup-seconds", type=float, default=30)
    p.add_argument("--measurement-seconds", type=float, default=300)
    p.add_argument("--slo-seconds", type=float, default=30)
    p.add_argument("--drain-seconds", type=float, default=600)
    p.add_argument("--persistent-workers", type=int, default=4)
    p.add_argument("--stable-workers", type=int, default=8)
    p.add_argument("--gpus-per-transient-node", type=int, default=4)
    a = p.parse_args()
    run_scenarios(
        trace=a.trace, profile=a.service_profile, output_dir=a.output_dir,
        lifetime_seconds=a.lifetime_seconds, startup_seconds=a.startup_seconds,
        warmup_seconds=a.warmup_seconds, measurement_seconds=a.measurement_seconds,
        slo_seconds=a.slo_seconds, drain_seconds=a.drain_seconds,
        schedule_file=a.schedule_jsonl,
        cold_start_work_seconds=a.cold_start_work_seconds,
        persistent_workers=a.persistent_workers,
        stable_workers=a.stable_workers,
        gpus_per_transient_node=a.gpus_per_transient_node,
    )


if __name__ == "__main__":
    main()
