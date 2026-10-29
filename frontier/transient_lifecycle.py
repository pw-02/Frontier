"""Opt-in native Frontier transient replica lifecycle (monolithic vLLM v1).

This module manages *existing* Replica schedulers; service work continues through
the ordinary Frontier batch-stage event engine and GPU predictor. It does not
estimate request service duration itself.

Initial scope is intentionally narrow: sequential, monolithic, single DP lane,
single GPU/replica, no thinking rounds, no speculative decoding and naive full
request replay on abrupt reclamation. Fail closed for unsupported settings.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.utils.replica_schedulers import _create_scheduler
from frontier.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from frontier.types import ClusterType


@dataclass(frozen=True)
class AvailabilityChange:
    time: float
    node: str
    action: str


def read_availability(path: str) -> list[AvailabilityChange]:
    events = []
    live = set()
    first_seen = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if set(raw) != {"timestamp", "node_id", "action"}:
                raise ValueError(f"Availability line {line_no} requires timestamp,node_id,action")
            at = raw["timestamp"]
            node = raw["node_id"]
            action = raw["action"]
            if type(at) not in (float, int) or not math.isfinite(at) or at < 0:
                raise ValueError(f"Availability line {line_no}: invalid timestamp")
            if not isinstance(node, str) or not node.strip():
                raise ValueError(f"Availability line {line_no}: invalid node_id")
            if action not in ("available", "reclaim"):
                raise ValueError(f"Availability line {line_no}: invalid action {action}")
            events.append(AvailabilityChange(float(at), node.strip(), action))
            if node.strip() not in first_seen:
                first_seen.append(node.strip())
    events.sort(key=lambda e: (e.time, 0 if e.action == "reclaim" else 1))
    for e in events:
        if e.action == "available":
            if e.node in live:
                raise ValueError(f"Duplicate availability for {e.node} at {e.time}")
            live.add(e.node)
        else:
            if e.node not in live:
                raise ValueError(f"Reclaim before availability for {e.node} at {e.time}")
            live.remove(e.node)
    if not events:
        raise ValueError("Transient availability trace is empty")
    return events


class NativeTransientLifecycle:
    """Lifecycle controller for a precreated native Frontier replica pool."""

    def __init__(self, config, global_scheduler):
        if config.sys_arch != "co-location":
            raise ValueError("Native transient lifecycle requires --sys_arch co-location")
        if config.simulation_mode != "online":
            raise ValueError("Native transient lifecycle requires --simulation_mode online")
        if getattr(config, "enable_thinking_mode", False):
            raise ValueError("Native transient lifecycle does not support thinking mode")
        if getattr(config.cluster_config.replica_config, "attn_dp", 1) != 1:
            raise ValueError("Native transient lifecycle requires attn_dp=1")
        if getattr(config.cluster_config.replica_config, "world_size", 1) != 1:
            raise ValueError("Native transient lifecycle requires world_size=1")
        if getattr(config.cluster_config.replica_config, "num_pipeline_stages", 1) != 1:
            raise ValueError("Native transient lifecycle requires no pipeline parallelism")
        # The current implementation explicitly supports vLLM v1 engine scheduling,
        # not Orca or speculative/multiround state machines.
        self.scheduler = global_scheduler.get_cluster_scheduler(ClusterType.MONOLITHIC)
        from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
            RoundRobinClusterScheduler,
        )
        if type(self.scheduler) is not RoundRobinClusterScheduler:
            raise ValueError("Native transient lifecycle requires round_robin cluster scheduler")
        replica_config = config.cluster_config.replica_config
        if getattr(getattr(replica_config, "speculative_decoding_config", None),
                   "enabled", False):
            raise ValueError("Native transient lifecycle does not support speculative decoding")
        self.cluster = self.scheduler._cluster
        self.ids = list(self.cluster.replicas)
        if self.scheduler._replica_dp_size != 1:
            raise ValueError("Native transient lifecycle requires one scheduler lane/replica")
        if any(
            not isinstance(self.scheduler.get_replica_scheduler(replica_id, 0),
                           VLLMv1EngineReplicaScheduler)
            for replica_id in self.ids
        ):
            raise ValueError("Native transient lifecycle needs vllm_v1 engine schedulers")
        if any(
            getattr(self.scheduler.get_replica_scheduler(replica_id, 0),
                    "_spec_decode_enabled", False)
            for replica_id in self.ids
        ):
            raise ValueError("Speculative decoding under transient reclamation is unsupported")
        self.events = read_availability(config.transient_schedule_file)
        nodes = list(dict.fromkeys(e.node for e in self.events))
        persistent = int(config.transient_persistent_replicas)
        per_node = int(config.transient_replicas_per_node)
        if persistent <= 0 or per_node <= 0:
            raise ValueError("Persistent and per-node replica counts must be positive")
        needed = persistent + per_node * len(nodes)
        if needed != len(self.ids):
            raise ValueError(
                f"Precreated replicas={len(self.ids)} but availability requires "
                f"{persistent} persistent + {len(nodes)}*{per_node} candidate = {needed}"
            )
        self.node_replicas = {
            node: self.ids[persistent + i * per_node:persistent + (i + 1) * per_node]
            for i, node in enumerate(nodes)
        }
        startup = float(config.transient_startup_seconds)
        if not math.isfinite(startup) or startup < 0:
            raise ValueError("transient_startup_seconds must be finite and nonnegative")
        self.startup = startup
        self.ready_ids: set[int] = set(self.ids[:persistent])
        self.available_nodes: set[str] = set()
        self.node_generation = {node: 0 for node in nodes}
        self.scheduler._transient_ready_replica_ids = self.ready_ids
        self.lifecycle_records: list[dict] = []
        self.retries = 0
        self.lost_processed_tokens = 0
        self.stale_event_purges = 0

    def _fresh_scheduler(self, replica_id: int) -> None:
        """Discard cache, running-batch and stage state; keep GPU predictor."""
        parent = self.scheduler
        cluster_config = parent._config
        replica_config = cluster_config.replica_config
        replica_scheduler_config = parent._get_cluster_specific_replica_scheduler_config(
            cluster_config, ClusterType.MONOLITHIC)
        new_scheduler = _create_scheduler(
            scheduler_type=replica_scheduler_config.get_type(),
            replica_config=replica_config,
            scheduler_config=replica_scheduler_config,
            request_generator_config=parent._request_generator_config,
            replica=parent._cluster.replicas[replica_id],
            predictor=parent._predictor,
            cluster_type=ClusterType.MONOLITHIC,
            replica_local_id=0,
            af_pipeline_num_micro_batch=getattr(
                cluster_config, "af_pipeline_num_micro_batch", -1),
            cluster_scheduler=parent,
            registry=ReplicaSchedulerRegistry,
        )
        if not isinstance(new_scheduler, VLLMv1EngineReplicaScheduler):
            raise RuntimeError("Reconstructed unexpected scheduler type")
        parent._replica_schedulers[(replica_id, 0)] = new_scheduler
        parent._full_stage_replica_schedulers[replica_id] = new_scheduler

    @staticmethod
    def _unfinished_requests(replica_scheduler) -> list:
        candidates = []
        for field in (
            "_request_queue", "_running_requests", "_waiting_requests",
            "_preempted_requests",
        ):
            candidates.extend(getattr(replica_scheduler, field, ()))
        # Validate the KV allocator does not own requests missing from queues.
        known = {r.id for r in candidates}
        allocations = set(getattr(replica_scheduler, "_allocation_map", {}))
        if not allocations.issubset(known):
            raise RuntimeError(
                "Cannot safely reclaim replica: KV allocations have no request "
                f"owner in scheduler queues: {allocations-known}"
            )
        result = []
        seen = set()
        for req in candidates:
            if req.id not in seen and not req.completed:
                result.append(req)
                seen.add(req.id)
        return result

    def change(self, event: AvailabilityChange, generation: int | None = None):
        """Apply native lifecycle transition; return scheduling events.

        The caller must purge queued GPU-stage events for reclaimed replicas
        BEFORE adding returned scheduling events to the Frontier event heap.
        """
        from frontier.events.cluster_schedule_event import ClusterScheduleEvent
        from frontier.events.replica_lifecycle_event import ReplicaLifecycleEvent

        node = event.node
        at = event.time
        if node not in self.node_replicas:
            raise ValueError(f"Unknown transient node {node}")
        replicas = self.node_replicas[node]
        result = []
        if event.action == "available":
            if node in self.available_nodes:
                raise RuntimeError(f"Node {node} already available")
            self.available_nodes.add(node)
            self.node_generation[node] += 1
            self.lifecycle_records.append({
                "timestamp": at, "action": "available", "node_id": node})
            result.append(ReplicaLifecycleEvent(
                at + self.startup, self, "ready", node,
                self.node_generation[node]))
        elif event.action == "ready":
            if node not in self.available_nodes or generation != self.node_generation[node]:
                return []
            self.ready_ids.update(replicas)
            self.lifecycle_records.append({
                "timestamp": at, "action": "ready", "node_id": node,
                "replica_ids": replicas})
            if self.scheduler._request_queue:
                result.append(ClusterScheduleEvent(at, ClusterType.MONOLITHIC))
        elif event.action == "reclaim":
            if node not in self.available_nodes:
                raise RuntimeError(f"Node {node} was not available")
            self.available_nodes.remove(node)
            self.node_generation[node] += 1
            self.ready_ids.difference_update(replicas)
            restarting = []
            for replica_id in replicas:
                old_scheduler = self.scheduler.get_replica_scheduler(replica_id, 0)
                restarting.extend(self._unfinished_requests(old_scheduler))
                self._fresh_scheduler(replica_id)
            unique = {}
            for request in restarting:
                unique[request.id] = request
            for req in unique.values():
                # Client-level retry: lose ALL prior generated output.
                # Set processed to the original prompt length before restart,
                # so restart() keeps prompt/output sizes unchanged and
                # discards all progress, not just the KV cache.
                original_prompt = int(req._user_facing_num_prefill_tokens)
                original_output = int(req._user_facing_num_decode_tokens)
                self.lost_processed_tokens += max(0, req.num_processed_tokens)
                req._num_prefill_tokens = original_prompt
                req._num_decode_tokens = original_output
                req._num_processed_tokens = original_prompt
                req.restart()
                req.on_enter_waiting_queue(at, ClusterType.MONOLITHIC)
                self.scheduler.add_request(req)
                self.retries += 1
            self.lifecycle_records.append({
                "timestamp": at, "action": "reclaim", "node_id": node,
                "replica_ids": replicas, "retried_requests": len(unique)})
            if self.scheduler._request_queue:
                result.append(ClusterScheduleEvent(at, ClusterType.MONOLITHIC))
        else:
            raise ValueError(f"Unknown lifecycle action {event.action}")
        return result
