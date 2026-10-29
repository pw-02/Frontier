"""Native lifecycle events for precreated transient monolithic replicas."""
from __future__ import annotations

from frontier.events.base_event import BaseEvent
from frontier.transient_lifecycle import AvailabilityChange
from frontier.types import ClusterType, EventType


class ReplicaLifecycleEvent(BaseEvent):
    def __init__(self, time: float, controller, action: str, node: str,
                 generation: int | None = None):
        if action not in ("available", "reclaim", "ready"):
            raise ValueError(f"Unknown lifecycle action {action}")
        super().__init__(time, EventType.REPLICA_LIFECYCLE)
        self.controller = controller
        self.action = action
        self.node = node
        self.generation = generation
        self._cluster_type = ClusterType.MONOLITHIC
        # Deterministically process loss/availability before ordinary request
        # or GPU execution at exactly the same simulator timestamp.
        self._priority_number = (time, -1, int(self.event_type))

    def handle_event(self, scheduler, metrics_store):
        return self.controller.change(
            AvailabilityChange(self.time, self.node, self.action),
            self.generation,
        )

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": "REPLICA_LIFECYCLE",
            "action": self.action,
            "node_id": self.node,
            "generation": self.generation,
        }
