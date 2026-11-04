"""Join-shortest-queue routing for monolithic LLM serving replicas.

Queue length is the number of outstanding requests reported by each replica
scheduler (waiting + admitted/running), not tokens or estimated GPU time.
For one-GPU-per-replica MalleServe workloads, this is the natural JSQ baseline.
"""

from typing import List, Optional, Tuple

from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR
from frontier.entities import Request
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


class JSQClusterScheduler(BaseClusterScheduler):
    """Route each arrival to the currently least-loaded ready replica lane.

    Count assignments selected earlier in the *same* cluster scheduling event,
    since those requests have not yet been inserted into the replica schedulers.
    Ties are broken by a rotating cursor to avoid always preferring replica 0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._cluster_type != ClusterType.MONOLITHIC:
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
        self._tie_cursor = 0

    @staticmethod
    def _choose_lane(loads: dict[tuple[int, int], int], cursor: int) -> tuple[int, int]:
        """Choose the least loaded lane; rotate ties in stable lane order."""
        candidates = sorted(loads)
        if not candidates:
            raise ValueError("JSQ requires at least one eligible replica lane")
        minimum = min(loads.values())
        tied = [lane for lane in candidates if loads[lane] == minimum]
        return tied[cursor % len(tied)]

    def schedule(self) -> List[Tuple[int, Optional[int], Request]]:
        self.sort_requests()
        ready = getattr(self, "_transient_ready_replica_ids", None)
        loads: dict[tuple[int, int], int] = {}
        for replica_id in self._cluster.replicas:
            if ready is not None and replica_id not in ready:
                continue
            for lane_id in range(self._replica_dp_size):
                key = (replica_id, lane_id)
                loads[key] = int(self._replica_schedulers[key].num_pending_requests)

        # No ready capacity: keep requests in the cluster queue until a
        # replica is activated. Do not discard or route to an offline worker.
        if not loads:
            return []

        placements: List[Tuple[int, Optional[int], Request]] = []
        while self._request_queue:
            target = self._choose_lane(loads, self._tie_cursor)
            request = self._request_queue.pop(0)
            placements.append((target[0], target[1], request))
            loads[target] += 1
            self._tie_cursor += 1
        return placements
