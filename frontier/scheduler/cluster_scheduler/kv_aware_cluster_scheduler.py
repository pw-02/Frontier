"""KV-prefix-aware, token-work-aware routing for co-located vLLM V1 workers.

This is a configurable heuristic, not NVIDIA Dynamo's exact router or a
calibrated response-time prediction. No KV transfer or cross-worker sharing.
"""

import math
from typing import List, Optional, Tuple

from frontier.entities import Request
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


class KVAwareClusterScheduler(BaseClusterScheduler):
    """Choose a ready worker minimising prefill miss + outstanding token work.

    Cost is in *prefill-equivalent tokens*:
        missing_prompt_tokens
        + load_weight * (
            worker_pending_prefill_tokens
            + decode_token_weight * worker_pending_decode_tokens
          )

    Running and waiting requests both contribute to pending work. Estimating
    remaining work is a deliberately simple heuristic; decode_token_weight is
    tunable to reflect different relative decode/prefill costs. For multiple
    requests routed in one event, account for each earlier assignment without
    pretending its KV blocks have already been computed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._cluster_type != ClusterType.MONOLITHIC:
            raise ValueError("kv_aware currently supports monolithic serving only")
        if self._replica_dp_size != 1:
            raise ValueError("kv_aware requires one attention-DP lane per replica")
        self._load_weight = float(self._config.cluster_scheduler_config.load_weight)
        self._decode_token_weight = float(
            self._config.cluster_scheduler_config.decode_token_weight
        )
        for name, weight in (
            ("load_weight", self._load_weight),
            ("decode_token_weight", self._decode_token_weight),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"kv_aware {name} must be finite and nonnegative")
        for replica_id in self._cluster.replicas:
            lane = self.get_replica_scheduler(replica_id, 0)
            if not isinstance(lane, VLLMv1EngineReplicaScheduler):
                raise ValueError("kv_aware requires vllm_v1 replica schedulers")

    def schedule(self) -> List[Tuple[int, Optional[int], Request]]:
        self.sort_requests()
        ready = getattr(self, "_transient_ready_replica_ids", None)
        lanes = {}
        projected = {}
        for replica_id in self._cluster.replicas:
            if ready is not None and replica_id not in ready:
                continue
            lane = self.get_replica_scheduler(replica_id, 0)
            lanes[replica_id] = lane
            projected[replica_id] = lane.get_routing_workload()

        if not lanes:
            return []  # retain arrivals until capacity comes online

        placements: List[Tuple[int, Optional[int], Request]] = []
        while self._request_queue:
            request = self._request_queue[0]
            prompt = int(request.num_prefill_tokens)
            decode = int(request.total_tokens) - prompt

            def cost(replica_id: int) -> tuple[float, int]:
                lane = lanes[replica_id]
                cached = lane.get_routing_cached_prefix_tokens(request)
                missing = max(0, prompt - cached)
                state = projected[replica_id]
                outstanding_work = (
                    state["prefill_tokens"]
                    + self._decode_token_weight * state["decode_tokens"]
                )
                return (missing + self._load_weight * outstanding_work, replica_id)

            target = min(lanes, key=cost)
            self._request_queue.pop(0)
            placements.append((target, 0, request))
            projected[target]["prefill_tokens"] += max(
                0, prompt - lanes[target].get_routing_cached_prefix_tokens(request)
            )
            projected[target]["decode_tokens"] += max(0, decode)
            projected[target]["outstanding_requests"] += 1
        return placements
