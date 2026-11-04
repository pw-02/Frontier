"""Read-only KV-prefix and token-work routing regression tests."""
import unittest
from types import SimpleNamespace

from frontier.kv_cache.base_kv_cache_manager import KVCacheManager
from frontier.scheduler.cluster_scheduler.kv_aware_cluster_scheduler import (
    KVAwareClusterScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterSchedulerType


def request(req_id, prefill=1000, decode=0, processed=0):
    return SimpleNamespace(
        id=req_id, _arrived_at=req_id, num_prefill_tokens=prefill,
        num_processed_tokens=processed, total_tokens=prefill + decode,
    )


class Lane:
    def __init__(self, *, prefill=0, decode=0, cached=0):
        self.prefill = prefill
        self.decode = decode
        self.cached = cached

    def get_routing_workload(self):
        return {
            "prefill_tokens": self.prefill,
            "decode_tokens": self.decode,
            "outstanding_requests": 0,
        }

    def get_routing_cached_prefix_tokens(self, req):
        return self.cached


class KVAwareRoutingTests(unittest.TestCase):
    def scheduler(self, lanes, ready=None):
        obj = KVAwareClusterScheduler.__new__(KVAwareClusterScheduler)
        obj._cluster = SimpleNamespace(replicas={i: None for i in range(len(lanes))})
        obj._replica_dp_size = 1
        obj._replica_schedulers = {(i, 0): lane for i, lane in enumerate(lanes)}
        obj._request_queue = []
        obj._load_weight = 1.0
        obj._cache_weight = 1.0
        obj._decode_token_weight = 16.0
        if ready is not None:
            obj._transient_ready_replica_ids = set(ready)
        return obj

    def test_cache_reuse_helps_when_work_equal(self):
        obj = self.scheduler([Lane(cached=800), Lane(cached=0)])
        obj._request_queue = [request(0)]
        self.assertEqual(obj.schedule()[0][0], 0)

    def test_token_only_ignores_incoming_cache_overlap(self):
        # Worker 0 has more cached tokens but also more outstanding work.
        # With cache_weight=0, only outstanding work affects placement.
        obj = self.scheduler([Lane(prefill=100, cached=900), Lane(cached=0)])
        obj._cache_weight = 0.0
        obj._request_queue = [request(0)]
        self.assertEqual(obj.schedule()[0][0], 1)

    def test_cache_awareness_reverses_token_only_decision(self):
        obj = self.scheduler([Lane(prefill=100, cached=900), Lane(cached=0)])
        obj._cache_weight = 1.0
        obj._request_queue = [request(0)]
        self.assertEqual(obj.schedule()[0][0], 0)

    def test_active_token_work_can_outweigh_cache_hit(self):
        obj = self.scheduler([Lane(prefill=2000, cached=800), Lane()])
        obj._request_queue = [request(0)]
        self.assertEqual(obj.schedule()[0][0], 1)

    def test_decode_work_contributes_to_load(self):
        obj = self.scheduler([Lane(decode=100, cached=500), Lane()])
        obj._request_queue = [request(0)]
        self.assertEqual(obj.schedule()[0][0], 1)

    def test_same_event_assignments_counted(self):
        obj = self.scheduler([Lane(), Lane()])
        obj._request_queue = [request(0), request(1)]
        self.assertEqual([x[0] for x in obj.schedule()], [0, 1])

    def test_unready_workers_excluded_and_queue_retained(self):
        obj = self.scheduler([Lane(), Lane(cached=900)], ready={0})
        obj._request_queue = [request(0)]
        self.assertEqual(obj.schedule()[0][0], 0)
        obj._transient_ready_replica_ids = set()
        waiting = request(1)
        obj._request_queue = [waiting]
        self.assertEqual(obj.schedule(), [])
        self.assertEqual(obj._request_queue, [waiting])

    def test_vllm_snapshot_counts_running_as_well_as_waiting(self):
        lane = VLLMv1EngineReplicaScheduler.__new__(VLLMv1EngineReplicaScheduler)
        lane._preempted_requests = []
        lane._request_queue = [request(1, prefill=1000, decode=10)]
        lane._waiting_requests = []
        lane._running_requests = [request(2, prefill=200, decode=50, processed=220)]
        state = lane.get_routing_workload()
        self.assertEqual(state["outstanding_requests"], 2)
        self.assertEqual(state["prefill_tokens"], 1000)
        self.assertEqual(state["decode_tokens"], 40)

    def test_prefix_probe_does_not_mutate_hit_stats(self):
        manager = KVCacheManager.__new__(KVCacheManager)
        manager.enable_caching = True
        manager.block_size = 16
        cached = {11, 22}
        manager.block_pool = SimpleNamespace(get_cached_block=lambda h: h if h in cached else None)
        manager.prefix_cache_stats = SimpleNamespace(requests=0, queries=0, hits=0)
        req = SimpleNamespace(
            num_prefill_tokens=64,
            prompt_block_hash_ids=lambda block_size: [11, 22, 33, 44],
        )
        self.assertEqual(manager.count_cached_prefix_tokens(req), 32)
        self.assertEqual(manager.prefix_cache_stats.hits, 0)
        self.assertEqual(manager.prefix_cache_stats.queries, 0)
        self.assertEqual(manager.prefix_cache_stats.requests, 0)

    def test_cli_type_discoverable(self):
        from frontier.config.cluster_scheduler_config import (
            BaseClusterSchedulerConfig, KVAwareClusterSchedulerConfig,
        )
        from frontier.config.utils import get_all_subclasses
        subclasses = {str(cls.get_type()): cls for cls in get_all_subclasses(BaseClusterSchedulerConfig)}
        self.assertIs(subclasses["kv_aware"], KVAwareClusterSchedulerConfig)
        self.assertEqual(ClusterSchedulerType.from_str("kv_aware"), ClusterSchedulerType.KV_AWARE)


if __name__ == "__main__":
    unittest.main()
