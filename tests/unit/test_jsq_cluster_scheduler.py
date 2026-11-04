"""JSQ routing regression checks without running a Frontier simulation."""
import unittest
from types import SimpleNamespace

from frontier.scheduler.cluster_scheduler.jsq_cluster_scheduler import (
    JSQClusterScheduler,
)
from frontier.types import ClusterSchedulerType


class JSQClusterSchedulerTests(unittest.TestCase):
    def make_scheduler(self, *, loads=(0, 0, 0, 0), ready=None):
        scheduler = JSQClusterScheduler.__new__(JSQClusterScheduler)
        scheduler._cluster = SimpleNamespace(
            replicas={i: object() for i in range(len(loads))}
        )
        scheduler._replica_dp_size = 1
        scheduler._replica_schedulers = {
            (i, 0): SimpleNamespace(num_pending_requests=load)
            for i, load in enumerate(loads)
        }
        scheduler._request_queue = []
        scheduler._tie_cursor = 0
        if ready is not None:
            scheduler._transient_ready_replica_ids = set(ready)
        return scheduler

    def test_enum_and_registry(self):
        from frontier.scheduler.cluster_scheduler.cluster_scheduler_registry import (
            ClusterSchedulerRegistry,
        )
        self.assertEqual(ClusterSchedulerType.from_str("jsq"), ClusterSchedulerType.JSQ)
        self.assertIs(ClusterSchedulerRegistry.get_key_from_str("jsq"), ClusterSchedulerType.JSQ)

    def test_assigns_to_shortest_queue_and_counts_in_event(self):
        scheduler = self.make_scheduler(loads=(4, 0, 1))
        scheduler._request_queue = [SimpleNamespace(_arrived_at=i) for i in range(4)]
        placements = scheduler.schedule()
        self.assertEqual([replica for replica, _, _ in placements], [1, 2, 1, 2])
        self.assertEqual(scheduler._request_queue, [])

    def test_excludes_unready_transient_replicas(self):
        scheduler = self.make_scheduler(loads=(8, 0, 2), ready={0, 2})
        scheduler._request_queue = [SimpleNamespace(_arrived_at=0)]
        self.assertEqual(scheduler.schedule()[0][0], 2)

    def test_no_ready_replica_retains_requests(self):
        scheduler = self.make_scheduler(loads=(0, 0), ready=set())
        request = SimpleNamespace(_arrived_at=0)
        scheduler._request_queue = [request]
        self.assertEqual(scheduler.schedule(), [])
        self.assertEqual(scheduler._request_queue, [request])

    def test_round_robin_ties(self):
        scheduler = self.make_scheduler(loads=(0, 0, 0, 0))
        scheduler._request_queue = [SimpleNamespace(_arrived_at=i) for i in range(4)]
        self.assertEqual(
            [replica for replica, _, _ in scheduler.schedule()],
            [0, 1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()
