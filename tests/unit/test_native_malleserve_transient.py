"""CPU-only lifecycle contract tests for native Frontier transient replicas.

They exercise the controller and real round-robin placement without
requiring profiling CSVs or a GPU. Run:
    python -m unittest discover -s tests/unit -p test_native_malleserve_transient.py -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from frontier.transient_lifecycle import (
    AvailabilityChange, NativeTransientLifecycle, read_availability,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)


class FakeRequest:
    def __init__(self, id, original_prompt=128, output=16, processed=145):
        self.id = id
        self._user_facing_num_prefill_tokens = original_prompt
        self._user_facing_num_decode_tokens = output
        self._num_prefill_tokens = original_prompt
        self._num_decode_tokens = output
        self._num_processed_tokens = processed
        self.completed = False
        self.restart_called = 0
        self.arrived = None

    @property
    def num_processed_tokens(self):
        return self._num_processed_tokens

    def restart(self):
        self.restart_called += 1
        total = self._num_prefill_tokens + self._num_decode_tokens
        self._num_prefill_tokens = self._num_processed_tokens
        self._num_decode_tokens = total - self._num_prefill_tokens
        self._num_processed_tokens = 0

    def on_enter_waiting_queue(self, time, cluster):
        self.arrived = time


class FakeReplicaScheduler:
    def __init__(self, requests=()):
        self._request_queue = []
        self._running_requests = list(requests)
        self._waiting_requests = []
        self._preempted_requests = []
        self._allocation_map = {r.id: 1 for r in requests}


class FakeParent:
    def __init__(self, replica_ids, occupied):
        self._cluster = SimpleNamespace(replicas={i: object() for i in replica_ids})
        self._replicas = dict(occupied)
        self._request_queue = []
        self._transient_ready_replica_ids = set()

    def get_replica_scheduler(self, replica_id, lane):
        return self._replicas[replica_id]

    def add_request(self, request):
        self._request_queue.append(request)


def fake_controller(occupied=(), startup=4):
    controller = object.__new__(NativeTransientLifecycle)
    controller.scheduler = FakeParent(
        [0, 1], {0: FakeReplicaScheduler(), 1: FakeReplicaScheduler(occupied)})
    controller.node_replicas = {"transient-a": [1]}
    controller.ready_ids = {0}
    controller.available_nodes = set()
    controller.node_generation = {"transient-a": 0}
    controller.lifecycle_records = []
    controller.startup = startup
    controller.retries = 0
    controller.lost_processed_tokens = 0
    controller.stale_event_purges = 0
    controller.recovery_mode = "naive"
    controller.oracle_restored_requests = 0
    controller.oracle_restored_tokens = 0
    controller.reclamation_diagnostics = []
    controller.retried_request_ids = set()
    controller._fresh_scheduler = lambda replica_id: controller.scheduler._replicas.__setitem__(
        replica_id, FakeReplicaScheduler())
    return controller


class AvailabilityParsingTests(unittest.TestCase):
    def test_read_schedule_and_validate_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.jsonl"
            lines = [
                {"timestamp": 10, "action": "reclaim", "node_id": "a"},
                {"timestamp": 0, "action": "available", "node_id": "a"},
            ]
            path.write_text("".join(json.dumps(x) + "\n" for x in lines))
            self.assertEqual(
                read_availability(str(path)),
                [AvailabilityChange(0, "a", "available"),
                 AvailabilityChange(10, "a", "reclaim")],
            )

    def test_available_without_reclaim_is_valid_for_no_reclamation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.jsonl"
            path.write_text(json.dumps({
                "timestamp": 0, "node_id": "transient-0", "action": "available"
            }) + "\n")
            events = read_availability(str(path))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].action, "available")

    def test_unmatched_reclaim_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.jsonl"
            path.write_text(
                json.dumps({"timestamp": 2, "action": "reclaim", "node_id": "a"})
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "before availability"):
                read_availability(str(path))


class ControllerTests(unittest.TestCase):
    def test_ready_delay_and_stale_startup(self):
        c = fake_controller()
        events = c.change(AvailabilityChange(0, "transient-a", "available"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].time, 4)
        self.assertEqual(c.ready_ids, {0})
        c.change(AvailabilityChange(2, "transient-a", "reclaim"))
        c.change(AvailabilityChange(4, "transient-a", "ready"), generation=1)
        self.assertEqual(c.ready_ids, {0})

    def test_ready_worker_joins_router(self):
        c = fake_controller()
        c.change(AvailabilityChange(0, "transient-a", "available"))
        c.change(AvailabilityChange(4, "transient-a", "ready"), generation=1)
        self.assertEqual(c.ready_ids, {0, 1})

    def test_reclaim_resets_kv_and_retries_once(self):
        r = FakeRequest(44, processed=145)
        c = fake_controller((r,))
        c.change(AvailabilityChange(0, "transient-a", "available"))
        c.change(AvailabilityChange(4, "transient-a", "ready"), generation=1)
        generated = c.change(AvailabilityChange(7, "transient-a", "reclaim"))
        self.assertEqual(c.ready_ids, {0})
        self.assertEqual(c.retries, 1)
        self.assertEqual(c.lost_processed_tokens, 145)
        self.assertEqual(len(c.scheduler._request_queue), 1)
        self.assertIs(c.scheduler._request_queue[0], r)
        self.assertEqual(r.restart_called, 1)
        self.assertEqual(r.arrived, 7)
        self.assertEqual((r._num_prefill_tokens, r._num_decode_tokens), (128, 16))
        self.assertEqual(r._num_processed_tokens, 0)
        self.assertEqual(len(generated), 1)
        self.assertEqual(c.scheduler._replicas[1]._allocation_map, {})

    def test_reclamation_diagnostics_count_queued_and_running_once(self):
        running = FakeRequest(3, processed=145)
        queued = FakeRequest(4, processed=0)
        c = fake_controller((running,))
        worker = c.scheduler._replicas[1]
        worker._request_queue = [queued]
        # Simulate a request also referenced by the waiting collection.
        worker._waiting_requests = [running]
        c.change(AvailabilityChange(0, "transient-a", "available"))
        c.change(AvailabilityChange(4, "transient-a", "ready"), generation=1)
        c.change(AvailabilityChange(7, "transient-a", "reclaim"))
        diagnostic = c.reclamation_diagnostics[0]
        self.assertEqual(diagnostic["unique_interrupted"], 2)
        self.assertEqual(diagnostic["queued_only"], 1)
        self.assertEqual(diagnostic["never_processed"], 1)
        self.assertEqual(diagnostic["replicas"][0]["deduplicated_states"]["running"], 1)
        self.assertEqual(c.retries, 2)
        self.assertEqual(c.retried_request_ids, {3, 4})

    def test_duplicate_global_request_fails_closed(self):
        r = FakeRequest(8)
        c = fake_controller((r,))
        c.scheduler._request_queue = [r]
        c.change(AvailabilityChange(0, "transient-a", "available"))
        with self.assertRaisesRegex(RuntimeError, "simultaneously globally queued"):
            c.change(AvailabilityChange(3, "transient-a", "reclaim"))

    def test_ideal_recovery_preserves_committed_progress(self):
        r = FakeRequest(55, original_prompt=128, output=16, processed=137)
        c = fake_controller((r,))
        c.recovery_mode = "ideal_recovery"
        c.change(AvailabilityChange(0, "transient-a", "available"))
        c.change(AvailabilityChange(7, "transient-a", "reclaim"))
        self.assertEqual(c.retries, 1)
        self.assertEqual(c.lost_processed_tokens, 0)
        self.assertEqual(c.oracle_restored_tokens, 137)
        self.assertEqual(r._oracle_restore_pending, 137)
        self.assertEqual(r.num_processed_tokens, 137)

    def test_allocation_ownership_check_fails_closed(self):
        worker = FakeReplicaScheduler()
        worker._allocation_map = {123: 4}
        with self.assertRaisesRegex(RuntimeError, "no request owner"):
            NativeTransientLifecycle._unfinished_requests(worker)

    def test_round_robin_only_uses_ready_replicas(self):
        scheduler = object.__new__(RoundRobinClusterScheduler)
        scheduler._cluster = SimpleNamespace(replicas={0: None, 1: None, 2: None})
        scheduler._replica_dp_size = 1
        scheduler._request_counter = 0
        scheduler._request_queue = [FakeRequest(i) for i in range(4)]
        scheduler._transient_ready_replica_ids = {0, 2}
        mapping = scheduler._schedule_lane_round_robin()
        self.assertEqual([row[0] for row in mapping], [0, 2, 0, 2])
        self.assertEqual(scheduler._request_queue, [])

    def test_no_ready_replica_retains_queue(self):
        scheduler = object.__new__(RoundRobinClusterScheduler)
        scheduler._cluster = SimpleNamespace(replicas={0: None, 1: None})
        scheduler._replica_dp_size = 1
        scheduler._request_counter = 0
        pending = FakeRequest(11)
        scheduler._request_queue = [pending]
        scheduler._transient_ready_replica_ids = set()
        self.assertEqual(scheduler._schedule_lane_round_robin(), [])
        self.assertEqual(scheduler._request_queue, [pending])


if __name__ == "__main__":
    unittest.main()
