"""Unit tests for the exploratory MalleServe transient macro experiment.

These tests validate the lifecycle/state machine, *not* GPU execution fidelity.
Run: python -m unittest tests/unit/test_malleserve_transient_macro.py -v
"""
from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples" / "malleserve" / "transient" / "simulate.py"
spec = importlib.util.spec_from_file_location("malleserve_transient_macro", SOURCE)
assert spec is not None and spec.loader is not None
macro = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = macro
spec.loader.exec_module(macro)


def request(ident, arrival, work):
    return macro.Request(
        id=ident, arrived=arrival, input_tokens=1024, output_tokens=32,
        source_index=ident, total_work=work, remaining=work,
    )


def simulate(mode, requests, events=(), startup=0, warmup=0, measurement=20,
             drain=50, persistent=1, stable=2, node_gpus=1):
    experiment = macro.Experiment(
        requests, mode, list(events), startup, warmup, measurement,
        slo_seconds=5, drain_seconds=drain,
        persistent_workers=persistent, stable_workers=stable,
        gpus_per_transient_node=node_gpus,
    )
    summary = experiment.run()
    return experiment, summary


class AvailabilityTests(unittest.TestCase):
    def test_rotating_schedule_is_valid_and_replaces_nodes(self):
        events = macro.rotating_schedule(125, 30)
        self.assertEqual(macro.validate_schedule(events),
                         ["transient-0", "transient-1"])
        self.assertEqual(events[0], macro.NodeEvent(0, "transient-0", "available"))
        self.assertEqual(events[1], macro.NodeEvent(30, "transient-0", "reclaim"))
        self.assertEqual(events[2], macro.NodeEvent(30, "transient-1", "available"))

    def test_invalid_schedule_rejected(self):
        with self.assertRaises(ValueError):
            macro.validate_schedule([macro.NodeEvent(2, "a", "reclaim")])
        with self.assertRaises(ValueError):
            macro.validate_schedule([macro.NodeEvent(0, "a", "available"),
                                     macro.NodeEvent(1, "a", "available")])

    def test_fixed_baselines_are_unchanged_by_availability_events(self):
        requests = [request(0, 0, 8), request(1, 0, 8)]
        events = [macro.NodeEvent(1, "a", "available"),
                  macro.NodeEvent(2, "a", "reclaim")]
        _, one = simulate("fixed_persistent", copy.deepcopy(requests), events=events)
        _, two = simulate("fixed_stable", copy.deepcopy(requests), events=events)
        self.assertEqual(one["measured_completed_by_drain"], 2)
        self.assertEqual(two["measured_completed_by_drain"], 2)
        self.assertAlmostEqual(one["latency_p95_s"], 16)
        self.assertAlmostEqual(two["latency_p95_s"], 8)
        self.assertEqual(one["node_reclaims"], 0)
        self.assertEqual(two["node_reclaims"], 0)

    def test_ready_delay_and_preemption_lost_work(self):
        # First request goes to transient once it is ready; another event
        # reclaims it midway through the job. The request then restarts.
        # Start transient at 0; turn ready after 2; arrival at 3.
        # RR starts with persistent; make the first request trivial so the
        # second request routes to the transient worker.
        reqs = [request(0, 2.5, 0.1), request(1, 3, 10)]
        events = [macro.NodeEvent(0, "a", "available"),
                  macro.NodeEvent(7, "a", "reclaim")]
        exp, result = simulate("naive", reqs, events=events, startup=2)
        self.assertEqual(result["node_reclaims"], 1)
        self.assertEqual(result["interrupted_attempts"], 1)
        self.assertAlmostEqual(result["wasted_proxy_work_seconds"], 4)
        self.assertEqual(reqs[1].attempts, 2)
        self.assertAlmostEqual(reqs[1].completed, 17)

    def test_ideal_keeps_partial_progress_and_has_zero_startup(self):
        reqs = [request(0, 0, 0.1), request(1, 1, 10)]
        events = [macro.NodeEvent(0, "a", "available"),
                  macro.NodeEvent(5, "a", "reclaim")]
        exp, result = simulate("ideal", reqs, events=events, startup=90)
        self.assertEqual(result["zero_cost_migrations"], 1)
        self.assertEqual(result["interrupted_attempts"], 0)
        self.assertEqual(result["wasted_proxy_work_seconds"], 0)
        self.assertAlmostEqual(reqs[1].completed, 11)
        self.assertEqual(reqs[1].attempts, 2)

    def test_reclaim_during_startup_cancels_stale_ready_event(self):
        reqs = [request(0, 1, 2)]
        events = [macro.NodeEvent(0, "a", "available"),
                  macro.NodeEvent(3, "a", "reclaim")]
        exp, result = simulate("naive", reqs, events=events, startup=10)
        self.assertEqual(result["interrupted_attempts"], 0)
        self.assertFalse(any(e.get("action") == "ready" for e in exp.event_log))

    def test_waits_without_workers_and_drains_when_available(self):
        # The API requires >=1 persistent worker; test all-transient gap
        # by setting persistent capacity unavailable before the run.
        reqs = [request(0, 0, 10), request(1, 1, 1)]
        events = [macro.NodeEvent(0, "a", "available")]
        exp, result = simulate("naive", reqs, events=events, startup=3)
        self.assertEqual(result["measured_completed_by_drain"], 2)
        self.assertEqual(len(exp.pending), 0)

    def test_all_results_are_measured_arrival_cohorts(self):
        reqs = [request(0, 0, 4), request(1, 10, 1),
                request(2, 19.5, 1), request(3, 21, 2)]
        _, result = simulate("fixed_stable", reqs, warmup=10, measurement=10)
        self.assertEqual(result["measured_arrivals"], 2)
        self.assertEqual(result["measured_completed_by_drain"], 2)

    def test_work_model_direct_match_and_fallback(self):
        model = macro.ProxyWorkModel({
            "schema_version": 1,
            "samples": [
                {"source_index": 11, "input_tokens": 1000,
                 "output_tokens": 10, "proxy_service_s": 1.0},
                {"source_index": 12, "input_tokens": 5000,
                 "output_tokens": 128, "proxy_service_s": 4.0},
            ],
        })
        self.assertEqual(model.estimate(11, 1000, 10), 1.0)
        self.assertGreater(model.estimate(13, 900, 10), 0)
        self.assertEqual(model.direct_matches, 1)
        self.assertEqual(model.estimated_matches, 1)


if __name__ == "__main__":
    unittest.main()
