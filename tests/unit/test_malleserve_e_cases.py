"""E3 failure accounting regression tests, using ground-truth event records."""
import unittest

from malleserve.tools.analyze_mooncake import summary


class FailureAccountingTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "warmup_seconds": 0,
            "measurement_seconds": 1,
            "warmup_count": 0,
            "measurement_count": 2,
            "rate_per_second": 2,
        }
        self.events = [
            {"event_type": "request_arrival", "request_id": 1, "arrived_at": 0.0},
            {"event_type": "request_arrival", "request_id": 2, "arrived_at": 0.5},
            {"event_type": "request_completion", "request_id": 1,
             "completed_at": 0.2, "request_waiting_time_total_s": 0, "ttft_s": 0.1},
            {"event_type": "request_failure", "request_id": 2,
             "failed_at": 0.6, "reason": "transient_node_reclaimed"},
        ]

    def test_failure_is_not_completion_or_backlog(self):
        result = summary(self.events, self.manifest, slo_seconds=30)
        self.assertEqual(result["measured_arrivals"], 2)
        self.assertEqual(result["measured_completed_eventually"], 1)
        self.assertEqual(result["measured_failed"], 1)
        self.assertEqual(result["measured_unresolved"], 0)
        self.assertEqual(result["measured_still_outstanding_at_window_end"], 0)
        self.assertEqual(result["all_request_backlog_at_window_end"], 0)
        self.assertEqual(result["measured_success_fraction"], 0.5)
        self.assertEqual(result["measured_failure_fraction"], 0.5)
        self.assertEqual(result["measured_slo_attainment"], 0.5)

    def test_duplicate_terminal_outcome_is_rejected(self):
        events = self.events + [
            {"event_type": "request_completion", "request_id": 2,
             "completed_at": 0.9},
        ]
        with self.assertRaisesRegex(ValueError, "both completion and terminal failure"):
            summary(events, self.manifest, slo_seconds=30)

    def test_unresolved_not_mislabeled_failure(self):
        events = self.events[:-1]
        result = summary(events, self.manifest, slo_seconds=30)
        self.assertEqual(result["measured_failed"], 0)
        self.assertEqual(result["measured_unresolved"], 1)
        self.assertEqual(result["measured_still_outstanding_at_window_end"], 1)


if __name__ == "__main__":
    unittest.main()
