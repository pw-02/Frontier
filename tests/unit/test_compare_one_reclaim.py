import unittest
from malleserve.tools.compare_one_reclaim import frontier_records, window_rows


class CompareOneReclaimTest(unittest.TestCase):
    def test_counts_unfinished_and_late_completions(self):
        events = [
            {"event_type": "request_arrival", "request_id": 1, "arrived_at": 210},
            {"event_type": "request_arrival", "request_id": 2, "arrived_at": 211},
            {"event_type": "request_completion", "request_id": 1, "completed_at": 300},
        ]
        records = frontier_records(events, 0)
        row = window_rows(records, "frontier")[1]
        self.assertEqual(row["arrived"], 2)
        self.assertEqual(row["completed_eventually"], 1)
        self.assertEqual(row["no_completion_event"], 1)
        self.assertEqual(row["completed_after_window_end"], 1)
        self.assertEqual(row["p95_s"], 90)

    def test_exact_order_statistic(self):
        records = [{"arrival_s": 1, "latency_s": i+1,
                    "completed_at_s": i+2} for i in range(1501)]
        result = window_rows(records, "frontier")[0]
        self.assertEqual(result["n"], 1501)
        self.assertEqual(result["p50_s"], 751)

    def test_unknown_completion_rejected(self):
        with self.assertRaisesRegex(ValueError, "without arrivals"):
            frontier_records([{"event_type": "request_completion",
                               "request_id": 1, "completed_at": 1}], 0)


if __name__ == "__main__":
    unittest.main()
