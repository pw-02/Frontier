"""Unit tests for fixed-capacity latency diagnostic, no Frontier GPU required."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from malleserve.tools.analyze_fixed_latency import load, grouped, token_bin


class FixedLatencyDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.trace = self.root / "trace.csv"
        self.events = self.root / "metrics_ground_truth.jsonl"

    def fixture(self, *, duplicate=False):
        rows = [
            {"arrived_at": "0.000000000", "num_prefill_tokens": "128", "num_decode_tokens": "1"},
            {"arrived_at": "30.000000000", "num_prefill_tokens": "1024", "num_decode_tokens": "128"},
            {"arrived_at": "30.100000000", "num_prefill_tokens": "256", "num_decode_tokens": "32"},
        ]
        if duplicate:
            rows[2]["arrived_at"] = rows[1]["arrived_at"]
        with self.trace.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        events = [
            {"event_type": "request_arrival", "request_id": "a", "arrived_at": 0.},
            {"event_type": "request_completion", "request_id": "a", "completed_at": 1.},
            {"event_type": "request_arrival", "request_id": "b", "arrived_at": 30.},
            {"event_type": "request_completion", "request_id": "b", "completed_at": 40.,
             "ttft_s": 2., "request_waiting_time_total_s": 1.},
            {"event_type": "request_arrival", "request_id": "c", "arrived_at": 30.1},
            {"event_type": "request_completion", "request_id": "c", "completed_at": 35.1,
             "ttft_s": 1., "request_waiting_time_total_s": .2},
        ]
        self.events.write_text("".join(json.dumps(ev) + "\n" for ev in events))

    def test_join_bins_window(self):
        self.fixture()
        records, diag = load(self.root, self.trace, 30., 330.)
        self.assertEqual(2, len(records))
        self.assertEqual(0, diag["measured_missing_completions"])
        lookup = {r["output_tokens"]: r for r in records}
        self.assertAlmostEqual(10., lookup[128]["latency_s"])
        self.assertAlmostEqual(8., lookup[128]["post_first_token_s"])
        self.assertEqual("128+", token_bin(128))
        bins, windows = grouped(records, 20., 30., 50.)
        self.assertEqual(2, sum(x["n"] for x in bins))
        self.assertEqual(2, windows[0]["completed"])

    def test_missing_metric_does_not_default_zero(self):
        self.fixture()
        data = [json.loads(line) for line in self.events.read_text().splitlines()]
        for ev in data:
            if ev.get("request_id") == "b" and ev.get("event_type") == "request_completion":
                ev.pop("ttft_s")
        self.events.write_text("".join(json.dumps(ev) + "\n" for ev in data))
        records, diag = load(self.root, self.trace, 30., 330.)
        self.assertEqual(1, diag["missing_ttft_field"])
        self.assertIsNone(next(r for r in records if r["output_tokens"] == 128)["ttft_s"])

    def test_duplicate_scheduled_arrival_rejected(self):
        self.fixture(duplicate=True)
        with self.assertRaisesRegex(ValueError, "collision"):
            load(self.root, self.trace, 30., 330.)


if __name__ == "__main__":
    unittest.main()
