"""Polaris summary CSV comparison tests, no simulator execution required."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from malleserve.validation import compare


class SummaryCSVTests(unittest.TestCase):
    def test_repeats_and_missing_dropped_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            polaris = base / "polaris.csv"
            headers = [
                "workers", "offered_rps", "run_id", "expected_measured",
                "submitted_measured", "submission_pct", "completed_ok",
                "failed", "dropped_iterations", "latency_mean_s",
                "latency_p50_s", "latency_p95_s", "latency_p99_s",
                "slo_attainment_pct", "slo_goodput_rps",
                "completion_in_window_rps_est", "measured_outstanding_at_end_est",
                "prefix_cache_hit_pct", "validity_status", "run_dir",
            ]
            baseline = dict.fromkeys(headers, "")
            baseline.update(workers="6", offered_rps="12.0",
                            expected_measured="3600", submitted_measured="3601",
                            submission_pct="100.028", completed_ok="3601",
                            failed="0", dropped_iterations="0",
                            latency_mean_s="1.686", latency_p50_s="0.638",
                            latency_p95_s="5.731", latency_p99_s="7.445",
                            slo_attainment_pct="100", slo_goodput_rps="12.003",
                            completion_in_window_rps_est="11.977",
                            measured_outstanding_at_end_est="8",
                            prefix_cache_hit_pct="58.08",
                            validity_status="completed")
            copies = [dict(baseline, run_id="first"),
                      dict(baseline, run_id="repeat", latency_p95_s="5.700"),
                      dict(baseline, run_id="failed", validity_status="failed",
                           dropped_iterations="")]
            with polaris.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(copies)
            out = base / "comparison.csv"
            rows = compare(polaris, base / "simulations", out)
            self.assertEqual(len(rows), 3)
            self.assertEqual([r["polaris_valid_load"] for r in rows], [False, True, True])
            self.assertTrue(all(not r["frontier_found"] for r in rows))
            self.assertTrue(out.exists())
            self.assertEqual(rows[1]["latency_p95_s"], 5.731)


if __name__ == "__main__":
    unittest.main()
