import csv
import json
import tempfile
import unittest
from pathlib import Path
from malleserve.tools.decode_predictor_diagnostic import compare


class DecodeDiagnosticTests(unittest.TestCase):
    def test_missing_context_is_never_claimed_as_exact_match(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ledger = root / "ledger.jsonl"
            ledger.write_text(json.dumps({
                "cluster_type": "MONOLITHIC", "batch_id": 1,
                "request_num_tokens": [1, 1],
                "request_num_prefill_tokens": [128, 128],
                "execution_time": {"model_time_ms": 19, "total_time_ms": 21,
                                   "component_ledger_ms": {"schedule_time": 2}},
            }) + "\n")
            polaris = root / "summary.csv"
            with polaris.open("w", newline="") as h:
                w = csv.writer(h)
                w.writerow(["batch_size", "prompt_len", "median_wall_ms_per_decode_step"])
                for b in (1, 2, 4, 8, 16):
                    for c in (128, 1024, 4096):
                        w.writerow([b, c, 14])
            rows = compare(ledger, polaris, root / "out.csv")
            self.assertEqual(len(rows), 15)
            self.assertTrue(all(r["frontier_exact_grid_model_ms"] is None for r in rows))
            row = next(r for r in rows if r["batch_size"] == 2)
            self.assertEqual(row["frontier_observed_batch_model_median_ms"], 19)
            self.assertEqual(row["frontier_modelled_host_component_median_ms"], 2)


if __name__ == "__main__":
    unittest.main()
