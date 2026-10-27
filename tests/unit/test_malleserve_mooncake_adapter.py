"""Pure-stdlib checks for the Mooncake -> Frontier bridge (no GPU needed)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


BASE = Path(__file__).resolve().parents[2] / "examples" / "malleserve"


def module_from_path(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conv = module_from_path("moon_convert", "convert_mooncake.py")
ana = module_from_path("moon_analyze", "analyze_mooncake.py")


class MooncakeBridgeTests(unittest.TestCase):
    @staticmethod
    def dataset():
        return {
            "schema_version": 1,
            "block_size": 512,
            "workload_name": "test",
            "requests": [
                {
                    "request_id": f"t-{i}",
                    "trace_index": i,
                    "input_length": 528,
                    "output_length": 500 if i == 0 else 4,
                    "hash_ids": ["shared-first-block", f"tail-{i}"],
                }
                for i in range(10)
            ],
        }

    def test_source_block_becomes_16_token_hashes_with_shared_identity(self):
        records = self.dataset()["requests"]
        ids0 = conv.frontier_block_ids(records[0], 512, 16)
        ids1 = conv.frontier_block_ids(records[1], 512, 16)
        self.assertEqual(len(ids0), 33)
        self.assertEqual(ids0[:32], ids1[:32])
        self.assertNotEqual(ids0[32], ids1[32])
        self.assertEqual(len(set(ids0)), 33)
        self.assertTrue(all(0 <= x < (1 << 63) for x in ids0))

    def test_same_k6_warmup_gap_and_measured_slice(self):
        rows, manifest = conv.convert(
            self.dataset(), rate=2, warmup_seconds=1, measurement_seconds=2,
            output_cap=128, frontier_block_size=16,
        )
        self.assertEqual([r["source_index"] for r in rows], [0, 1, 4, 5, 6, 7])
        self.assertEqual([r["arrived_at"] for r in rows],
                         ["0.000000000", "0.500000000", "1.000000000",
                          "1.500000000", "2.000000000", "2.500000000"])
        self.assertEqual(rows[0]["num_decode_tokens"], 128)
        self.assertEqual(rows[2]["num_decode_tokens"], 4)
        self.assertEqual(manifest["required_frontier_max_tokens"], 656)
        self.assertEqual(manifest["measurement_count"], 4)

    def test_backlog_throughput_and_slo_are_distinct(self):
        manifest = {
            "rate_per_second": 2,
            "warmup_seconds": 1,
            "measurement_seconds": 2,
            "warmup_count": 2,
            "measurement_count": 4,
        }
        events = []
        for rid, arrival in enumerate([0, .5, 1, 1.5, 2, 2.5]):
            events.append({"event_type": "request_arrival",
                           "request_id": rid, "arrived_at": arrival})
        for rid, arrival, finish in [
            (0, 0, .2), (1, .5, .8), (2, 1, 1.2),
            (3, 1.5, 1.7), (4, 2, 3.5), (5, 2.5, 4),
        ]:
            events.append({"event_type": "request_completion",
                           "request_id": rid, "arrived_at": arrival,
                           "completed_at": finish,
                           "request_waiting_time_total_s": .2,
                           "ttft_s": .1})
        result = ana.summary(events, manifest, slo_seconds=1.0)
        self.assertEqual(result["measured_arrivals"], 4)
        self.assertEqual(result["measured_completed_inside_window"], 2)
        self.assertEqual(result["measured_completions_during_drain"], 2)
        self.assertEqual(result["all_request_backlog_at_window_end"], 2)
        self.assertEqual(result["measured_slo_successes_allowing_drain"], 2)
        self.assertEqual(result["measured_completion_throughput_in_window_rps"], 1.0)
        self.assertEqual(result["measured_slo_attainment"], .5)

    def test_dataset_exhaustion_fails_before_replay(self):
        with self.assertRaisesRegex(ValueError, "Not enough prepared requests"):
            conv.convert(self.dataset(), rate=4, warmup_seconds=30,
                         measurement_seconds=300, output_cap=128,
                         frontier_block_size=16)


if __name__ == "__main__":
    unittest.main()
