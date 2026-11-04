"""Check LOR + prefix caching without relaxing trace validation."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from frontier.scheduler.utils.prefix_cache import validate_prefix_cache_config
from frontier.types import (
    ClusterSchedulerType,
    ClusterType,
    ReplicaSchedulerType,
    RequestGeneratorType,
)


class Config:
    def __init__(self, kind):
        self.kind = kind

    def get_type(self):
        return self.kind


class PrefixCacheLORTests(unittest.TestCase):
    def validate(self, scheduler, source):
        return validate_prefix_cache_config(
            replica_scheduler_config=SimpleNamespace(
                enable_prefix_caching=True,
                get_type=lambda: ReplicaSchedulerType.VLLM_V1,
            ),
            cluster_type=ClusterType.MONOLITHIC,
            num_replicas=4,
            cluster_scheduler_config=Config(scheduler),
            request_generator_config=source,
        )

    def test_lor_accepts_valid_prefix_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.csv"
            with trace.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["session_id", "block_hash_ids"])
                writer.writeheader()
                writer.writerow({"session_id": "1", "block_hash_ids": "[1, 2]"})
            source = SimpleNamespace(
                get_type=lambda: RequestGeneratorType.TRACE_REPLAY,
                trace_file=str(trace),
            )
            self.assertIsNone(self.validate(ClusterSchedulerType.LOR, source))
            self.assertIsNone(self.validate(ClusterSchedulerType.ROUND_ROBIN, source))
            with self.assertRaisesRegex(ValueError, "Multi-replica prefix caching"):
                self.validate(ClusterSchedulerType.RANDOM, source)

    def test_lor_still_requires_trace_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.csv"
            trace.write_text("session_id\n1\n", encoding="utf-8")
            source = SimpleNamespace(
                get_type=lambda: RequestGeneratorType.TRACE_REPLAY,
                trace_file=str(trace),
            )
            with self.assertRaisesRegex(ValueError, "Missing columns"):
                self.validate(ClusterSchedulerType.LOR, source)


if __name__ == "__main__":
    unittest.main()
