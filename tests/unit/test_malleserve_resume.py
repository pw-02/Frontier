"""Unit tests for automatic trace preparation and fingerprinted run resume."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from malleserve import run_state


class ResumeTests(unittest.TestCase):
    def test_completion_requires_files_and_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self.assertFalse(run_state.is_complete(out, "abc"))
            (out / "malleserve_summary.json").write_text("{}", encoding="utf-8")
            (out / "metrics_ground_truth.jsonl").write_text("{}\n", encoding="utf-8")
            run_state.mark_complete(out, "abc")
            self.assertTrue(run_state.is_complete(out, "abc"))
            self.assertFalse(run_state.is_complete(out, "different"))

    def test_generate_and_reuse_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            dataset = base / "prepared.json"
            dataset.write_text(json.dumps({
                "schema_version": 1, "block_size": 512,
                "requests": [{
                    "input_length": 512, "output_length": 4,
                    "hash_ids": ["block"]
                } for _ in range(12)],
            }), encoding="utf-8")
            trace = base / "trace.csv"
            workload = {
                "trace": str(trace), "dataset": str(dataset), "rate": 2,
                "warmup_seconds": 1, "measurement_seconds": 2,
                "max_output_tokens": 128,
            }
            path, manifest = run_state.ensure_trace(workload, 16)
            self.assertTrue(path.is_file())
            self.assertTrue(manifest.is_file())
            first = path.read_bytes()
            dataset.unlink()
            path2, _ = run_state.ensure_trace(workload, 16)
            self.assertEqual(path2.read_bytes(), first)
            with self.assertRaisesRegex(ValueError, "rate"):
                run_state.ensure_trace(dict(workload, rate=4), 16)

    def test_trace_fingerprint_changes_on_config(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            trace, manifest = base / "trace.csv", base / "trace.manifest.json"
            trace.write_text("x\n", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            a = run_state.fingerprint({"resources": {"gpus": 4}}, trace, manifest)
            b = run_state.fingerprint({"resources": {"gpus": 8}}, trace, manifest)
            self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
