from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from frontier.scheduler.utils.decode_prediction_replay import maybe_probe, _seen


class Timing:
    model_time_ms = 20.0
    total_time = 0.021
    attention_time = 8.0
    mlp_layer_up_proj_execution_time = 3.0
    mlp_layer_down_proj_execution_time = 2.0
    mlp_layer_act_execution_time = 1.0
    forward_launch_time = 0.0


class ReplayTest(unittest.TestCase):
    def test_identical_batch_and_context_are_replayed(self):
        class Predictor:
            def __init__(self):
                self.calls = []

            def predict_stage_execution_time(self, batch, stage_id, **kwargs):
                self.calls.append((batch, stage_id, kwargs))
                return Timing()

        batch = SimpleNamespace(
            id=7,
            num_tokens=[1, 1],
            num_context_tokens=[1024, 2048],
            request_is_decoding=[True, True],
        )
        predictor = Predictor()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe.jsonl"
            _seen.clear()
            with patch.dict("os.environ", {"FRONTIER_DECODE_REPLAY_PATH": str(output)}):
                maybe_probe(batch, Timing(), predictor,
                            stage_id=0,
                            cluster_type=SimpleNamespace(name="MONOLITHIC"),
                            num_layers=36, layer_id=0)
                maybe_probe(batch, Timing(), predictor,
                            stage_id=0,
                            cluster_type=SimpleNamespace(name="MONOLITHIC"),
                            num_layers=36, layer_id=0)
            self.assertEqual(len(predictor.calls), 1)
            self.assertIs(predictor.calls[0][0], batch)
            row = json.loads(output.read_text().strip())
            self.assertEqual(row["context_tokens"], [1024, 2048])
            self.assertEqual(row["delta_model_ms"], 0.0)
            self.assertAlmostEqual(row["original"]["host_ms"], 1.0)


if __name__ == "__main__":
    unittest.main()
