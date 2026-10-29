import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from malleserve.tools.decode_predictor_reconcile import build_batch, reconcile


class ReconcileTests(unittest.TestCase):
    def test_batch_explicit_context_vectors(self):
        batch = build_batch([2048, 4096], prompts=[512, 2048])
        self.assertEqual(batch.num_context_tokens, [2048, 4096])
        self.assertEqual(batch.num_tokens, [1, 1])
        self.assertEqual(batch.request_is_decoding, [True, True])

    def test_reports_mismatch_without_claiming_exactness(self):
        original = {
            "batch_id": 1, "context_tokens": [1024],
            "num_tokens": [1], "request_is_decoding": [True],
            "stage_id": 0, "layer_id": 0, "num_layers": 36,
            "original": {"model_ms": 20, "attention_ms": 9,
                "mlp_up_ms": 5, "mlp_down_ms": 3, "mlp_act_ms": 0.3,
                "host_ms": 0, "forward_launch_ms": 0},
        }
        class Predictor:
            def predict_stage_execution_time(self, batch, **kwargs):
                return SimpleNamespace(
                    model_time_ms=47.0, total_time=0.047,
                    attention_time=18.0, mlp_layer_up_proj_execution_time=10,
                    mlp_layer_down_proj_execution_time=6,
                    mlp_layer_act_execution_time=1,
                    forward_launch_time=0)
        with tempfile.TemporaryDirectory() as directory:
            captured = Path(directory) / "captured.jsonl"
            captured.write_text(json.dumps(original) + "\n")
            result = reconcile(Predictor(), captured)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["matched_within_tolerance"])
        self.assertEqual(result[0]["delta_ms"], 27.0)


if __name__ == "__main__":
    unittest.main()
