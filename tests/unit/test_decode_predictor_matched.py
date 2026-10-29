import unittest
from types import SimpleNamespace
from malleserve.tools.decode_predictor_matched import BATCH_SIZES, CONTEXTS, probe


class MatchedDecodeTests(unittest.TestCase):
    def test_all_fifteen_cells_and_host_separation(self):
        class Predictor:
            def predict_stage_execution_time(self, batch, **kwargs):
                self_batch = batch
                assert self_batch.size in BATCH_SIZES
                assert self_batch.num_tokens == [1] * self_batch.size
                assert len(set(self_batch.num_context_tokens)) == 1
                return SimpleNamespace(model_time_ms=14, total_time=0.016)
        polaris = {(b, c): 13.0 for b in BATCH_SIZES for c in CONTEXTS}
        rows = probe(Predictor(), 36, polaris)
        self.assertEqual(len(rows), 15)
        self.assertEqual({(r["batch_size"], r["context_tokens"]) for r in rows}, set(polaris))
        self.assertTrue(all(r["frontier_modelled_host_ms"] == 2.0 for r in rows))


if __name__ == "__main__":
    unittest.main()
