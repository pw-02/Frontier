import unittest
from malleserve.tools.decode_gap_analysis import analyze, fit_summary


class DecodeGapAnalysisTest(unittest.TestCase):
    def test_components_and_fit(self):
        row = {"batch_size":"1","context_tokens":"128",
               "measurement_family":"kernel_only","frontier_model_ms":"20",
               "polaris_engine_wall_ms":"13","frontier_attention_ms":"10",
               "frontier_mlp_up_ms":"5","frontier_mlp_down_ms":"3",
               "frontier_mlp_act_ms":"1","frontier_other_model_ms":"1"}
        other = {**row, "batch_size":"2","frontier_model_ms":"22",
                 "polaris_engine_wall_ms":"14","frontier_attention_ms":"12"}
        rows = analyze([row, other])
        self.assertEqual(rows[0]["gap_ms"], 7)
        self.assertAlmostEqual(fit_summary(rows)["descriptive_affine_wall_from_model"]["slope"], .5)

    def test_rejects_eager(self):
        with self.assertRaisesRegex(ValueError,"measurement families"):
            analyze([{"measurement_family":"eager", **{k:"1" for k in
                ("batch_size","context_tokens","frontier_model_ms",
                 "polaris_engine_wall_ms","frontier_attention_ms",
                 "frontier_mlp_up_ms","frontier_mlp_down_ms",
                 "frontier_mlp_act_ms","frontier_other_model_ms")}}])


if __name__ == "__main__":
    unittest.main()
