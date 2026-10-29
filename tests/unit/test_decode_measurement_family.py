import unittest
from types import SimpleNamespace
from malleserve.tools.decode_predictor_reconcile import select_live_decode_family
from frontier.types import MeasurementType


class MeasurementFamilyTest(unittest.TestCase):
    def test_kernel_only_accepted(self):
        class Predictor:
            def _activate_measurement_type_for_batch(self, batch):
                return MeasurementType.KERNEL_ONLY
            def _measurement_family_name(self, kind):
                return "kernel_only"
        self.assertEqual(select_live_decode_family(Predictor(), object()), "kernel_only")

    def test_eager_rejected(self):
        class Predictor:
            def _activate_measurement_type_for_batch(self, batch):
                return MeasurementType.CUDA_EVENT
            def _measurement_family_name(self, kind):
                return "eager"
        with self.assertRaisesRegex(RuntimeError, "expected KERNEL_ONLY"):
            select_live_decode_family(Predictor(), object())


if __name__ == "__main__":
    unittest.main()
