"""CPU-only scenario configuration tests; no Frontier profiling data needed."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/malleserve/transient/run_native_scenario.py"
spec = importlib.util.spec_from_file_location("native_scenario", SCRIPT)
scenario = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(scenario)
PRESETS = ROOT / "examples/malleserve/transient/scenarios"


class ScenarioTests(unittest.TestCase):
    def test_fixed_four_and_eight(self):
        for fname, count in (("fixed4.yaml", 4), ("fixed8.yaml", 8)):
            config = scenario.load_scenario(PRESETS / fname)
            self.assertEqual(config["resources"]["gpus"], count)
            plan = scenario.run_scenario(config, dry_run=True)
            self.assertEqual(plan["mode"], "fixed")
            self.assertIsNone(plan["schedule_command"])
            self.assertEqual(plan["resource_environment"]["FIXED_REPLICAS"], str(count))

    def test_transient(self):
        config = scenario.load_scenario(PRESETS / "naive.yaml")
        self.assertEqual(config["experiment"]["policy"], "naive")
        plan = scenario.run_scenario(config, dry_run=True)
        self.assertEqual(plan["resource_environment"]["TRANSIENT_STARTUP_SECONDS"], "40")
        self.assertIsNotNone(plan["schedule_command"])

    def test_identical_trace_for_parity(self):
        traces = {
            scenario.load_scenario(PRESETS / file)["workload"]["trace"]
            for file in ("fixed4.yaml", "fixed8.yaml", "naive.yaml")
        }
        self.assertEqual(len(traces), 1)

    def test_extends_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.yaml"
            second = Path(tmp) / "b.yaml"
            first.write_text("extends: b.yaml\n", encoding="utf-8")
            second.write_text("extends: a.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Circular"):
                scenario.read_config(first)

    def test_invalid_mode_policy_combination(self):
        config = scenario.load_scenario(PRESETS / "fixed4.yaml")
        config["experiment"]["policy"] = "naive"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "policy"):
                scenario.load_scenario(path)


if __name__ == "__main__":
    unittest.main()
