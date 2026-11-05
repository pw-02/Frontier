"""Canonical E1–E4 Hydra and lifecycle unit smoke tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from malleserve.run_state import experiment_output
from malleserve.scenario import resolve_routing_policy
from malleserve.tools.generate_cluster_availability import generate
from malleserve.tools.summarize_e1_e4 import PATTERN
from frontier.transient_lifecycle import read_availability

CONF = Path(__file__).resolve().parents[2] / "malleserve/conf"


def loaded(overrides=None):
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        value = OmegaConf.to_container(
            compose(config_name="config", overrides=overrides or []), resolve=True
        )
    assert isinstance(value, dict)
    return value


class CanonicalCaseTests(unittest.TestCase):
    def test_default_is_e4(self):
        cfg = loaded()
        self.assertEqual(cfg["resources"]["label"], "e4_immediate_full_restart")
        self.assertEqual(cfg["resources"]["recovery_mode"], "full_restart")
        self.assertEqual(cfg["resources"]["candidate_nodes"], 16)
        self.assertEqual(cfg["workload"]["rate"], 34)

    def test_all_cases(self):
        expected = {
            "e1": ("fixed", "fixed", None),
            "e2": ("transient", "transient", "full_restart"),
            "e3": ("transient", "transient", "no_recovery"),
            "e4": ("transient", "transient", "full_restart"),
        }
        for case, (resource_mode, policy, recovery) in expected.items():
            with self.subTest(case=case):
                cfg = loaded([f"case={case}", "scheduler.routing_policy=cwa",
                              "experiment.group=e1_e4_hpc_r34",
                              f"experiment.name={case}_test"])
                self.assertEqual(cfg["resources"]["mode"], resource_mode)
                self.assertEqual(cfg["policy"]["name"], policy)
                if recovery:
                    self.assertEqual(cfg["resources"]["recovery_mode"], recovery)
                self.assertEqual(
                    experiment_output(cfg).parts[-3:],
                    ("e1_e4_hpc_r34", resource_mode, f"{case}_test_qwen_qwen3_8b"),
                )

    def test_routing_aliases(self):
        for alias, settings in {
            "rr": ("round_robin", 1.0),
            "wa": ("kv_aware", 0.0),
            "cwa": ("kv_aware", 1.0),
        }.items():
            cfg = loaded([f"scheduler.routing_policy={alias}"])
            self.assertEqual(resolve_routing_policy(cfg["scheduler"]), settings)
        with self.assertRaises(ValueError):
            resolve_routing_policy({"cluster": "round_robin", "routing_policy": "bad"})

    def test_large_availability_is_deterministic_and_valid(self):
        events = generate(duration=330, nodes=16, seed=7)
        self.assertEqual(events, generate(duration=330, nodes=16, seed=7))
        self.assertNotEqual(events, generate(duration=330, nodes=16, seed=19))
        self.assertEqual(len({x["node_id"] for x in events}), 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "availability.jsonl"
            path.write_text("".join(json.dumps(event) + "\n" for event in events),
                            encoding="utf-8")
            self.assertEqual(len(read_availability(str(path))), len(events))

    def test_summary_names(self):
        for name in (
            "e1_fixed8_rr_r34", "e1_fixed16_wa_r34",
            "e1_fixed32_cwa_r34", "e2_s7_rr_r34",
            "e3_s19_wa_r34", "e4_s41_cwa_r34",
        ):
            self.assertIsNotNone(PATTERN.fullmatch(name), name)

    def test_synthetic_availability_validation(self):
        for kwargs in (
            {"duration": -1}, {"nodes": 0},
            {"min_lifetime": 240, "max_lifetime": 90},
            {"initial_spread": 330},
        ):
            args = dict(duration=330, nodes=16)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                generate(**args)


if __name__ == "__main__":
    unittest.main()
