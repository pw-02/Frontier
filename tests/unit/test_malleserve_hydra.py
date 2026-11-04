"""Hydra config composition smoke checks (no Frontier simulation required)."""
from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from malleserve.run_state import experiment_output


CONF = Path(__file__).resolve().parents[2] / "malleserve/conf"


def loaded(overrides: list[str] | None = None) -> dict:
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        cfg = compose(config_name="config", overrides=overrides or [])
        result = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(result, dict)
        return result


class HydraScenarioTests(unittest.TestCase):
    def test_default_naive(self):
        cfg = loaded()
        self.assertEqual(cfg["policy"]["name"], "naive")
        self.assertEqual(cfg["resources"]["mode"], "transient")
        self.assertEqual(cfg["resources"]["persistent_gpus"], 4)
        self.assertEqual(cfg["experiment"]["name"], "naive_r10")

    def test_fixed_resource_groups(self):
        for group, size in (("fixed4", 4), ("fixed8", 8)):
            cfg = loaded([f"resources={group}", "policy=fixed"])
            self.assertEqual(cfg["resources"]["gpus"], size)
            self.assertEqual(cfg["resources"]["mode"], "fixed")
            self.assertEqual(cfg["experiment"]["name"], f"{group}_r10")

    def test_organized_output_paths(self):
        fixed = loaded(["resources=fixed4", "policy=fixed", "workload=mooncake_r12"])
        self.assertEqual(
            experiment_output(fixed).parts[-3:],
            ("runs", "fixed", "fixed4_r12_qwen_qwen3_8b"),
        )
        transient = loaded()
        self.assertEqual(
            experiment_output(transient).parts[-3:],
            ("runs", "transient", "naive_r10_qwen_qwen3_8b"),
        )

    def test_kv_aware_routing_config(self):
        cfg = loaded([
            "resources=fixed4", "policy=fixed",
            "scheduler.cluster=kv_aware",
            "scheduler.load_weight=0.5",
            "scheduler.decode_token_weight=12.0",
        ])
        self.assertEqual(cfg["scheduler"]["cluster"], "kv_aware")
        self.assertEqual(cfg["scheduler"]["load_weight"], 0.5)
        self.assertEqual(cfg["scheduler"]["decode_token_weight"], 12.0)

    def test_composable_overrides(self):
        cfg = loaded(["workload=mooncake_r2", "resources.startup_seconds=0",
                      "scheduler.max_tokens_in_batch=1024"])
        self.assertEqual(cfg["workload"]["trace"], "data/malleserve/mooncake_r2_330s.csv")
        self.assertEqual(cfg["resources"]["startup_seconds"], 0)
        self.assertEqual(cfg["scheduler"]["max_tokens_in_batch"], 1024)


if __name__ == "__main__":
    unittest.main()
