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
    def test_default_transient_baseline_recovery(self):
        cfg = loaded()
        self.assertEqual(cfg["policy"]["name"], "transient")
        self.assertEqual(cfg["resources"]["mode"], "transient")
        self.assertEqual(cfg["resources"]["persistent_gpus"], 4)
        self.assertEqual(cfg["resources"]["lifecycle_mode"], "baseline_recovery")
        self.assertEqual(cfg["experiment"]["name"], "transient_r10")

    def test_recovery_modes_compose(self):
        baseline = loaded(["resources.lifecycle_mode=baseline_recovery"])
        ideal = loaded(["resources.lifecycle_mode=ideal_recovery"])
        self.assertEqual(baseline["resources"]["lifecycle_mode"], "baseline_recovery")
        self.assertEqual(ideal["resources"]["lifecycle_mode"], "ideal_recovery")
        legacy = loaded(["resources=naive", "policy=naive"])
        self.assertEqual(legacy["resources"]["lifecycle_mode"], "naive")

    def test_transient_lifetime_sweep_scenario(self):
        from malleserve.tools.generate_native_schedule import generate
        cfg = loaded([
            "resources=transient", "policy=transient", "workload=mooncake_r12",
            "resources.node_lifetime_seconds=180",
            "resources.lifecycle_mode=ideal_recovery",
            "scheduler.routing_policy=cwa",
            "experiment.group=transient_lifetime_sweep_r12",
            "experiment.name=transient_l180_s40_ideal_recovery_cwa_r12",
        ])
        self.assertEqual(cfg["workload"]["rate"], 12)
        self.assertEqual(cfg["resources"]["node_lifetime_seconds"], 180)
        self.assertEqual(
            experiment_output(cfg).parts[-3:],
            ("transient_lifetime_sweep_r12", "transient",
             "transient_l180_s40_ideal_recovery_cwa_r12_qwen_qwen3_8b"),
        )
        events = generate(330, 180)
        self.assertEqual(
            [(e["timestamp"], e["action"]) for e in events],
            [(0.0, "available"), (180, "reclaim"),
             (180, "available"), (330, "reclaim")],
        )

    def test_transient_ready_gpu_exposure(self):
        import json
        import tempfile
        from malleserve.tools.generate_native_schedule import generate
        from malleserve.tools.summarize_transient_lifetime_sweep import ready_gpu_seconds
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "availability.jsonl"
            for lifetime, expected in ((60, 400.0), (180, 1000.0)):
                path.write_text(
                    "".join(json.dumps(e) + "\n" for e in generate(330, lifetime)),
                    encoding="utf-8",
                )
                self.assertEqual(ready_gpu_seconds(path, 40), expected)

    def test_reclamation_controls_compose(self):
        from malleserve.tools.summarize_transient_lifetime_sweep import PATTERN
        for routing in ("rr", "wa", "cwa"):
            no = f"transient_no_reclamation_s40_{routing}_r12"
            self.assertIsNotNone(PATTERN.fullmatch(no))
            no_cfg = loaded([
                "resources=transient", "policy=transient", "workload=mooncake_r12",
                "resources.lifecycle_mode=no_reclamation",
                f"scheduler.routing_policy={routing}",
                "experiment.group=transient_lifetime_sweep_r12",
                f"experiment.name={no}",
            ])
            self.assertEqual(no_cfg["resources"]["lifecycle_mode"], "no_reclamation")
            for recovery in ("baseline_recovery", "ideal_recovery"):
                name = f"transient_single_reclaim_t180_s40_{recovery}_{routing}_r12"
                self.assertIsNotNone(PATTERN.fullmatch(name))
                cfg = loaded([
                    "resources=transient", "policy=transient", "workload=mooncake_r12",
                    "resources.lifecycle_mode=single_reclaim",
                    "+resources.reclaim_at_seconds=180",
                    f"+resources.recovery_mode={recovery}",
                    f"scheduler.routing_policy={routing}",
                    "experiment.group=transient_lifetime_sweep_r12",
                    f"experiment.name={name}",
                ])
                self.assertEqual(cfg["resources"]["reclaim_at_seconds"], 180)
                self.assertEqual(cfg["resources"]["recovery_mode"], recovery)

    def test_fixed_resource_groups(self):
        for group, size in (("fixed3", 3), ("fixed4", 4), ("fixed8", 8)):
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
            ("runs", "transient", "transient_r10_qwen_qwen3_8b"),
        )

    def test_routing_aliases_and_grouped_paths(self):
        from malleserve.scenario import resolve_routing_policy
        expected = {
            "rr": ("round_robin", 1.0),
            "wa": ("kv_aware", 0.0),
            "cwa": ("kv_aware", 1.0),
        }
        for alias, native in expected.items():
            cfg = loaded([
                "resources=fixed4", "policy=fixed",
                f"scheduler.routing_policy={alias}",
                "experiment.group=routing_policy_sweep",
                f"experiment.name=fixed4_{alias}_r10",
            ])
            self.assertEqual(resolve_routing_policy(cfg["scheduler"]), native)
            self.assertEqual(
                experiment_output(cfg).parts[-3:],
                ("routing_policy_sweep", "fixed", f"fixed4_{alias}_r10_qwen_qwen3_8b"),
            )

    def test_alias_validation_and_legacy_routing(self):
        from malleserve.scenario import resolve_routing_policy
        self.assertEqual(resolve_routing_policy({"cluster": "round_robin"}), ("round_robin", 1.0))
        self.assertEqual(resolve_routing_policy({"cluster": "kv_aware", "cache_weight": 0.5}), ("kv_aware", 0.5))
        with self.assertRaises(ValueError):
            resolve_routing_policy({"routing_policy": "invalid", "cluster": "round_robin"})

    def test_kv_aware_routing_config(self):
        cfg = loaded([
            "resources=fixed4", "policy=fixed",
            "scheduler.cluster=kv_aware",
            "scheduler.load_weight=0.5",
            "scheduler.cache_weight=0.0",
            "scheduler.decode_token_weight=12.0",
        ])
        self.assertEqual(cfg["scheduler"]["cluster"], "kv_aware")
        self.assertEqual(cfg["scheduler"]["load_weight"], 0.5)
        self.assertEqual(cfg["scheduler"]["cache_weight"], 0.0)
        self.assertEqual(cfg["scheduler"]["decode_token_weight"], 12.0)

    def test_composable_overrides(self):
        cfg = loaded(["workload=mooncake_r2", "resources.startup_seconds=0",
                      "scheduler.max_tokens_in_batch=1024"])
        self.assertEqual(cfg["workload"]["trace"], "data/malleserve/mooncake_r2_330s.csv")
        self.assertEqual(cfg["resources"]["startup_seconds"], 0)
        self.assertEqual(cfg["scheduler"]["max_tokens_in_batch"], 1024)


if __name__ == "__main__":
    unittest.main()
