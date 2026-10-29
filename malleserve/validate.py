"""Hydra entry point for Polaris-versus-Frontier validation.

python -m malleserve.validate validation.polaris=data/malleserve/validation/polaris_summary.csv
"""
from pathlib import Path

import hydra
from omegaconf import DictConfig

from malleserve.validation import compare


@hydra.main(version_base=None, config_path="conf", config_name="validate")
def main(cfg: DictConfig) -> None:
    spec = cfg.validation
    rows = compare(Path(spec.polaris), Path(spec.frontier), Path(spec.output))
    print(f"Validated {len(rows)} Polaris cells; wrote {spec.output}")
    for row in rows:
        print(
            f"GPUs={row['workers']}, RPS={row['offered_rps']}, "
            f"valid_load={row['polaris_valid_load']}, "
            f"Polaris p95={row.get('latency_p95_s')}s, "
            f"Frontier p95={row['frontier_p95_engine_latency_s']}"
        )


if __name__ == "__main__":
    main()
