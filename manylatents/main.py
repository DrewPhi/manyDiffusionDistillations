"""CLI entry point for manylatents experiments.

Usage:
    python -m manylatents.main data=swissroll algorithms/latent=pca
    python -m manylatents.main -m experiment=spectral_discrimination cluster=mila resources=cpu_light
"""

from typing import Dict, Any

import hydra
from omegaconf import DictConfig

import manylatents.configs  # noqa: F401 — registers SearchPathPlugin on import
from manylatents.experiment import run_algorithm, run_pipeline
from manylatents.pipeline import run_stage_pipeline

# Auto-discover extension plugins (shop, omics, etc.)
from manylatents.extensions import discover_extensions
discover_extensions()


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: DictConfig) -> Dict[str, Any]:
    has_stage_pipeline = cfg.get("stage_pipeline") is not None
    has_legacy_pipeline = cfg.get("pipeline") is not None and len(cfg.pipeline) > 0

    if has_stage_pipeline:
        return run_stage_pipeline(cfg)
    if has_legacy_pipeline:
        return run_pipeline(cfg)
    return run_algorithm(cfg)


if __name__ == "__main__":
    main()
