"""DAG-style pipeline runner for config-native staged experiments."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import hydra
from omegaconf import DictConfig, OmegaConf

from manylatents.experiment import should_disable_wandb
from manylatents.pipeline.artifacts import ArtifactManifest, ArtifactRegistry
from manylatents.pipeline.stages.base import PipelineStage, StageContext

try:
    import wandb
    wandb.init
except (ImportError, AttributeError):
    wandb = None


def _wandb_init_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    logger_cfg = cfg.get("logger")
    kwargs: Dict[str, Any] = {
        "project": logger_cfg.get("project", cfg.get("project", "manylatents")),
        "name": logger_cfg.get("name", cfg.get("name", "pipeline_run")),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mode": logger_cfg.get("mode", "online"),
    }
    entity = logger_cfg.get("entity")
    if entity:
        kwargs["entity"] = entity
    return kwargs

logger = logging.getLogger(__name__)


def _topological_order(stage_specs: List[DictConfig]) -> List[str]:
    name_to_deps: Dict[str, Set[str]] = {}
    for spec in stage_specs:
        name = str(spec.name)
        deps = set(spec.get("depends_on", []) or [])
        name_to_deps[name] = deps

    unknown = set().union(*name_to_deps.values()) - set(name_to_deps.keys()) if name_to_deps else set()
    if unknown:
        raise ValueError(f"Pipeline depends_on references unknown stages: {sorted(unknown)}")

    ordered: List[str] = []
    remaining = dict(name_to_deps)

    while remaining:
        ready = sorted([name for name, deps in remaining.items() if deps.issubset(set(ordered))])
        if not ready:
            raise ValueError("Pipeline stage graph has a cycle or unresolved dependency")
        ordered.extend(ready)
        for name in ready:
            remaining.pop(name)

    return ordered


def run_stage_pipeline(cfg: DictConfig) -> Dict[str, Any]:
    """Execute configured stage pipeline with manifested artifacts."""
    stage_pipeline_cfg = cfg.get("stage_pipeline")
    if stage_pipeline_cfg is None:
        raise ValueError("No stage_pipeline configuration found")

    stages = stage_pipeline_cfg.get("stages") or []
    if not stages:
        raise ValueError("stage_pipeline.stages is empty")

    run_id = str(stage_pipeline_cfg.get("run_id") or cfg.get("name") or "pipeline_run")
    output_root = Path(str(stage_pipeline_cfg.get("output_root") or Path(cfg.output_dir) / "pipelines"))
    resume = bool(stage_pipeline_cfg.get("resume", True))

    registry = ArtifactRegistry(output_root=output_root, run_id=run_id)
    registry.run_dir.mkdir(parents=True, exist_ok=True)

    stage_by_name = {str(s.name): s for s in stages}
    order = _topological_order(stages)

    context = StageContext(run_id=run_id, run_dir=registry.run_dir, cfg=cfg)
    executed = []
    skipped = []

    wandb_disabled = should_disable_wandb(cfg) or wandb is None
    owns_wandb_run = False
    if not wandb_disabled and wandb is not None and wandb.run is None and cfg.logger is not None:
        logger.info("Initializing wandb for stage pipeline run_id=%s", run_id)
        wandb.init(**_wandb_init_kwargs(cfg))
        owns_wandb_run = True

    try:
        logger.info("Running stage pipeline run_id=%s with %d stage(s)", run_id, len(order))

        for stage_name in order:
            spec = stage_by_name[stage_name]
            stage_dir = registry.stage_dir(stage_name)

            if resume and registry.has_manifest(stage_name):
                manifest = registry.load_manifest(stage_name)
                context.artifacts[stage_name] = manifest.outputs
                skipped.append(stage_name)
                logger.info("Skipping stage '%s' (manifest exists)", stage_name)
                continue

            instantiate_cfg = {
                "_target_": str(spec.target),
                "stage_name": stage_name,
            }
            params = spec.get("params")
            if params:
                instantiate_cfg.update(OmegaConf.to_container(params, resolve=True))

            stage = hydra.utils.instantiate(instantiate_cfg)
            if not isinstance(stage, PipelineStage):
                raise TypeError(f"Stage '{stage_name}' does not inherit PipelineStage")

            result = stage.run(context=context, stage_dir=stage_dir)
            context.artifacts[stage_name] = result.outputs

            manifest = ArtifactManifest.create(
                run_id=run_id,
                stage_name=stage_name,
                stage_type=type(stage).__name__,
                config=spec,
                inputs={dep: context.artifacts.get(dep, {}) for dep in spec.get("depends_on", []) or []},
                outputs=result.outputs,
                metadata=result.metadata,
                status="completed",
            )
            registry.save_manifest(manifest)
            executed.append(stage_name)
            logger.info("Completed stage '%s'", stage_name)

        manifests = {name: str(registry.manifest_path(name)) for name in order}
        return {
            "run_id": run_id,
            "run_dir": str(registry.run_dir),
            "stage_order": order,
            "executed_stages": executed,
            "skipped_stages": skipped,
            "manifests": manifests,
            "artifacts": context.artifacts,
        }
    finally:
        if wandb is not None and owns_wandb_run and wandb.run is not None:
            wandb.finish()
