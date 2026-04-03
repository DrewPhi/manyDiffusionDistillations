"""Artifact manifest and registry helpers for stage-based pipelines."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


@dataclass
class ArtifactManifest:
    """Typed manifest emitted by each stage run."""

    run_id: str
    stage_name: str
    stage_type: str
    status: str
    created_at: str
    config_hash: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        stage_name: str,
        stage_type: str,
        config: Any,
        inputs: Dict[str, Any] | None = None,
        outputs: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
        status: str = "completed",
    ) -> "ArtifactManifest":
        config_resolved = OmegaConf.to_container(config, resolve=True)
        config_blob = json.dumps(config_resolved, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_blob.encode("utf-8")).hexdigest()
        return cls(
            run_id=run_id,
            stage_name=stage_name,
            stage_type=stage_type,
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(),
            config_hash=config_hash,
            inputs=inputs or {},
            outputs=outputs or {},
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArtifactRegistry:
    """Filesystem-backed artifact registry keyed by run_id and stage name."""

    MANIFEST_FILENAME = "manifest.json"

    def __init__(self, output_root: str | Path, run_id: str):
        self.output_root = Path(output_root)
        self.run_id = run_id
        self.run_dir = self.output_root / run_id

    def stage_dir(self, stage_name: str) -> Path:
        return self.run_dir / stage_name

    def manifest_path(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name) / self.MANIFEST_FILENAME

    def has_manifest(self, stage_name: str) -> bool:
        return self.manifest_path(stage_name).exists()

    def load_manifest(self, stage_name: str) -> ArtifactManifest:
        path = self.manifest_path(stage_name)
        data = json.loads(path.read_text(encoding="utf-8"))
        return ArtifactManifest(**data)

    def save_manifest(self, manifest: ArtifactManifest) -> Path:
        stage_dir = self.stage_dir(manifest.stage_name)
        stage_dir.mkdir(parents=True, exist_ok=True)
        path = stage_dir / self.MANIFEST_FILENAME
        path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path
