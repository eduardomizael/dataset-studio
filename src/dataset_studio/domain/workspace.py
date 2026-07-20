"""Gerenciamento de caminhos e workspaces do Dataset Studio."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dataset_studio.domain.errors import WorkflowError

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SPLITS = ("train", "val", "test_normal", "test_stress")
BOX_BOUNDARY_TOLERANCE = 0.001


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_id(value: str, label: str) -> None:
    if not ID_PATTERN.fullmatch(value):
        raise WorkflowError(
            f"{label} deve usar apenas letras, numeros, hifen e sublinhado."
        )


def label_studio_region_id(source_id: str | int, index: int) -> str:
    """Gera ID estável usando apenas caracteres aceitos pelo Label Studio."""
    digest = hashlib.sha1(str(source_id).encode("utf-8")).hexdigest()[:16]
    return f"pred_{digest}_{index}"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise WorkflowError(f"Arquivo nao encontrado: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WorkflowError(f"YAML invalido: {path}")
    return payload


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class Workspace:
    """Representa a raiz do workspace configurável e seus caminhos derivados."""

    root: Path

    @classmethod
    def from_path(cls, path: str | Path) -> Workspace:
        return cls(root=Path(path).resolve())

    def resolve_path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def campaigns_root(self) -> Path:
        return self.root / "dataset" / "campaigns"

    @property
    def releases_root(self) -> Path:
        return self.root / "dataset" / "releases"

    @property
    def videos_root(self) -> Path:
        return self.root / "videos"

    @property
    def models_root(self) -> Path:
        return self.root / "models"

    @property
    def config_root(self) -> Path:
        return self.root / "config"

    def campaign_root(self, campaign_id: str) -> Path:
        return self.campaigns_root / campaign_id

    def release_root(self, release_id: str) -> Path:
        return self.releases_root / release_id

    def campaign_config_path(self, campaign_id: str) -> Path:
        return self.campaign_root(campaign_id) / "campaign.yaml"

    def release_config_path(self, release_id: str) -> Path:
        return self.release_root(release_id) / "release.yaml"

    def defaults_dict(self, classes: list[str] | None = None) -> dict[str, Any]:
        """Gera dicionário de defaults para retrocompatibilidade."""
        return {
            "paths": {
                "campaigns_root": self.campaigns_root,
                "releases_root": self.releases_root,
                "videos_root": self.videos_root,
                "models_root": self.models_root,
            },
            "extraction": {
                "mode": "uniform",
                "fps_sample": 1.0,
                "confidence_threshold": 0.25,
                "model": None,
            },
            "annotation": {
                "classes": classes or ["objeto"],
            },
        }
