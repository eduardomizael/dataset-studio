"""Porta de abstração para a etapa de treinamento de modelos."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class TrainingParams:
    """Parâmetros de treinamento totalmente configuráveis pelo usuário."""

    model: str = "yolo26n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = -1
    workers: int = 0
    device: str = "auto"
    patience: int = 50
    lr0: float = 0.01
    optimizer: str = "auto"
    project: str | None = None
    name: str | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "workers": self.workers,
            "device": self.device,
            "patience": self.patience,
            "lr0": self.lr0,
            "optimizer": self.optimizer,
            "project": self.project,
            "name": self.name,
            "extra_args": self.extra_args,
        }


class Trainer(Protocol):
    """Contrato para geradores de comandos e executores de treinamento."""

    def build_command(self, data_yaml_path: Path, params: TrainingParams) -> list[str]:
        ...
