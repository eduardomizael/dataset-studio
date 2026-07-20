"""Adaptador configurável de treinamento usando Ultralytics."""

from __future__ import annotations

import sys
from pathlib import Path

from dataset_studio.ports.trainer import Trainer, TrainingParams


class UltralyticsCommandTrainer(Trainer):
    """Monta os argumentos e comando CLI para treinar o modelo via Ultralytics."""

    def build_command(self, data_yaml_path: Path, params: TrainingParams) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "ultralytics",
            "detect",
            "train",
            f"data={data_yaml_path.resolve()}",
            f"model={params.model}",
            f"epochs={params.epochs}",
            f"imgsz={params.imgsz}",
            f"batch={params.batch}",
            f"workers={params.workers}",
            f"device={params.device}",
            f"patience={params.patience}",
            f"lr0={params.lr0}",
            f"optimizer={params.optimizer}",
        ]
        if params.project:
            cmd.append(f"project={Path(params.project).resolve()}")
        if params.name:
            cmd.append(f"name={params.name}")
        for k, v in params.extra_args.items():
            cmd.append(f"{k}={v}")
        return cmd
