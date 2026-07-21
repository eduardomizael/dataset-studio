"""Adaptador configurável de treinamento usando Ultralytics."""

from __future__ import annotations

import sys
from pathlib import Path

from dataset_studio.ports.trainer import Trainer, TrainingParams


class UltralyticsCommandTrainer(Trainer):
    """Monta os argumentos e comando CLI para treinar o modelo via Ultralytics."""

    def build_command(self, data_yaml_path: Path, params: TrainingParams) -> list[str]:
        """Constrói a lista de comandos CLI para o subprocesso do Ultralytics YOLO."""

        fish_venv_yolo = Path(r"C:\Users\eduar\Desktop\fish_detection\.venv\Scripts\yolo.exe")
        if fish_venv_yolo.is_file():
            cmd = [str(fish_venv_yolo)]
        else:
            python_exec = sys.executable
            fish_venv_python = Path(r"C:\Users\eduar\Desktop\fish_detection\.venv\Scripts\python.exe")
            if fish_venv_python.is_file():
                python_exec = str(fish_venv_python)
            cmd = [python_exec, "-u", "-c", "import sys, ultralytics.cfg; sys.argv=['yolo', *sys.argv[1:]]; ultralytics.cfg.entrypoint()"]

        device_val = params.device
        if device_val == "auto":
            try:
                import torch
                device_val = "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device_val = "cpu"

        cmd.extend([
            "detect",
            "train",
            f"data={data_yaml_path.resolve()}",
            f"model={params.model}",
            f"epochs={params.epochs}",
            f"imgsz={params.imgsz}",
            f"batch={params.batch}",
            f"workers={params.workers}",
            f"device={device_val}",
            f"patience={params.patience}",
            f"lr0={params.lr0}",
            f"optimizer={params.optimizer}",
        ])
        if params.project:
            cmd.append(f"project={Path(params.project).resolve()}")
        if params.name:
            cmd.append(f"name={params.name}")
        for k, v in params.extra_args.items():
            cmd.append(f"{k}={v}")
        return cmd
