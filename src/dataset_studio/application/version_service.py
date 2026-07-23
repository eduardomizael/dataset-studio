"""Serviços da aplicação para gerenciamento e montagem de versões de dataset (versions)."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from dataset_studio.adapters.ultralytics.trainer import UltralyticsCommandTrainer
from dataset_studio.domain import (
    Workspace,
    WorkflowError,
    assess_split_sufficiency,
    list_annotation_revisions,
    load_annotation_revision_report,
    load_yaml,
    version_config_path,
    version_root,
)
from dataset_studio.ports.trainer import TrainingParams


def preview_split_metrics(
    ws: Workspace,
    source_id: str,
    assignments: dict[str, list[str]],
    revision_id: str | None = None,
    evaluation_level: str = "standard",
) -> dict[str, Any]:
    """Calcula a prévia das métricas de contagem (vídeos, frames e boxes) para cada split."""

    revisions = list_annotation_revisions(ws, source_id)
    target_rev = revision_id or (revisions[-1] if revisions else None)
    if not target_rev:
        empty_split = {"videos": 0, "frames": 0, "boxes": 0}
        return {
            "train": empty_split,
            "val": empty_split,
            "test_normal": empty_split,
            "test_stress": empty_split,
        }

    report = load_annotation_revision_report(ws, source_id, target_rev)
    per_video = report.get("per_unit") or report.get("per_video", {})

    def sum_metrics(v_list: list[str]) -> dict[str, int]:
        total_f = 0
        total_b = 0
        for item in v_list:
            v_name = item.split("/")[-1] if "/" in item else item
            v_data = per_video.get(v_name, {})
            total_f += v_data.get("included", v_data.get("completed", 0))
            total_b += v_data.get("boxes", 0)
        return {"videos": len(v_list), "frames": total_f, "boxes": total_b}

    split_metrics = {
        "train": sum_metrics(assignments.get("train", [])),
        "val": sum_metrics(assignments.get("val", [])),
        "test_normal": sum_metrics(assignments.get("test_normal", [])),
        "test_stress": sum_metrics(assignments.get("test_stress", [])),
    }
    unit_metrics = {
        f"{source_id}/{unit_id}": metrics
        for unit_id, metrics in per_video.items()
    }
    split_metrics["quality_assessment"] = assess_split_sufficiency(
        assignments,
        unit_metrics,
        evaluation_level,
    )
    return split_metrics


def training_recipe(ws: Workspace, version_id: str, params: TrainingParams | None = None) -> dict[str, Any]:
    """Prepara o comando de treinamento e parâmetros para execução de uma versão de dataset."""

    root = version_root(ws, version_id)
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        raise WorkflowError("Materialize a versao antes de configurar o treinamento.")

    p = params or TrainingParams()
    p = replace(
        p,
        project=p.project or str(ws.runs_root),
        name=p.name or version_id,
    )
    trainer = UltralyticsCommandTrainer()
    command = trainer.build_command(data_yaml, p)

    return {
        "version_id": version_id,
        "release_id": version_id,
        "data_yaml": str(data_yaml),
        "params": p.to_dict(),
        "command": command,
        "command_str": " ".join(command),
    }


def version_status(ws: Workspace, version_id: str) -> dict[str, Any]:
    """Retorna o status completo e relatório de materialização de uma versão de dataset."""

    config = load_yaml(version_config_path(ws, version_id))
    root = version_root(ws, version_id)
    report_path = root / "build_report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else None
    )
    data_yaml = root / "data.yaml"
    recipe = training_recipe(ws, version_id) if data_yaml.exists() else None
    return {
        "version_id": version_id,
        "release_id": version_id,
        "sources": config.get("sources") or config.get("campaigns", []),
        "campaigns": config.get("sources") or config.get("campaigns", []),
        "annotation_revisions": config.get("annotation_revisions", {}),
        "provisional": bool(config.get("provisional", False)),
        "assignments": config["assignments"],
        "materialized": (root / "manifest.csv").exists(),
        "build_report": report,
        "training_recipe": recipe,
    }


# Alias retrocompatível
release_status = version_status
