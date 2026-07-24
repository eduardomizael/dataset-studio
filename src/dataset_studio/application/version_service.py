"""Serviços da aplicação para gerenciamento e montagem de versões de dataset (versions)."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from dataset_studio.adapters.ultralytics.trainer import UltralyticsCommandTrainer
from dataset_studio.domain import (
    Workspace,
    WorkflowError,
    annotation_source_paths,
    assess_split_sufficiency,
    list_annotation_revisions,
    load_annotation_revision_report,
    load_frame_manifest,
    load_source,
    load_yaml,
    parse_native_export,
    resolve_class_mapping,
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


def preview_combined_split_metrics(
    ws: Workspace,
    source_ids: list[str],
    assignments: dict[str, list[str]],
    revision_ids: dict[str, str],
    *,
    evaluation_level: str = "standard",
    class_mapping: dict[str, dict[str, str | None]] | None = None,
    final_classes: list[str] | None = None,
) -> dict[str, Any]:
    """Calcula a prévia consolidada de uma release com múltiplas origens."""
    if not source_ids:
        raise WorkflowError("Selecione ao menos uma origem.")
    if set(revision_ids) != set(source_ids):
        raise WorkflowError("Selecione exatamente uma revisão para cada origem.")

    source_classes: dict[str, list[str]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for source_id in source_ids:
        source = load_source(ws, source_id)
        source_classes[source_id] = list(source["annotation"]["classes"])
        report = load_annotation_revision_report(
            ws, source_id, revision_ids[source_id]
        )
        reports[source_id] = report
        source_counts[source_id] = {
            str(name): int(count)
            for name, count in (report.get("class_counts") or {}).items()
        }

    class_resolution = resolve_class_mapping(
        source_classes,
        source_counts,
        class_mapping=class_mapping,
        final_classes=final_classes,
        acknowledged=True,
    )
    unit_metrics: dict[str, dict[str, Any]] = {}
    per_source: dict[str, dict[str, int]] = {}
    for source_id in source_ids:
        report = reports[source_id]
        per_unit = report.get("per_unit") or report.get("per_video") or {}
        source_total = {"units": len(per_unit), "frames": 0, "boxes": 0}
        for unit_id, metrics in per_unit.items():
            key = f"{source_id}/{unit_id}"
            unit_metrics[key] = {
                **metrics,
                "boxes": 0,
            }
            source_total["frames"] += int(
                metrics.get("included", metrics.get("completed", 0)) or 0
            )

        export_path, _ = annotation_source_paths(
            ws, source_id, revision_ids[source_id]
        )
        annotations, _ = parse_native_export(
            ws,
            source_id,
            export_path,
            allow_pending=bool(report.get("allow_pending", False)),
        )
        manifest = load_frame_manifest(ws, source_id)
        unit_by_frame = {
            str(frame["frame_id"]): str(
                frame.get("unit_id") or frame["source_video"]
            )
            for frame in manifest["frames"]
        }
        original_classes = source_classes[source_id]
        mapping = class_resolution["mapping"][source_id]
        for frame_id, annotation in annotations.items():
            if annotation.excluded:
                continue
            unit_id = unit_by_frame[frame_id]
            key = f"{source_id}/{unit_id}"
            mapped_boxes = 0
            for box in annotation.boxes:
                original_id = int(box.split()[0])
                original_name = original_classes[original_id]
                if mapping[original_name] is not None:
                    mapped_boxes += 1
            unit_metrics[key]["boxes"] += mapped_boxes
            source_total["boxes"] += mapped_boxes
        per_source[source_id] = source_total

    def sum_metrics(keys: list[str]) -> dict[str, int]:
        return {
            "videos": len(keys),
            "frames": sum(
                int(
                    (unit_metrics.get(key) or {}).get(
                        "included",
                        (unit_metrics.get(key) or {}).get("completed", 0),
                    )
                    or 0
                )
                for key in keys
            ),
            "boxes": sum(
                int((unit_metrics.get(key) or {}).get("boxes", 0) or 0)
                for key in keys
            ),
        }

    result = {
        split: sum_metrics(assignments.get(split, []))
        for split in ("train", "val", "test_normal", "test_stress")
    }
    result["quality_assessment"] = assess_split_sufficiency(
        assignments,
        unit_metrics,
        evaluation_level,
    )
    result["per_source"] = per_source
    result["class_resolution"] = class_resolution
    return result


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
