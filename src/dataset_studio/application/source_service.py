"""Serviços da aplicação para gerenciamento de origens de dataset (sources)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dataset_studio.adapters.opencv.media import get_video_info
from dataset_studio.domain import (
    Workspace,
    accept_native_export,
    export_annotations_path,
    frame_manifest_path,
    import_tasks_path,
    inspect_native_export,
    list_annotation_revisions,
    load_annotation_revision_report,
    load_source,
    load_frame_manifest,
    selected_export_path,
    source_root,
)


def inspect_finished_tasks(ws: Workspace, source_id: str) -> dict[str, Any]:
    """Inspeciona a pasta finished_tasks buscando por relatórios de tarefas recém-anotadas."""

    finished_dir = ws.source_root(source_id) / "label_studio" / "finished_tasks"
    finished_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(finished_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        return {"found": False, "files": [], "exports": [], "finished_tasks_dir": str(finished_dir)}

    exports = []
    for file in json_files:
        try:
            insp = inspect_native_export(ws, source_id, file, allow_pending=True)
            rep = insp.get("report") or {}
            exports.append(
                {
                    "name": file.name,
                    "path": str(file),
                    "mtime": file.stat().st_mtime,
                    "size": file.stat().st_size,
                    "valid": insp.get("valid", False),
                    "metrics": {
                        "total_tasks": rep.get("tasks_valid", rep.get("tasks_expected", 0)),
                        "tasks_completed": rep.get("tasks_completed", 0),
                        "tasks_deferred": rep.get("tasks_deferred", 0),
                        "tasks_cancelled": rep.get("exclusion_reasons", {}).get("skipped_or_cancelled", 0),
                        "tasks_excluded": rep.get("tasks_excluded", 0),
                        "positive_frames": rep.get("positive_frames", 0),
                        "confirmed_negatives": rep.get("confirmed_negatives", 0),
                        "total_boxes": rep.get("boxes", 0),
                        "class_counts": rep.get("class_counts", {}),
                        "snapshot_type": rep.get("snapshot_type", "complete"),
                        "per_video": rep.get("per_video", {}),
                    },
                }
            )
        except Exception as exc:
            exports.append(
                {
                    "name": file.name,
                    "path": str(file),
                    "error": str(exc),
                }
            )

    latest_export = exports[0] if exports else None
    latest_file = json_files[0]
    return {
        "found": True,
        "finished_tasks_dir": str(finished_dir),
        "files": [f.name for f in json_files],
        "exports": exports,
        "latest_file": {
            "name": latest_file.name,
            "path": str(latest_file),
        },
        "metrics": latest_export.get("metrics", {}) if (latest_export and "metrics" in latest_export) else {},
    }


def source_status(ws: Workspace, source_id: str) -> dict[str, Any]:
    """Retorna o estado detalhado do pipeline e das etapas de uma fonte de dados."""

    source = load_source(ws, source_id)
    annotation = source["annotation"]
    annotation_backend = annotation.get("backend") or (
        "local" if annotation.get("model") else "none"
    )
    manifest_path = frame_manifest_path(ws, source_id)
    frames = 0
    videos = len(source["videos"]["files"])
    if manifest_path.exists():
        manifest = load_frame_manifest(ws, source_id)
        frames = len(manifest["frames"])
    tasks_path = import_tasks_path(ws, source_id)
    tasks = 0
    if tasks_path.exists():
        payload = json.loads(tasks_path.read_text(encoding="utf-8"))
        tasks = len(payload) if isinstance(payload, list) else -1

    # Verificar pasta finished_tasks
    finished_info = inspect_finished_tasks(ws, source_id)
    if finished_info["found"] and not list_annotation_revisions(ws, source_id):
        # Auto-aceitar exportação do finished_tasks
        latest_path = Path(finished_info["latest_file"]["path"])
        accept_native_export(ws, source_id, latest_path, revision_id="rev_auto", allow_pending=True)

    revisions = []
    for revision_id in list_annotation_revisions(ws, source_id):
        revision_report = load_annotation_revision_report(ws, source_id, revision_id)
        revisions.append(
            {
                "revision_id": revision_id,
                "snapshot_type": revision_report.get("snapshot_type", "complete"),
                "tasks_completed": revision_report.get("tasks_completed", 0),
                "tasks_deferred": revision_report.get("tasks_deferred", 0),
                "tasks_excluded": revision_report.get("tasks_excluded", 0),
                "positive_frames": revision_report.get("positive_frames", 0),
                "confirmed_negatives": revision_report.get("confirmed_negatives", 0),
                "boxes": revision_report.get("boxes", 0),
                "per_video": revision_report.get("per_video", {}),
            }
        )
    accepted = bool(revisions)
    selected_export = selected_export_path(ws, source_id)
    report = (
        load_annotation_revision_report(
            ws, source_id, revisions[-1]["revision_id"]
        )
        if revisions
        else None
    )

    if not frames:
        next_action = "extract"
    elif tasks != frames:
        next_action = "build-import"
    elif not accepted:
        next_action = "annotate"
    else:
        next_action = "ready-for-release"

    videos_dir = ws.resolve_path(source["videos"]["directory"])
    video_details = []
    for v_item in source["videos"].get("files", []):
        v_name = v_item["name"] if isinstance(v_item, dict) else str(v_item)
        v_path = videos_dir / v_name
        fallback_size = v_item.get("size", 0) if isinstance(v_item, dict) else 0
        video_details.append(get_video_info(v_path, fallback_size=fallback_size))

    return {
        "source_id": source_id,
        "campaign_id": source_id,
        "videos": videos,
        "video_details": video_details,
        "frames": frames,
        "import_tasks": tasks,
        "export_accepted": accepted,
        "finished_info": finished_info,
        "annotation_revisions": revisions,
        "latest_annotation_revision": (
            revisions[-1]["revision_id"] if revisions else None
        ),
        "annotation_report": report,
        "selected_export": (
            {"path": str(selected_export), "name": selected_export.name}
            if selected_export.is_file()
            else None
        ),
        "annotation_backend": annotation_backend,
        "annotation_model": annotation.get("model"),
        "annotation_detection_config": annotation.get("detection_config"),
        "local_files_storage_path": str(
            (
                source_root(ws, source_id)
                / "frames"
                / "raw"
                / "images"
            ).resolve()
        ),
        "next_action": next_action,
    }


def list_available_models(ws: Workspace) -> list[str]:
    """Lista os modelos YOLO (.pt) disponíveis no diretório de modelos do workspace."""

    models_dir = ws.models_root
    if not models_dir.exists():
        return []
    return [
        (
            path.relative_to(ws.root).as_posix()
            if path.is_relative_to(ws.root)
            else str(path)
        )
        for path in sorted(models_dir.glob("*.pt"))
        if path.is_file()
    ]


# Alias retrocompatível
campaign_status = source_status
