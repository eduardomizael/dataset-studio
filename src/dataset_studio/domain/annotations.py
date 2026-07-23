"""Regras de domínio para parsing, validação e aceitação de anotações."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_studio.domain.sources import (
    annotation_report_path,
    annotation_revision_export_path,
    annotation_revision_report_path,
    export_annotations_path,
    load_frame_manifest,
    load_source as load_campaign,
)
from dataset_studio.domain.errors import WorkflowError
from dataset_studio.domain.workspace import (
    BOX_BOUNDARY_TOLERANCE,
    Workspace,
    sha256,
    utc_now,
    validate_id,
)


@dataclass(frozen=True)
class AnnotationFrame:
    """Representa a estrutura anotada de um frame específico do dataset."""

    frame_id: str
    boxes: tuple[str, ...]
    is_negative: bool
    excluded: bool = False
    exclusion_reason: str | None = None



def _select_human_annotation(
    task: dict[str, Any], frame_id: str, *, allow_pending: bool = False
) -> tuple[dict[str, Any] | None, str | None]:
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        raise WorkflowError(f"Task {frame_id}: campo annotations ausente.")
    cancelled = [
        annotation
        for annotation in annotations
        if isinstance(annotation, dict) and annotation.get("was_cancelled", False)
    ]
    valid = [
        annotation
        for annotation in annotations
        if isinstance(annotation, dict) and not annotation.get("was_cancelled", False)
    ]
    if not valid:
        if cancelled:
            return None, "skipped_or_cancelled"
        if allow_pending:
            return None, "deferred"
        raise WorkflowError(f"Task {frame_id}: nenhuma anotacao humana concluida.")
    if len(valid) != 1:
        raise WorkflowError(
            f"Task {frame_id}: {len(valid)} anotacoes humanas validas; resolva a ambiguidade."
        )
    result = valid[0].get("result")
    if not isinstance(result, list):
        raise WorkflowError(f"Task {frame_id}: result nao e uma lista.")
    return valid[0], None


def _normalized_box(
    value: dict[str, Any], *, frame_id: str
) -> tuple[float, float, float, float, bool]:
    try:
        x = float(value["x"])
        y = float(value["y"])
        width = float(value["width"])
        height = float(value["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(f"Task {frame_id}: coordenadas invalidas.") from exc
    right = x + width
    bottom = y + height
    if (
        width <= 0
        or height <= 0
        or x < -BOX_BOUNDARY_TOLERANCE
        or y < -BOX_BOUNDARY_TOLERANCE
        or right > 100 + BOX_BOUNDARY_TOLERANCE
        or bottom > 100 + BOX_BOUNDARY_TOLERANCE
    ):
        raise WorkflowError(f"Task {frame_id}: caixa fora dos limites da imagem.")
    normalized_x = max(0.0, x)
    normalized_y = max(0.0, y)
    normalized_right = min(100.0, right)
    normalized_bottom = min(100.0, bottom)
    normalized_width = normalized_right - normalized_x
    normalized_height = normalized_bottom - normalized_y
    if normalized_width <= 0 or normalized_height <= 0:
        raise WorkflowError(f"Task {frame_id}: caixa fora dos limites da imagem.")
    adjusted = (normalized_x, normalized_y, normalized_right, normalized_bottom) != (
        x,
        y,
        right,
        bottom,
    )
    return (
        normalized_x,
        normalized_y,
        normalized_width,
        normalized_height,
        adjusted,
    )


def _result_to_yolo(
    result: dict[str, Any],
    *,
    frame_id: str,
    class_to_id: dict[str, int],
) -> str:
    if result.get("type") != "rectanglelabels":
        raise WorkflowError(
            f"Task {frame_id}: resultado inesperado do tipo {result.get('type')}."
        )
    value = result.get("value")
    if not isinstance(value, dict):
        raise WorkflowError(f"Task {frame_id}: value invalido.")
    labels = value.get("rectanglelabels")
    if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in class_to_id:
        raise WorkflowError(f"Task {frame_id}: classe ausente ou desconhecida.")
    if float(value.get("rotation", 0) or 0) != 0:
        raise WorkflowError(f"Task {frame_id}: caixas rotacionadas nao sao suportadas.")
    x, y, width, height, _ = _normalized_box(value, frame_id=frame_id)
    xc = (x + width / 2) / 100
    yc = (y + height / 2) / 100
    return (
        f"{class_to_id[labels[0]]} {xc:.6f} {yc:.6f} "
        f"{width / 100:.6f} {height / 100:.6f}"
    )


def parse_native_export(
    defaults_or_ws: dict[str, Any] | Workspace,
    campaign_id: str,
    export_path: Path,
    *,
    allow_pending: bool = False,
) -> tuple[dict[str, AnnotationFrame], dict[str, Any]]:
    """Realiza o parsing do arquivo JSON de exportação nativa do Label Studio.

    Args:
        defaults_or_ws: Objeto Workspace ou dicionário de configurações.
        campaign_id: Identificador da campanha/fonte de dados.
        export_path: Caminho para o arquivo JSON de exportação.
        allow_pending: Se True, tolera tarefas pendentes de anotação.

    Returns:
        Tupla contendo o dicionário de frames anotados e o relatório estatístico.
    """

    campaign = load_campaign(defaults_or_ws, campaign_id)
    manifest = load_frame_manifest(defaults_or_ws, campaign_id)
    expected = {frame["frame_id"]: frame for frame in manifest["frames"]}
    try:
        tasks = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Export JSON invalido: {export_path}") from exc
    if not isinstance(tasks, list):
        raise WorkflowError("A exportacao nativa deve ser uma lista de tasks.")

    by_frame: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
            raise WorkflowError("Task exportada sem data.")
        frame_id = task["data"].get("frame_id")
        if not isinstance(frame_id, str):
            raise WorkflowError("Task exportada sem data.frame_id.")
        if frame_id in by_frame:
            raise WorkflowError(f"frame_id duplicado na exportacao: {frame_id}")
        by_frame[frame_id] = task

    missing = set(expected) - set(by_frame)
    extra = set(by_frame) - set(expected)
    if extra or (missing and not allow_pending):
        parts = []
        if missing:
            parts.append(f"{len(missing)} tasks ausentes")
        if extra:
            parts.append(f"{len(extra)} tasks de outra campanha")
        raise WorkflowError("Cobertura da exportacao invalida: " + ", ".join(parts))

    class_names = list(campaign["annotation"]["classes"])
    class_to_id = {name: index for index, name in enumerate(class_names)}
    parsed: dict[str, AnnotationFrame] = {}
    box_count = 0
    negatives = 0
    excluded = 0
    exclusion_reasons: Counter[str] = Counter()
    per_video: dict[str, Counter[str]] = {
        video: Counter()
        for video in sorted({frame["source_video"] for frame in expected.values()})
    }
    per_unit: dict[str, Counter[str]] = {
        str(frame.get("unit_id") or frame["source_video"]): Counter()
        for frame in expected.values()
    }
    for frame_id in sorted(expected) if isinstance(tasks, list) else []:
        frame = expected[frame_id]
        video_stats = per_video[frame["source_video"]]
        unit_stats = per_unit[str(frame.get("unit_id") or frame["source_video"])]
        video_stats["source_frames"] += 1
        unit_stats["source_frames"] += 1
        task = by_frame.get(frame_id)
        if task is None:
            parsed[frame_id] = AnnotationFrame(
                frame_id=frame_id,
                boxes=(),
                is_negative=False,
                excluded=True,
                exclusion_reason="deferred",
            )
            excluded += 1
            exclusion_reasons["deferred"] += 1
            video_stats["deferred"] += 1
            unit_stats["deferred"] += 1
            continue
        annotation, cancelled_reason = _select_human_annotation(
            task, frame_id, allow_pending=allow_pending
        )
        if annotation is None:
            parsed[frame_id] = AnnotationFrame(
                frame_id=frame_id,
                boxes=(),
                is_negative=False,
                excluded=True,
                exclusion_reason=cancelled_reason,
            )
            excluded += 1
            exclusion_reasons[str(cancelled_reason)] += 1
            video_stats[str(cancelled_reason)] += 1
            unit_stats[str(cancelled_reason)] += 1
            continue
        rows = tuple(
            _result_to_yolo(result, frame_id=frame_id, class_to_id=class_to_id)
            for result in annotation["result"]
        )
        parsed[frame_id] = AnnotationFrame(
            frame_id=frame_id, boxes=rows, is_negative=not rows
        )
        box_count += len(rows)
        negatives += int(not rows)
        video_stats["completed"] += 1
        video_stats["boxes"] += len(rows)
        video_stats["confirmed_negatives" if not rows else "positive_frames"] += 1
        unit_stats["completed"] += 1
        unit_stats["boxes"] += len(rows)
        unit_stats["confirmed_negatives" if not rows else "positive_frames"] += 1
    deferred = exclusion_reasons["deferred"]
    class_counts: Counter[str] = Counter()
    for frame_obj in parsed.values():
        for box_str in frame_obj.boxes:
            parts = box_str.split()
            if parts:
                cls_idx = int(parts[0])
                cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                class_counts[cls_name] += 1

    report = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "validated_at": utc_now(),
        "export_file": str(export_path),
        "export_sha256": sha256(export_path),
        "tasks_expected": len(expected),
        "tasks_exported": len(by_frame),
        "tasks_valid": len(parsed),
        "tasks_completed": len(parsed) - excluded,
        "tasks_excluded": excluded,
        "tasks_deferred": deferred,
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "boxes": box_count,
        "positive_frames": len(parsed) - excluded - negatives,
        "confirmed_negatives": negatives,
        "class_counts": dict(sorted(class_counts.items())),
        "allow_pending": allow_pending,
        "snapshot_type": "provisional" if deferred else "complete",
        "per_video": {
            video: {
                key: stats[key]
                for key in (
                    "source_frames",
                    "completed",
                    "positive_frames",
                    "confirmed_negatives",
                    "skipped_or_cancelled",
                    "deferred",
                    "boxes",
                )
            }
            for video, stats in per_video.items()
        },
        "per_unit": {
            unit_id: {
                key: stats[key]
                for key in (
                    "source_frames",
                    "completed",
                    "positive_frames",
                    "confirmed_negatives",
                    "skipped_or_cancelled",
                    "deferred",
                    "boxes",
                )
            }
            for unit_id, stats in per_unit.items()
        },
    }
    return parsed, report


def inspect_native_export(
    defaults_or_ws: dict[str, Any] | Workspace,
    campaign_id: str,
    export_path: Path,
    *,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Inspeciona e valida a integridade de um arquivo JSON de exportação de anotações.

    Args:
        defaults_or_ws: Objeto Workspace ou dicionário de configurações.
        campaign_id: Identificador da fonte de dados.
        export_path: Caminho do arquivo JSON a ser inspecionado.
        allow_pending: Se True, ignora tarefas não finalizadas.

    Returns:
        Dicionário com o relatório de inspeção e métricas detectadas.
    """

    campaign = load_campaign(defaults_or_ws, campaign_id)
    manifest = load_frame_manifest(defaults_or_ws, campaign_id)
    expected = {frame["frame_id"]: frame for frame in manifest["frames"]}
    export_path = (
        export_path
        if export_path.is_absolute()
        else (
            defaults_or_ws.root / export_path
            if isinstance(defaults_or_ws, Workspace)
            else Path(export_path)
        )
    )
    issues: list[dict[str, Any]] = []

    def add_issue(
        *,
        severity: str,
        code: str,
        message: str,
        resolution: str,
        task: dict[str, Any] | None = None,
        frame_id: str | None = None,
        region_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        data = task.get("data", {}) if isinstance(task, dict) else {}
        issues.append(
            {
                "severity": severity,
                "code": code,
                "message": message,
                "resolution": resolution,
                "frame_id": frame_id,
                "task_id": task.get("id") if isinstance(task, dict) else None,
                "project_id": (
                    task.get("project") if isinstance(task, dict) else None
                ),
                "image": data.get("image") if isinstance(data, dict) else None,
                "region_id": region_id,
                "details": details or {},
            }
        )

    try:
        tasks = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        add_issue(
            severity="error",
            code="invalid_json",
            message=f"O arquivo nao e um JSON valido: {exc}",
            resolution="Exporte novamente no formato JSON nativo do Label Studio.",
        )
        tasks = None

    if tasks is not None and not isinstance(tasks, list):
        add_issue(
            severity="error",
            code="invalid_export_format",
            message="A exportacao nativa deve ser uma lista de tasks.",
            resolution="No Label Studio, escolha Export > JSON, nao JSON_MIN nem YOLO.",
        )
        tasks = None

    by_frame: dict[str, dict[str, Any]] = {}
    if isinstance(tasks, list):
        for position, task in enumerate(tasks, 1):
            if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
                add_issue(
                    severity="error",
                    code="invalid_task",
                    message=f"Item {position} nao possui o objeto data.",
                    resolution="Exporte novamente o projeto em JSON nativo.",
                )
                continue
            frame_id = task["data"].get("frame_id")
            if not isinstance(frame_id, str):
                add_issue(
                    severity="error",
                    code="missing_frame_id",
                    message=f"Task {task.get('id', position)} sem data.frame_id.",
                    resolution=(
                        "Use as tasks importadas pelo workflow; nao remova frame_id "
                        "dos dados da task."
                    ),
                    task=task,
                )
                continue
            if frame_id in by_frame:
                add_issue(
                    severity="error",
                    code="duplicate_frame",
                    message=f"O frame {frame_id} aparece mais de uma vez na exportacao.",
                    resolution=(
                        "Remova a task duplicada no projeto ou refaca a importacao "
                        "em um projeto vazio."
                    ),
                    task=task,
                    frame_id=frame_id,
                )
                continue
            by_frame[frame_id] = task
            if frame_id not in expected:
                add_issue(
                    severity="error",
                    code="foreign_task",
                    message=f"O frame {frame_id} nao pertence a esta campanha.",
                    resolution="Exporte somente o projeto da campanha selecionada.",
                    task=task,
                    frame_id=frame_id,
                )

    class_names = list(campaign["annotation"]["classes"])
    class_to_id = {name: index for index, name in enumerate(class_names)}
    for frame_id in sorted(expected):
        task = by_frame.get(frame_id)
        if task is None:
            add_issue(
                severity="warning" if allow_pending else "error",
                code="pending_task",
                message=f"O frame {frame_id} nao esta presente na exportacao.",
                resolution=(
                    "Conclua a task no Label Studio ou marque Snapshot parcial "
                    "para adia-la sem transforma-la em negativo."
                ),
                frame_id=frame_id,
            )
            continue
        try:
            annotation, excluded_reason = _select_human_annotation(
                task, frame_id, allow_pending=allow_pending
            )
        except WorkflowError as exc:
            message = str(exc)
            ambiguous = "anotacoes humanas validas" in message
            pending = "nenhuma anotacao humana" in message
            add_issue(
                severity="error",
                code="ambiguous_annotations" if ambiguous else "pending_task" if pending else "invalid_annotation",
                message=message,
                resolution=(
                    "Mantenha somente uma anotacao concluida para a task."
                    if ambiguous
                    else "Conclua a task no Label Studio ou marque Snapshot parcial."
                    if pending
                    else "Abra a task no Label Studio, revise e salve novamente."
                ),
                task=task,
                frame_id=frame_id,
            )
            continue
        if annotation is None:
            if excluded_reason == "deferred":
                add_issue(
                    severity="warning",
                    code="pending_task",
                    message=f"Task {frame_id} ainda nao foi concluida.",
                    resolution=(
                        "Ela sera adiada nesta revisao parcial e podera ser "
                        "anotada em uma revisao futura."
                    ),
                    task=task,
                    frame_id=frame_id,
                )
            continue
        for region_index, result in enumerate(annotation["result"], 1):
            region_id = result.get("id") if isinstance(result, dict) else None
            try:
                _result_to_yolo(
                    result, frame_id=frame_id, class_to_id=class_to_id
                )
            except (AttributeError, WorkflowError) as exc:
                message = str(exc)
                if "fora dos limites" in message:
                    code = "box_out_of_bounds"
                    resolution = (
                        "Abra esta task no Label Studio, selecione a caixa indicada "
                        "e redimensione-a para ficar inteiramente dentro da imagem."
                    )
                elif "rotacionadas" in message:
                    code = "rotated_box"
                    resolution = "Apague a caixa rotacionada e desenhe outra sem rotacao."
                elif "classe" in message:
                    code = "invalid_class"
                    resolution = "Atribua uma das classes configuradas para a campanha."
                else:
                    code = "invalid_region"
                    resolution = "Revise ou redesenhe esta caixa no Label Studio."
                value = result.get("value", {}) if isinstance(result, dict) else {}
                add_issue(
                    severity="error",
                    code=code,
                    message=f"{message} Regiao {region_index}.",
                    resolution=resolution,
                    task=task,
                    frame_id=frame_id,
                    region_id=region_id,
                    details={
                        key: value.get(key)
                        for key in ("x", "y", "width", "height")
                        if isinstance(value, dict) and key in value
                    },
                )
                continue
            value = result.get("value", {})
            _, _, _, _, adjusted = _normalized_box(value, frame_id=frame_id)
            if adjusted:
                add_issue(
                    severity="warning",
                    code="box_boundary_adjusted",
                    message=(
                        f"Task {frame_id}: a regiao {region_index} excede a borda "
                        "por uma fracao inferior a um pixel."
                    ),
                    resolution=(
                        "Nenhuma acao necessaria; o workflow ajustara a coordenada "
                        "numericamente para a borda da imagem."
                    ),
                    task=task,
                    frame_id=frame_id,
                    region_id=region_id,
                )

    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = len(issues) - error_count
    report = None
    if error_count == 0:
        try:
            _, report = parse_native_export(
                defaults_or_ws,
                campaign_id,
                export_path,
                allow_pending=allow_pending,
            )
        except WorkflowError as exc:
            add_issue(
                severity="error",
                code="validation_error",
                message=str(exc),
                resolution="Revise o problema indicado e exporte o projeto novamente.",
            )
            error_count += 1
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    counts = Counter(issue["code"] for issue in issues)
    return {
        "valid": error_count == 0,
        "errors": error_count,
        "warnings": warning_count,
        "issue_counts": dict(sorted(counts.items())),
        "issues": issues,
        "report": report,
    }


def accept_native_export(
    defaults_or_ws: dict[str, Any] | Workspace,
    campaign_id: str,
    source: Path,
    *,
    revision_id: str | None = None,
    allow_pending: bool = False,
) -> tuple[Path, Path]:
    """Valida, aceita e registra uma revisão final de exportação de anotações.

    Args:
        defaults_or_ws: Instância do Workspace ou dicionário de diretórios.
        campaign_id: Identificador da campanha.
        source: Caminho do arquivo JSON contendo as anotações a aceitar.
        revision_id: Identificador da revisão. Se None, é gerado um timestamp.
        allow_pending: Permite aceitação com tarefas incompletas.

    Returns:
        Tupla com os caminhos da cópia de exportação aceita e seu relatório JSON.
    """

    source = (
        source
        if source.is_absolute()
        else (
            defaults_or_ws.root / source
            if isinstance(defaults_or_ws, Workspace)
            else Path(source)
        )
    )
    _, report = parse_native_export(
        defaults_or_ws, campaign_id, source, allow_pending=allow_pending
    )
    if revision_id is None:
        destination = export_annotations_path(defaults_or_ws, campaign_id)
        report_path = annotation_report_path(defaults_or_ws, campaign_id)
    else:
        validate_id(revision_id, "revision_id")
        destination = annotation_revision_export_path(
            defaults_or_ws, campaign_id, revision_id
        )
        report_path = annotation_revision_report_path(
            defaults_or_ws, campaign_id, revision_id
        )
        report["revision_id"] = revision_id
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise WorkflowError(
                "A revisao ja possui uma exportacao aceita diferente. Use outro revision_id."
            )
        if revision_id is not None and report_path.exists():
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            if bool(existing_report.get("allow_pending", False)) != allow_pending:
                raise WorkflowError(
                    "A revisao ja foi criada com outro modo de tratamento de pendencias. "
                    "Use outro revision_id."
                )
            return destination, report_path
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination, report_path
