"""Regras de domínio e ciclo de vida de origens de dataset (sources)."""

from __future__ import annotations

import hashlib
import json
import shutil
from html import escape
from pathlib import Path
from typing import Any

from dataset_studio.domain.errors import WorkflowError
from dataset_studio.domain.registry import (
    register_source_manifest,
    unregister_source,
)
from dataset_studio.domain.workspace import (
    Workspace,
    dump_yaml,
    label_studio_region_id,
    load_yaml,
    sha256,
    utc_now,
    validate_id,
)


def sources_root(defaults_or_ws: dict[str, Any] | Workspace) -> Path:
    """Retorna o diretório raiz onde todas as fontes de dados/campanhas são armazenadas."""
    if isinstance(defaults_or_ws, Workspace):
        return defaults_or_ws.sources_root
    p = Path(defaults_or_ws["paths"].get("sources_root") or defaults_or_ws["paths"].get("campaigns_root"))
    return p if p.is_absolute() else p.resolve()


def workspace_root(defaults_or_ws: dict[str, Any] | Workspace) -> Path:
    """Retorna o diretório raiz do workspace."""
    if isinstance(defaults_or_ws, Workspace):
        return defaults_or_ws.root
    return sources_root(defaults_or_ws).parent.parent



def source_root(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> Path:
    return sources_root(defaults_or_ws) / source_id


def source_config_path(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> Path:
    p = source_root(defaults_or_ws, source_id)
    if (p / "source.yaml").is_file():
        return p / "source.yaml"
    return p / "campaign.yaml"


def list_sources(defaults_or_ws: dict[str, Any] | Workspace) -> list[str]:
    """Lista os identificadores de todas as fontes registradas no workspace."""
    root = sources_root(defaults_or_ws)
    if not root.exists():
        return []
    return sorted(
        path.name for path in root.iterdir()
        if (path / "source.yaml").is_file() or (path / "campaign.yaml").is_file()
    )


def prediction_profile_from_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Extrai somente parâmetros realmente consumidos pelo backend de predição."""
    detection = dict(payload.get("detection") or {})
    roi = dict(payload.get("roi") or {})
    return {
        "schema_version": 1,
        "detection": {
            key: detection.get(key)
            for key in (
                "conf_threshold",
                "device",
                "img_size",
                "iou_threshold",
                "max_det",
                "half_precision",
            )
            if detection.get(key) is not None
        },
        "roi": {
            "enabled": bool(roi.get("enabled", False)),
            "points": roi.get("points") if isinstance(roi.get("points"), list) else [],
        },
    }


def prediction_profile_sha256(profile: dict[str, Any]) -> str:
    serialized = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_capture_units(
    videos: list[Path],
    capture_units: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Valida segmentos virtuais que representam unidades experimentais."""
    video_names = {video.name for video in videos}
    if not capture_units:
        return [
            {
                "unit_id": video.name,
                "source_video": video.name,
                "start_seconds": 0.0,
                "end_seconds": None,
                "note": "",
                "estimated_subjects": None,
                "condition": None,
            }
            for video in videos
        ]

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    by_video: dict[str, list[dict[str, Any]]] = {}
    for raw in capture_units:
        unit_id = str(raw.get("unit_id") or "").strip()
        validate_id(unit_id, "unit_id")
        if unit_id in seen_ids:
            raise WorkflowError(f"unit_id duplicado: {unit_id}")
        seen_ids.add(unit_id)
        source_video = str(raw.get("source_video") or "")
        if source_video not in video_names:
            raise WorkflowError(
                f"Unidade {unit_id} referencia vídeo não selecionado: {source_video}"
            )
        try:
            start = float(raw.get("start_seconds", 0.0))
            end_value = raw.get("end_seconds")
            end = float(end_value) if end_value not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise WorkflowError(f"Intervalo inválido na unidade {unit_id}.") from exc
        if start < 0 or (end is not None and end <= start):
            raise WorkflowError(
                f"A unidade {unit_id} deve possuir 0 <= início < fim."
            )
        unit = {
            "unit_id": unit_id,
            "source_video": source_video,
            "start_seconds": start,
            "end_seconds": end,
            "note": str(raw.get("note") or ""),
            "estimated_subjects": raw.get("estimated_subjects"),
            "condition": raw.get("condition"),
        }
        normalized.append(unit)
        by_video.setdefault(source_video, []).append(unit)

    missing = sorted(video_names - by_video.keys())
    if missing:
        raise WorkflowError(
            "Todo vídeo selecionado deve possuir ao menos uma unidade de captura: "
            + ", ".join(missing)
        )
    for video_name, units in by_video.items():
        ordered = sorted(units, key=lambda item: item["start_seconds"])
        for current, following in zip(ordered, ordered[1:]):
            if current["end_seconds"] is None:
                raise WorkflowError(
                    f"A unidade aberta {current['unit_id']} impede outro segmento "
                    f"no vídeo {video_name}."
                )
            if current["end_seconds"] > following["start_seconds"]:
                raise WorkflowError(
                    f"As unidades {current['unit_id']} e {following['unit_id']} "
                    f"se sobrepõem no vídeo {video_name}."
                )
    return normalized


def source_capture_units(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Retorna unidades explícitas ou deriva uma unidade por vídeo legado."""
    units = source.get("capture_units")
    if isinstance(units, list) and units:
        return [dict(item) for item in units]
    return [
        {
            "unit_id": item["name"],
            "source_video": item["name"],
            "start_seconds": 0.0,
            "end_seconds": None,
            "note": item.get("note", ""),
            "estimated_subjects": None,
            "condition": None,
        }
        for item in (source.get("videos", {}).get("files") or [])
    ]



def create_source(
    defaults_or_ws: dict[str, Any] | Workspace,
    *,
    source_id: str | None = None,
    campaign_id: str | None = None,
    videos_dir: Path,
    video_pattern: str,
    video_files: list[str] | None = None,
    video_notes: dict[str, str] | None = None,
    capture_units: list[dict[str, Any]] | None = None,
    extraction: dict[str, Any] | None = None,
    annotation: dict[str, Any] | None = None,
) -> Path:
    target_id = source_id or campaign_id
    if not target_id:
        raise WorkflowError("Identificador da origem obrigatorio.")
    validate_id(target_id, "source_id")

    if isinstance(defaults_or_ws, Workspace):
        ws = defaults_or_ws
        defaults = ws.defaults_dict()
    else:
        defaults = defaults_or_ws
        p_root = Path(defaults["paths"].get("sources_root") or defaults["paths"].get("campaigns_root")).parent.parent
        ws = Workspace.from_path(p_root)

    root = source_root(defaults, target_id)
    if root.exists():
        raise WorkflowError(f"A origem ja existe: {root}")

    videos_dir = ws.resolve_path(videos_dir).resolve()
    if video_files:
        if len(video_files) != len(set(video_files)):
            raise WorkflowError("A selecao possui videos duplicados.")
        videos = []
        for name in video_files:
            if Path(name).name != name:
                raise WorkflowError(f"Nome de video invalido: {name}")
            video = (videos_dir / name).resolve()
            if video.parent != videos_dir or not video.is_file():
                raise WorkflowError(f"Video selecionado nao encontrado: {video}")
            videos.append(video)
        videos.sort()
        selection = "explicit"
    else:
        videos = sorted(videos_dir.glob(video_pattern)) if videos_dir.exists() else []
        selection = "pattern"
    if not videos:
        raise WorkflowError(f"Nenhum video foi selecionado em {videos_dir}.")

    extraction_config = dict(extraction or defaults["extraction"])
    mode = extraction_config.get("mode")
    if mode not in {"smart", "uniform"}:
        raise WorkflowError("A origem aceita extraction.mode smart ou uniform.")
    if mode == "uniform":
        extraction_config["model"] = None
        extraction_config["model_sha256"] = None
    else:
        model_value = extraction_config.get("model")
        if not model_value:
            raise WorkflowError("O modo smart exige um modelo de deteccao.")
        model_path = ws.resolve_path(str(model_value)).resolve()
        models_dir = ws.models_root.resolve()
        try:
            model_path.relative_to(models_dir)
        except ValueError as exc:
            raise WorkflowError(
                f"Copie o modelo candidato para {models_dir} antes de usa-lo."
            ) from exc
        if not model_path.is_file():
            raise WorkflowError(f"Modelo de extracao nao encontrado: {model_path}")
        extraction_config["model_sha256"] = sha256(model_path)

    annotation_config = dict(annotation or defaults["annotation"])
    annotation_backend = annotation_config.get("backend") or (
        "local" if annotation_config.get("model") else "none"
    )
    if annotation_backend not in {"none", "local"}:
        raise WorkflowError(
            "A origem aceita annotation.backend none ou local."
        )
    annotation_config["backend"] = annotation_backend
    annotation_model = annotation_config.get("model")
    if annotation_backend == "local":
        if not annotation_model:
            raise WorkflowError(
                "O servidor auxiliar local exige um modelo em models/."
            )
        annotation_model_path = ws.resolve_path(str(annotation_model)).resolve()
        models_dir = ws.models_root.resolve()
        try:
            annotation_model_path.relative_to(models_dir)
        except ValueError as exc:
            raise WorkflowError(
                f"Copie o modelo auxiliar para {models_dir} antes de usa-lo."
            ) from exc
        if not annotation_model_path.is_file():
            raise WorkflowError(
                f"Modelo auxiliar de anotacao nao encontrado: {annotation_model_path}"
            )
        annotation_config["model_sha256"] = sha256(annotation_model_path)
        detection_config_value = annotation_config.get("detection_config")
        if not detection_config_value:
            raise WorkflowError(
                "O servidor auxiliar local exige um perfil YAML de deteccao."
            )
        detection_config_path = ws.resolve_path(str(detection_config_value)).resolve()
        config_dir = ws.config_root.resolve()
        try:
            detection_config_path.relative_to(config_dir)
        except ValueError as exc:
            raise WorkflowError(
                f"Use um perfil de deteccao presente em {config_dir}."
            ) from exc
        if not detection_config_path.is_file():
            raise WorkflowError(
                f"Perfil de deteccao nao encontrado: {detection_config_path}"
            )
        detection_payload = load_yaml(detection_config_path)
        if not isinstance(detection_payload.get("detection"), dict):
            raise WorkflowError(
                f"YAML sem secao detection: {detection_config_path}"
            )
        annotation_config["detection_config_sha256"] = sha256(
            detection_config_path
        )
        profile = prediction_profile_from_config(detection_payload)
        annotation_config["prediction_profile"] = profile
        annotation_config["prediction_profile_sha256"] = (
            prediction_profile_sha256(profile)
        )
    else:
        annotation_config["model"] = None
        annotation_config["model_sha256"] = None
        annotation_config["detection_config"] = None
        annotation_config["detection_config_sha256"] = None
        annotation_config["prediction_profile"] = None
        annotation_config["prediction_profile_sha256"] = None

    (root / "frames" / "raw" / "images").mkdir(parents=True)
    (root / "label_studio").mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "source_id": target_id,
        "campaign_id": target_id,
        "created_at": utc_now(),
        "videos": {
            "directory": str(videos_dir),
            "selection": selection,
            "pattern": video_pattern if selection == "pattern" else None,
            "files": [
                {
                    "name": video.name,
                    "size": video.stat().st_size,
                    "mtime_ns": video.stat().st_mtime_ns,
                    "sha256": sha256(video),
                    "note": (video_notes or {}).get(video.name, ""),
                }
                for video in videos
            ],
        },
        "capture_units": normalize_capture_units(videos, capture_units),
        "extraction": extraction_config,
        "annotation": annotation_config,
    }
    config_path = root / "source.yaml"
    dump_yaml(config_path, payload)
    labels_xml = "\n".join(
        f'    <Label value="{escape(str(name))}"/>'
        for name in payload["annotation"]["classes"]
    )
    labeling_config = (
        "<View>\n"
        "  <Image name=\"image\" value=\"$image\"/>\n"
        "  <RectangleLabels name=\"label\" toName=\"image\">\n"
        f"{labels_xml}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )
    (root / "label_studio" / "labeling_config.xml").write_text(
        labeling_config, encoding="utf-8"
    )
    register_source_manifest(ws, target_id, config_path)
    return config_path


def load_source(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> dict[str, Any]:
    """Carrega o arquivo YAML de configuração de uma fonte específica."""
    path = source_config_path(defaults_or_ws, source_id)
    payload = load_yaml(path)
    if payload.get("source_id") != source_id and payload.get("campaign_id") != source_id:
        raise WorkflowError("source.yaml possui identificador divergente.")
    annotation = payload.get("annotation") or {}
    profile = annotation.get("prediction_profile")
    expected_hash = annotation.get("prediction_profile_sha256")
    if profile is not None:
        actual_hash = prediction_profile_sha256(profile)
        if not expected_hash or actual_hash != expected_hash:
            raise WorkflowError(
                "O perfil de predição congelado da origem foi alterado."
            )
    return payload



def frame_manifest_path(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> Path:
    return source_root(defaults_or_ws, source_id) / "frame_manifest.json"


def import_tasks_path(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> Path:
    return source_root(defaults_or_ws, source_id) / "label_studio" / "import_tasks.json"


def selected_export_path(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> Path:
    return source_root(defaults_or_ws, source_id) / "label_studio" / "export_selected.json"


def save_selected_export(
    defaults_or_ws: dict[str, Any] | Workspace, source_id: str, content: bytes
) -> Path:
    load_source(defaults_or_ws, source_id)
    try:
        text = content.decode("utf-8-sig")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowError("O arquivo selecionado nao e um JSON valido.") from exc
    if not isinstance(payload, list):
        raise WorkflowError("A exportacao nativa deve ser uma lista de tasks.")
    destination = selected_export_path(defaults_or_ws, source_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return destination


def export_annotations_path(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> Path:
    return source_root(defaults_or_ws, source_id) / "label_studio" / "export_annotations.json"


def annotation_report_path(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> Path:
    return source_root(defaults_or_ws, source_id) / "label_studio" / "annotation_report.json"


def annotation_revisions_root(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> Path:
    return source_root(defaults_or_ws, source_id) / "label_studio" / "revisions"


def annotation_revision_root(
    defaults_or_ws: dict[str, Any] | Workspace, source_id: str, revision_id: str
) -> Path:
    validate_id(revision_id, "revision_id")
    return annotation_revisions_root(defaults_or_ws, source_id) / revision_id


def annotation_revision_export_path(
    defaults_or_ws: dict[str, Any] | Workspace, source_id: str, revision_id: str
) -> Path:
    return (
        annotation_revision_root(defaults_or_ws, source_id, revision_id)
        / "export_annotations.json"
    )


def annotation_revision_report_path(
    defaults_or_ws: dict[str, Any] | Workspace, source_id: str, revision_id: str
) -> Path:
    return (
        annotation_revision_root(defaults_or_ws, source_id, revision_id)
        / "annotation_report.json"
    )


def list_annotation_revisions(
    defaults_or_ws: dict[str, Any] | Workspace, source_id: str
) -> list[str]:
    root = annotation_revisions_root(defaults_or_ws, source_id)
    revision_paths = (
        [
            path
            for path in root.iterdir()
            if path.is_dir()
            and (path / "export_annotations.json").is_file()
            and (path / "annotation_report.json").is_file()
        ]
        if root.exists()
        else []
    )

    def revision_order(path: Path) -> tuple[str, str]:
        """Ordena pela criação real da revisão, não pelo nome escolhido."""
        try:
            report = json.loads(
                (path / "annotation_report.json").read_text(encoding="utf-8")
            )
            validated_at = str(report.get("validated_at") or "")
        except (OSError, json.JSONDecodeError):
            validated_at = ""
        return validated_at, path.name

    revisions = [path.name for path in sorted(revision_paths, key=revision_order)]
    if (
        export_annotations_path(defaults_or_ws, source_id).exists()
        and annotation_report_path(defaults_or_ws, source_id).exists()
    ):
        revisions.insert(0, "legacy")
    return revisions


def annotation_source_paths(
    defaults_or_ws: dict[str, Any] | Workspace, source_id: str, revision_id: str
) -> tuple[Path, Path]:
    if revision_id == "legacy":
        return (
            export_annotations_path(defaults_or_ws, source_id),
            annotation_report_path(defaults_or_ws, source_id),
        )
    return (
        annotation_revision_export_path(defaults_or_ws, source_id, revision_id),
        annotation_revision_report_path(defaults_or_ws, source_id, revision_id),
    )


def load_annotation_revision_report(
    defaults_or_ws: dict[str, Any] | Workspace, source_id: str, revision_id: str
) -> dict[str, Any]:
    export_path, report_path = annotation_source_paths(
        defaults_or_ws, source_id, revision_id
    )
    if not export_path.exists() or not report_path.exists():
        raise WorkflowError(
            f"Revisao de anotacao nao encontrada: {source_id}/{revision_id}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise WorkflowError(f"Relatorio de revisao invalido: {report_path}")
    return report


def load_frame_manifest(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> dict[str, Any]:
    path = frame_manifest_path(defaults_or_ws, source_id)
    if not path.exists():
        raise WorkflowError(f"Extraia os frames antes de continuar: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list) or not frames:
        raise WorkflowError(f"Manifesto sem frames: {path}")
    ids = [frame.get("frame_id") for frame in frames if isinstance(frame, dict)]
    if len(ids) != len(frames) or len(ids) != len(set(ids)):
        raise WorkflowError("frame_manifest.json possui frame_id ausente ou duplicado.")
    return payload


def yolo_prediction_to_ls(
    prediction: dict[str, Any],
    *,
    frame: dict[str, Any],
    class_names: list[str],
    index: int,
) -> dict[str, Any]:
    """Converte uma predição no formato YOLO para a estrutura de resultados do Label Studio.

    Args:
        prediction: Dicionário contendo a predição (class_id, xc, yc, width, height).
        frame: Dados do frame (frame_id, width, height).
        class_names: Lista dos nomes de classe disponíveis.
        index: Índice da predição dentro do frame.

    Returns:
        Dicionário formatado no esquema de resultados do Label Studio.
    """

    class_id = int(prediction["class_id"])
    if class_id < 0 or class_id >= len(class_names):
        raise WorkflowError(f"class_id invalido na predicao: {class_id}")
    width = float(prediction["width"])
    height = float(prediction["height"])
    return {
        "id": label_studio_region_id(frame["frame_id"], index),
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "original_width": int(frame["width"]),
        "original_height": int(frame["height"]),
        "image_rotation": 0,
        "value": {
            "x": round((float(prediction["xc"]) - width / 2) * 100, 6),
            "y": round((float(prediction["yc"]) - height / 2) * 100, 6),
            "width": round(width * 100, 6),
            "height": round(height * 100, 6),
            "rotation": 0,
            "rectanglelabels": [class_names[class_id]],
        },
    }


def build_import_tasks(
    defaults_or_ws: dict[str, Any] | Workspace,
    source_id: str,
    *,
    include_predictions: bool = True,
) -> Path:
    """Gera o arquivo import_tasks.json com as pré-anotações formatadas para o Label Studio.

    Args:
        defaults_or_ws: Instância do Workspace ou dicionário de configurações.
        source_id: Identificador da fonte de dados.

    Returns:
        Caminho do arquivo import_tasks.json gerado.
    """

    output = import_tasks_path(defaults_or_ws, source_id)
    if output.exists():
        raise WorkflowError(
            "import_tasks.json ja foi gerado; a origem esta fixada e nao pode ser alterada."
        )
    source = load_source(defaults_or_ws, source_id)
    manifest = load_frame_manifest(defaults_or_ws, source_id)
    root = source_root(defaults_or_ws, source_id)
    ws_root = workspace_root(defaults_or_ws)
    class_names = list(source["annotation"]["classes"])
    tasks = []
    for frame in manifest["frames"]:
        image_path = root / "frames" / "raw" / "images" / frame["image"]
        if not image_path.exists():
            raise WorkflowError(f"Imagem do manifesto nao encontrada: {image_path}")
        relative = image_path.resolve().relative_to(ws_root.resolve()).as_posix()
        results = [
            yolo_prediction_to_ls(
                prediction, frame=frame, class_names=class_names, index=index
            )
            for index, prediction in enumerate(
                frame.get("predictions", []) if include_predictions else []
            )
        ]
        task = {
            "data": {
                "image": f"/data/local-files/?d={relative}",
                "frame_id": frame["frame_id"],
                "source_id": source_id,
                "campaign_id": source_id,
                "source_video": frame["source_video"],
                "unit_id": frame.get("unit_id") or frame["source_video"],
                "frame_index": frame["frame_index"],
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "unit_timestamp_seconds": frame.get("unit_timestamp_seconds"),
                "original_width": frame["width"],
                "original_height": frame["height"],
            },
            "predictions": ([
                {
                    "model_version": (manifest.get("model_sha256") or "no-model")[:12],
                    "score": 0.0,
                    "result": results,
                }
            ] if include_predictions else []),
        }
        tasks.append(task)
    output.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def delete_source(
    defaults_or_ws: dict[str, Any] | Workspace,
    source_id: str,
    *,
    delete_video_files: bool = False,
) -> None:
    """Remove completamente a pasta da origem de dados do disco."""
    root = source_root(defaults_or_ws, source_id)
    if not root.exists():
        raise WorkflowError(f"Origem não encontrada: {source_id}")
    source = load_source(defaults_or_ws, source_id)
    isolated_videos_dir: Path | None = None
    selected_video_paths: list[Path] = []
    if isinstance(defaults_or_ws, Workspace):
        candidate = defaults_or_ws.resolve_path(source["videos"]["directory"]).resolve()
        for item in source["videos"].get("files", []):
            name = item["name"] if isinstance(item, dict) else str(item)
            path = (candidate / name).resolve()
            if path.parent == candidate:
                selected_video_paths.append(path)
        expected = (defaults_or_ws.videos_root / source_id).resolve()
        if candidate == expected:
            isolated_videos_dir = candidate
    shutil.rmtree(root, ignore_errors=False)
    ws = (
        defaults_or_ws
        if isinstance(defaults_or_ws, Workspace)
        else Workspace.from_path(workspace_root(defaults_or_ws))
    )
    unregister_source(ws, source_id)
    if isolated_videos_dir is not None and isolated_videos_dir.is_dir():
        shutil.rmtree(isolated_videos_dir, ignore_errors=False)
    elif delete_video_files:
        for video_path in selected_video_paths:
            video_path.unlink(missing_ok=True)


# Aliases de retrocompatibilidade para campaign -> source
campaigns_root = sources_root
campaign_root = source_root
campaign_config_path = source_config_path
list_campaigns = list_sources
create_campaign = create_source
delete_campaign = delete_source
load_campaign = load_source
