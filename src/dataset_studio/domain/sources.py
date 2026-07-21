"""Regras de domínio e ciclo de vida de origens de dataset (sources)."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from dataset_studio.domain.errors import WorkflowError
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



def create_source(
    defaults_or_ws: dict[str, Any] | Workspace,
    *,
    source_id: str | None = None,
    campaign_id: str | None = None,
    videos_dir: Path,
    video_pattern: str,
    video_files: list[str] | None = None,
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
    else:
        annotation_config["model"] = None
        annotation_config["model_sha256"] = None
        annotation_config["detection_config"] = None
        annotation_config["detection_config_sha256"] = None

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
                }
                for video in videos
            ],
        },
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
    return config_path


def load_source(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> dict[str, Any]:
    """Carrega o arquivo YAML de configuração de uma fonte específica."""
    path = source_config_path(defaults_or_ws, source_id)
    payload = load_yaml(path)
    if payload.get("source_id") != source_id and payload.get("campaign_id") != source_id:
        raise WorkflowError("source.yaml possui identificador divergente.")
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
    revisions = (
        sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and (path / "export_annotations.json").is_file()
            and (path / "annotation_report.json").is_file()
        )
        if root.exists()
        else []
    )
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


def build_import_tasks(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> Path:
    """Gera o arquivo import_tasks.json com as pré-anotações formatadas para o Label Studio.

    Args:
        defaults_or_ws: Instância do Workspace ou dicionário de configurações.
        source_id: Identificador da fonte de dados.

    Returns:
        Caminho do arquivo import_tasks.json gerado.
    """

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
            for index, prediction in enumerate(frame.get("predictions", []))
        ]
        task = {
            "data": {
                "image": f"/data/local-files/?d={relative}",
                "frame_id": frame["frame_id"],
                "source_id": source_id,
                "campaign_id": source_id,
                "source_video": frame["source_video"],
                "frame_index": frame["frame_index"],
                "original_width": frame["width"],
                "original_height": frame["height"],
            },
            "predictions": [
                {
                    "model_version": (manifest.get("model_sha256") or "no-model")[:12],
                    "score": 0.0,
                    "result": results,
                }
            ],
        }
        tasks.append(task)
    output = import_tasks_path(defaults_or_ws, source_id)
    output.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


# Aliases de retrocompatibilidade para campaign -> source
campaigns_root = sources_root
campaign_root = source_root
campaign_config_path = source_config_path
list_campaigns = list_sources
create_campaign = create_source
load_campaign = load_source
