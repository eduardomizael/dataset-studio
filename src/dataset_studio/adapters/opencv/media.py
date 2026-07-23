"""Adaptador OpenCV para leitura de vídeos e extração de frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2

from dataset_studio.adapters.ultralytics.predictor import UltralyticsPredictor
from dataset_studio.domain import (
    WorkflowError,
    Workspace,
    frame_manifest_path,
    load_source,
    sha256,
    source_capture_units,
)

ProgressCallback = Callable[[dict[str, Any]], None]


def _report_progress(
    callback: ProgressCallback | None,
    *,
    fraction: float,
    stage: str,
    message: str,
    frames_saved: int,
) -> None:
    if callback is not None:
        callback(
            {
                "fraction": max(0.0, min(1.0, fraction)),
                "stage": stage,
                "message": message,
                "frames_saved": frames_saved,
            }
        )


def format_file_size(size_bytes: int) -> str:
    """Formata um tamanho em bytes para uma string legível (ex: KB, MB, GB)."""

    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    val = float(size_bytes)
    while val >= 1024.0 and i < len(units) - 1:
        val /= 1024.0
        i += 1
    if i == 0:
        return f"{int(val)} B"
    return f"{val:.1f} {units[i]}"


def get_video_info(video_path: Path, fallback_size: int = 0) -> dict[str, Any]:
    """Obtém metadados do arquivo de vídeo (dimensões, FPS, tamanho legível)."""

    name = video_path.name
    size_bytes = video_path.stat().st_size if video_path.is_file() else fallback_size
    width = 0
    height = 0
    fps = 0.0

    if video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

    return {
        "name": name,
        "size_bytes": size_bytes,
        "size_human": format_file_size(size_bytes),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}" if width and height else "N/A",
        "fps": round(fps, 2) if fps else 0.0,
    }


def xyxy_to_yolo(box: tuple[float, float, float, float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Converte coordenadas em pixels (x1, y1, x2, y2) para o formato YOLO normalizado (xc, yc, w, h)."""

    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x_c = max(0.0, min(1.0, (x1 + w / 2.0) / img_w))
    y_c = max(0.0, min(1.0, (y1 + h / 2.0) / img_h))
    nw = max(0.0, min(1.0, w / img_w))
    nh = max(0.0, min(1.0, h / img_h))
    return x_c, y_c, nw, nh


def save_frame(
    frame,
    detections: list[tuple[int, tuple]],
    frame_name: str,
    images_out: Path,
    labels_out: Path | None,
) -> None:
    """Salva a imagem do frame no disco e cria o arquivo de rótulos YOLO se fornecido."""

    cv2.imwrite(str(images_out / f"{frame_name}.jpg"), frame)
    if labels_out is not None:
        with open(labels_out / f"{frame_name}.txt", "w", encoding="utf-8") as f:
            for cls_id, (xc, yc, w, h) in detections:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def prediction_record(
    *,
    frame_name: str,
    source_video: str,
    frame_index: int,
    unit_id: str | None = None,
    timestamp_seconds: float | None = None,
    unit_timestamp_seconds: float | None = None,
    frame,
    detections: list[tuple[int, tuple]],
) -> dict[str, Any]:
    """Constrói o dicionário de registro de predições e metadados de um frame."""

    height, width = frame.shape[:2]
    return {
        "frame_id": frame_name,
        "image": f"{frame_name}.jpg",
        "source_video": source_video,
        "unit_id": unit_id or source_video,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "unit_timestamp_seconds": unit_timestamp_seconds,
        "width": width,
        "height": height,
        "predictions": [
            {
                "class_id": class_id,
                "xc": round(float(box[0]), 6),
                "yc": round(float(box[1]), 6),
                "width": round(float(box[2]), 6),
                "height": round(float(box[3]), 6),
            }
            for class_id, box in detections
        ],
    }


def scan_video(
    video_path: Path,
    predictor: UltralyticsPredictor,
    scan_step: int,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[int]:
    """Escaneia um vídeo com amostragem para detectar a presença de alvos (peixes)."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fish_frames: list[int] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    effective_end = end_frame or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, (effective_end - start_frame) // 100)
    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % scan_step == 0:
            dets = predictor.predict(frame)
            if dets:
                fish_frames.append(frame_idx)
        if (frame_idx - start_frame) % interval == 0:
            _report_progress(
                progress_callback,
                fraction=(frame_idx - start_frame) / max(1, effective_end - start_frame),
                stage="scanning",
                message="Localizando trechos com objetos.",
                frames_saved=0,
            )
        frame_idx += 1
    cap.release()
    return fish_frames


def find_fish_ranges(fish_frames: list[int], margin: int, total_frames: int) -> list[tuple[int, int]]:
    """Agrupa os quadros com presença detectada em intervalos temporais contínuos com margem."""

    if not fish_frames:
        return []
    ranges: list[tuple[int, int]] = []
    start = max(0, fish_frames[0] - margin)
    end = min(total_frames, fish_frames[0] + margin)
    for f in fish_frames[1:]:
        f_start = max(0, f - margin)
        f_end = min(total_frames, f + margin)
        if f_start <= end:
            end = f_end
        else:
            ranges.append((start, end))
            start, end = f_start, f_end
    ranges.append((start, end))
    return ranges


def is_in_ranges(frame_idx: int, ranges: list[tuple[int, int]]) -> bool:
    """Verifica se o índice do frame está contido em algum dos intervalos selecionados."""

    return any(s <= frame_idx <= e for s, e in ranges)


def run_uniform_mode(
    video_path: Path,
    frame_step: int,
    images_out: Path,
    labels_out: Path | None,
    records: list[dict],
    *,
    unit_id: str | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Executa a extração no modo uniforme com passo de amostragem constante."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"saved": 0, "analyzed": 0}
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(round(start_seconds * fps)))
    end_frame = (
        min(total_frames, int(round(end_seconds * fps)))
        if end_seconds is not None
        else total_frames
    )
    if start_frame >= total_frames or start_frame >= end_frame:
        cap.release()
        raise WorkflowError(
            f"A unidade {unit_id or video_path.name} está fora da duração do vídeo."
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    video_stem = video_path.stem if not unit_id or unit_id == video_path.name else unit_id
    frame_idx = start_frame
    saved = 0
    analyzed = 0
    interval = max(1, (end_frame - start_frame) // 100)
    while True:
        if frame_idx >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_step == 0:
            analyzed += 1
            frame_name = f"{video_stem}_f{frame_idx:06d}"
            save_frame(frame, [], frame_name, images_out, labels_out)
            records.append(
                prediction_record(
                    frame_name=frame_name,
                    source_video=video_path.name,
                    frame_index=frame_idx,
                    unit_id=unit_id,
                    timestamp_seconds=frame_idx / fps,
                    unit_timestamp_seconds=(frame_idx - start_frame) / fps,
                    frame=frame,
                    detections=[],
                )
            )
            saved += 1
        if (frame_idx - start_frame) % interval == 0:
            _report_progress(
                progress_callback,
                fraction=(frame_idx - start_frame) / max(1, end_frame - start_frame),
                stage="extracting",
                message="Extraindo frames em intervalo uniforme.",
                frames_saved=saved,
            )
        frame_idx += 1
    cap.release()
    _report_progress(
        progress_callback,
        fraction=1.0,
        stage="extracting",
        message="Unidade extraída.",
        frames_saved=saved,
    )
    return {"saved": saved, "analyzed": analyzed}


def _formatted_detections(predictor: UltralyticsPredictor, frame) -> list[tuple[int, tuple]]:
    height, width = frame.shape[:2]
    return [
        (item.class_id, xyxy_to_yolo(item.bbox_xyxy, width, height))
        for item in predictor.predict(frame)
    ]


def run_smart_mode(
    video_path: Path,
    predictor: UltralyticsPredictor,
    *,
    scan_step: int,
    dense_step: int,
    sparse_step: int,
    margin: int,
    max_negatives_per_video: int,
    images_out: Path,
    records: list[dict],
    unit_id: str | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Extrai mais frames nas regiões com objetos e negativos esparsos fora delas."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fish": 0, "negative": 0, "analyzed": 0}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    start_frame = max(0, int(round(start_seconds * fps)))
    end_frame = (
        min(total_frames, int(round(end_seconds * fps)))
        if end_seconds is not None
        else total_frames
    )
    if start_frame >= total_frames or start_frame >= end_frame:
        cap.release()
        raise WorkflowError(
            f"A unidade {unit_id or video_path.name} está fora da duração do vídeo."
        )
    cap.release()

    detected_frames = scan_video(
        video_path,
        predictor,
        scan_step,
        start_frame=start_frame,
        end_frame=end_frame,
        progress_callback=(
            (
                lambda update: progress_callback(
                    {
                        **update,
                        "fraction": float(update.get("fraction", 0.0)) * 0.45,
                    }
                )
            )
            if progress_callback
            else None
        ),
    )
    detected_ranges = find_fish_ranges(detected_frames, margin, end_frame)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    negative_count = 0
    stats = {"fish": 0, "negative": 0, "analyzed": 0}
    interval = max(1, (end_frame - start_frame) // 100)
    while True:
        if frame_idx >= end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        in_detection_range = is_in_ranges(frame_idx, detected_ranges)
        should_extract = (
            in_detection_range and (frame_idx - start_frame) % dense_step == 0
        ) or (
            not in_detection_range
            and (frame_idx - start_frame) % sparse_step == 0
            and negative_count < max_negatives_per_video
        )
        if should_extract:
            detections = _formatted_detections(predictor, frame)
            frame_prefix = (
                video_path.stem
                if not unit_id or unit_id == video_path.name
                else unit_id
            )
            frame_name = f"{frame_prefix}_f{frame_idx:06d}"
            save_frame(frame, detections, frame_name, images_out, None)
            records.append(
                prediction_record(
                    frame_name=frame_name,
                    source_video=video_path.name,
                    frame_index=frame_idx,
                    unit_id=unit_id,
                    timestamp_seconds=frame_idx / fps,
                    unit_timestamp_seconds=(frame_idx - start_frame) / fps,
                    frame=frame,
                    detections=detections,
                )
            )
            stats["analyzed"] += 1
            if detections:
                stats["fish"] += 1
            else:
                stats["negative"] += 1
                negative_count += 1
        if (frame_idx - start_frame) % interval == 0:
            _report_progress(
                progress_callback,
                fraction=0.45
                + 0.55
                * (frame_idx - start_frame)
                / max(1, end_frame - start_frame),
                stage="extracting",
                message="Extraindo e classificando frames selecionados.",
                frames_saved=stats["fish"] + stats["negative"],
            )
        frame_idx += 1
    cap.release()
    _report_progress(
        progress_callback,
        fraction=1.0,
        stage="extracting",
        message="Unidade extraída.",
        frames_saved=stats["fish"] + stats["negative"],
    )
    return stats


def extract_source_frames(
    defaults_or_ws: dict[str, Any] | Workspace,
    source_id: str,
    *,
    weights_path: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Executa o pipeline de extração de frames (Uniforme ou Inteligente com YOLO) para uma fonte de dados."""

    ws = (
        defaults_or_ws
        if isinstance(defaults_or_ws, Workspace)
        else Workspace.from_path(Path(defaults_or_ws["paths"]["sources_root"]).resolve().parent.parent)
    )
    manifest_path = frame_manifest_path(ws, source_id)
    root = ws.source_root(source_id)
    images_out = root / "frames" / "raw" / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    source_data = load_source(ws, source_id)
    extraction = source_data["extraction"]
    mode = extraction.get("mode", "uniform")
    videos_dir = ws.resolve_path(source_data["videos"]["directory"])
    records: list[dict] = []

    predictor = None
    model_path = weights_path
    if mode == "smart":
        model_path = model_path or ws.resolve_path(extraction["model"])
        predictor = UltralyticsPredictor(
            model_path,
            conf=float(extraction.get("confidence", extraction.get("confidence_threshold", 0.25))),
            device=extraction.get("device"),
        )
        try:
            predictor.load()
        except ImportError as exc:
            raise WorkflowError(
                "Ultralytics nao esta instalado. Execute: uv sync --all-extras"
            ) from exc

    units = source_capture_units(source_data)
    total_units = len(units)
    total_saved = 0
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "initializing",
                "message": f"Preparando {total_units} unidade(s) de captura.",
                "percent": 0,
                "processed_units": 0,
                "total_units": total_units,
                "frames_saved": 0,
            }
        )

    for unit_index, unit in enumerate(units):
        video_path = videos_dir / unit["source_video"]
        if video_path.is_file():
            def report_unit(update: dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                fraction = float(update.get("fraction", 0.0))
                progress_callback(
                    {
                        "stage": update.get("stage", "extracting"),
                        "message": (
                            f"{unit['unit_id']}: "
                            f"{update.get('message', 'Extraindo frames.')}"
                        ),
                        "percent": int(
                            ((unit_index + fraction) / max(1, total_units)) * 100
                        ),
                        "processed_units": unit_index,
                        "total_units": total_units,
                        "current_unit": unit["unit_id"],
                        "frames_saved": total_saved
                        + int(update.get("frames_saved", 0)),
                    }
                )

            if mode == "smart":
                assert predictor is not None
                stats = run_smart_mode(
                    video_path,
                    predictor,
                    scan_step=int(extraction.get("scan_step", 15)),
                    dense_step=int(extraction.get("dense_step", 30)),
                    sparse_step=int(extraction.get("sparse_step", 90)),
                    margin=int(extraction.get("margin", 45)),
                    max_negatives_per_video=int(extraction.get("max_negatives_per_video", 15)),
                    images_out=images_out,
                    records=records,
                    unit_id=unit["unit_id"],
                    start_seconds=float(unit.get("start_seconds") or 0.0),
                    end_seconds=unit.get("end_seconds"),
                    progress_callback=report_unit,
                )
                total_saved += stats["fish"] + stats["negative"]
            else:
                frame_step = int(
                    extraction.get("uniform_frame_step", extraction.get("frame_step", 30))
                )
                stats = run_uniform_mode(
                    video_path,
                    frame_step,
                    images_out,
                    None,
                    records,
                    unit_id=unit["unit_id"],
                    start_seconds=float(unit.get("start_seconds") or 0.0),
                    end_seconds=unit.get("end_seconds"),
                    progress_callback=report_unit,
                )
                total_saved += stats["saved"]
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "extracting",
                        "message": f"Unidade {unit['unit_id']} concluída.",
                        "percent": int(((unit_index + 1) / max(1, total_units)) * 100),
                        "processed_units": unit_index + 1,
                        "total_units": total_units,
                        "current_unit": unit["unit_id"],
                        "frames_saved": total_saved,
                    }
                )

    records_by_id = {str(item["frame_id"]): item for item in records}
    payload = {
        "schema_version": 2,
        "model": str(extraction.get("model")) if extraction.get("model") else None,
        "model_sha256": sha256(model_path) if model_path and model_path.is_file() else None,
        "confidence": extraction.get("confidence", extraction.get("confidence_threshold")),
        "mode": mode,
        "video_pattern": source_data["videos"].get("pattern"),
        "video_files": [f["name"] for f in source_data["videos"]["files"]],
        "capture_units": source_capture_units(source_data),
        "frame_step": extraction.get("uniform_frame_step", extraction.get("frame_step")),
        "frames": [records_by_id[key] for key in sorted(records_by_id)],
    }
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "finalizing",
                "message": "Gravando o manifesto da extração.",
                "percent": 99,
                "processed_units": total_units,
                "total_units": total_units,
                "frames_saved": total_saved,
            }
        )
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


extract_campaign_frames = extract_source_frames


def preannotate_source_frames(
    ws: Workspace,
    source_id: str,
    model_path: Path,
    *,
    confidence: float = 0.25,
    device: str | None = None,
) -> Path:
    """Gera sugestões para todos os frames antes da fixação do import_tasks.json."""
    manifest_path = frame_manifest_path(ws, source_id)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifesto de frames nao encontrado: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictor = UltralyticsPredictor(model_path, conf=confidence, device=device)
    try:
        predictor.load()
    except ImportError as exc:
        raise WorkflowError(
            "Ultralytics nao esta instalado. Execute: uv sync --all-extras"
        ) from exc
    images_root = ws.source_root(source_id) / "frames" / "raw" / "images"
    for frame_record in payload.get("frames", []):
        image = cv2.imread(str(images_root / frame_record["image"]))
        if image is None:
            raise FileNotFoundError(f"Frame nao encontrado: {frame_record['image']}")
        detections = _formatted_detections(predictor, image)
        frame_record["predictions"] = [
            {
                "class_id": class_id,
                "xc": round(float(box[0]), 6),
                "yc": round(float(box[1]), 6),
                "width": round(float(box[2]), 6),
                "height": round(float(box[3]), 6),
            }
            for class_id, box in detections
        ]
    payload["model"] = str(model_path)
    payload["model_sha256"] = sha256(model_path)
    payload["confidence"] = confidence
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path

