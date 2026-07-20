"""Adaptador OpenCV para leitura de vídeos e extração de frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2

from dataset_studio.adapters.ultralytics.predictor import UltralyticsPredictor
from dataset_studio.domain import Workspace, frame_manifest_path, load_campaign


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
    frame,
    detections: list[tuple[int, tuple]],
) -> dict[str, Any]:
    """Constrói o dicionário de registro de predições e metadados de um frame."""

    height, width = frame.shape[:2]
    return {
        "frame_id": frame_name,
        "image": f"{frame_name}.jpg",
        "source_video": source_video,
        "frame_index": frame_index,
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


def scan_video(video_path: Path, predictor: UltralyticsPredictor, scan_step: int) -> list[int]:
    """Escaneia um vídeo com amostragem para detectar a presença de alvos (peixes)."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fish_frames: list[int] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % scan_step == 0:
            dets = predictor.predict(frame)
            if dets:
                fish_frames.append(frame_idx)
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
) -> dict[str, int]:
    """Executa a extração no modo uniforme com passo de amostragem constante."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"saved": 0, "analyzed": 0}
    video_stem = video_path.stem
    frame_idx = 0
    saved = 0
    analyzed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            analyzed += 1
            frame_name = f"{video_stem}_f{frame_idx:06d}"
            save_frame(frame, [], frame_name, images_out, labels_out)
            records.append(
                prediction_record(
                    frame_name=frame_name,
                    source_video=video_path.name,
                    frame_index=frame_idx,
                    frame=frame,
                    detections=[],
                )
            )
            saved += 1
        frame_idx += 1
    cap.release()
    return {"saved": saved, "analyzed": analyzed}


def extract_source_frames(
    defaults_or_ws: dict[str, Any] | Workspace,
    campaign_id: str,
    *,
    weights_path: Path | None = None,
) -> dict[str, Any]:
    """Executa o pipeline de extração de frames (Uniforme ou Inteligente com YOLO) para uma fonte de dados."""

    source = ws.source_root(source_id)
    manifest_path = frame_manifest_path(ws, source_id)
    root = ws.source_root(source_id)
    images_out = root / "frames" / "raw" / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    from dataset_studio.domain.sources import load_source
    source_data = load_source(ws, source_id)
    videos_dir = ws.resolve_path(source_data["videos"]["directory"])
    records: list[dict] = []

    for v_file in source_data["videos"]["files"]:
        video_path = videos_dir / v_file["name"]
        if video_path.is_file():
            run_uniform_mode(video_path, frame_step, images_out, None, records)

    records_by_id = {str(item["frame_id"]): item for item in records}
    payload = {
        "schema_version": 1,
        "model": None,
        "model_sha256": None,
        "confidence": None,
        "mode": "uniform",
        "video_pattern": source_data["videos"].get("pattern"),
        "video_files": [f["name"] for f in source_data["videos"]["files"]],
        "frame_step": frame_step,
        "frames": [records_by_id[key] for key in sorted(records_by_id)],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


extract_campaign_frames = extract_source_frames

