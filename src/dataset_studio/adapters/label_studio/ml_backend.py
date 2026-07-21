"""Servidor de predição ML Backend neutro para Label Studio."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import yaml

from dataset_studio.domain.workspace import label_studio_region_id
from dataset_studio.ports.predictor import Detection, Predictor

DEFAULT_FROM_NAME = "label"
DEFAULT_TO_NAME = "image"
DEFAULT_VALUE_FIELD = "image"


def load_class_names(path: str | Path | None) -> list[str]:
    """Carrega os nomes de classes a partir de um arquivo YAML de configuração.

    Args:
        path: Caminho para o arquivo YAML (ex: data.yaml).

    Returns:
        Lista com os nomes das classes configuradas.
    """

    if not path:
        return ["objeto"]
    p = Path(path)
    if not p.exists():
        return ["objeto"]
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, list):
            return [str(n) for n in names]
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
    except Exception:
        pass
    return ["objeto"]


def read_image_from_task(task: dict[str, Any], value_field: str, default_root: Path) -> np.ndarray:
    """Lê e decodifica a imagem associada a uma tarefa do Label Studio.

    Args:
        task: Dicionário contendo a estrutura de dados da tarefa.
        value_field: Nome do campo que armazena a referência da imagem.
        default_root: Diretório raiz para resolução de caminhos relativos locais.

    Returns:
        Matriz da imagem no formato BGR do OpenCV.
    """

    data = task.get("data") or {}
    image_ref = data.get(value_field) or data.get("image")
    if not image_ref:
        raise ValueError(f"Task sem campo de imagem em data.{value_field}")

    image_bytes = _read_image_bytes(str(image_ref), default_root)
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Não foi possível decodificar imagem: {image_ref}")
    return image


def _read_image_bytes(image_ref: str, default_root: Path) -> bytes:
    parsed = urlparse(image_ref)
    if parsed.scheme in {"http", "https"}:
        request = urllib.request.Request(image_ref, headers={"User-Agent": "dataset-studio-ml-backend"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()

    if parsed.path.startswith("/data/local-files/"):
        query = parse_qs(parsed.query)
        local_ref = query.get("d", [""])[0]
        if not local_ref:
            raise ValueError(f"URL local-files sem parâmetro d: {image_ref}")
        document_root = Path(
            os.environ.get(
                "LOCAL_FILES_DOCUMENT_ROOT",
                os.environ.get(
                    "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT",
                    str(default_root),
                ),
            )
        ).resolve()
        path = (document_root / unquote(local_ref).lstrip("/\\")).resolve()
        if document_root not in path.parents and path != document_root:
            raise ValueError(f"Imagem fora do document root permitido: {path}")
        return path.read_bytes()

    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    else:
        path = Path(image_ref)
        if not path.is_absolute():
            path = default_root / path

    return path.read_bytes()


def detection_to_label_studio_result(
    detection: Detection,
    image_width: int,
    image_height: int,
    class_names: list[str],
    from_name: str,
    to_name: str,
    result_id: str,
) -> dict[str, Any]:
    """Converte uma estrutura neutra de Detection para o formato JSON de regiões do Label Studio."""

    x1, y1, x2, y2 = detection.bbox_xyxy
    x1 = max(0, min(image_width, x1))
    y1 = max(0, min(image_height, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    class_name = (
        class_names[detection.class_id]
        if 0 <= detection.class_id < len(class_names)
        else "objeto"
    )

    return {
        "id": result_id,
        "type": "rectanglelabels",
        "from_name": from_name,
        "to_name": to_name,
        "original_width": image_width,
        "original_height": image_height,
        "score": round(detection.confidence, 6),
        "value": {
            "x": round(x1 / image_width * 100.0, 4) if image_width else 0.0,
            "y": round(y1 / image_height * 100.0, 4) if image_height else 0.0,
            "width": round(width / image_width * 100.0, 4) if image_width else 0.0,
            "height": round(height / image_height * 100.0, 4) if image_height else 0.0,
            "rectanglelabels": [class_name],
        },
    }


class GenericLabelStudioBackend:
    """Servidor Backend genérico para comunicação e inferência com o Label Studio."""

    def __init__(
        self,
        predictor: Predictor,
        class_names: list[str],
        from_name: str = DEFAULT_FROM_NAME,
        to_name: str = DEFAULT_TO_NAME,
        value_field: str = DEFAULT_VALUE_FIELD,
        default_root: Path | None = None,
    ) -> None:
        """Inicializa o backend com o preditor configurado e parâmetros de rotulação."""

        self.predictor = predictor
        self.class_names = class_names
        self.from_name = from_name
        self.to_name = to_name
        self.value_field = value_field
        self.default_root = default_root or Path.cwd()

    def predict(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Gera as predições para uma lista de tarefas enviadas pelo Label Studio."""

        predictions: list[dict[str, Any]] = []
        for task_index, task in enumerate(tasks):
            image = read_image_from_task(task, self.value_field, self.default_root)
            image_height, image_width = image.shape[:2]
            detections = self.predictor.predict(image)
            results = [
                detection_to_label_studio_result(
                    detection=det,
                    image_width=image_width,
                    image_height=image_height,
                    class_names=self.class_names,
                    from_name=self.from_name,
                    to_name=self.to_name,
                    result_id=label_studio_region_id(
                        task.get("id", task_index), idx
                    ),
                )
                for idx, det in enumerate(detections)
            ]
            score = (
                sum(d.confidence for d in detections) / len(detections)
                if detections
                else 0.0
            )
            predictions.append(
                {
                    "result": results,
                    "score": round(score, 6),
                    "model_version": self.predictor.model_version,
                }
            )
        return predictions


def create_app(backend: GenericLabelStudioBackend):
    """Cria e configura a aplicação FastAPI que expõe os endpoints do ML Backend."""

    from fastapi import FastAPI

    app = FastAPI(title="Dataset Studio Label Studio ML Backend")

    @app.get("/")
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "UP",
            "model_version": backend.predictor.model_version,
        }

    @app.post("/setup")
    async def setup() -> dict[str, Any]:
        return {"model_version": backend.predictor.model_version}

    @app.post("/predict")
    async def predict(payload: dict[str, Any]) -> dict[str, Any]:
        tasks = payload.get("tasks") or []
        predictions = backend.predict(tasks)
        return {
            "results": predictions,
            "model_version": backend.predictor.model_version,
        }

    @app.post("/webhook")
    async def webhook(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_version": backend.predictor.model_version,
        }

    return app
