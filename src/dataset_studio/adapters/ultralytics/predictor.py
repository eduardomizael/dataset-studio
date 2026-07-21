"""Adaptador de predição usando Ultralytics YOLO."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import cv2

from dataset_studio.ports.predictor import Detection, Predictor


class UltralyticsPredictor(Predictor):
    """Implementação do Predictor baseada no Ultralytics YOLO."""

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        device: str | None = None,
        inference_options: dict[str, Any] | None = None,
        roi_points: list[list[int]] | None = None,
    ) -> None:
        self.path = Path(model_path).resolve()
        self.conf = conf
        self.device = device
        self.inference_options = dict(inference_options or {})
        self.roi_points = roi_points or []
        self._model = None
        self._version = self._build_version()

    def _build_version(self) -> str:
        if not self.path.exists():
            return self.path.name or str(self.path)
        stat = self.path.stat()
        raw = f"{self.path}:{stat.st_mtime_ns}:{stat.st_size}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        return f"{self.path.name}:{digest}"

    def load(self) -> None:
        """Carrega tardiamente (lazy load) o modelo YOLO na memória."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.path))

    @property
    def model_version(self) -> str:
        """Retorna o identificador de versão do modelo carregado."""
        return self._version

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Executa a inferência YOLO na imagem fornecida e converte os resultados."""
        self.load()
        inference_image = image
        if self.roi_points:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.asarray(self.roi_points, dtype=np.int32)], 255)
            inference_image = cv2.bitwise_and(image, image, mask=mask)
        kwargs = {"conf": self.conf, "verbose": False, **self.inference_options}
        if self.device:
            kwargs["device"] = self.device
        results = self._model(inference_image, **kwargs)
        detections: list[Detection] = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = tuple(box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_id=cls_id,
                        confidence=conf,
                        bbox_xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    )
                )
        return detections

