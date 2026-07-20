"""Porta de abstração para modelos de inferência e predição."""

from __future__ import annotations

from typing import NamedTuple, Protocol
import numpy as np


class Detection(NamedTuple):
    """Representa uma detecção neutra (classe, confiança e bbox em pixels xyxy)."""

    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class Predictor(Protocol):
    """Contrato que qualquer preditor de objetos deve implementar."""

    def predict(self, image: np.ndarray) -> list[Detection]:
        ...

    @property
    def model_version(self) -> str:
        ...
