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
    """Contrato abstrato para modelos de inferência e detecção de objetos."""

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Executa a inferência de detecção em uma imagem NumPy.

        Args:
            image: Imagem de entrada em formato BGR/RGB como matriz NumPy.

        Returns:
            Lista de objetos Detection contendo classe, confiança e bounding box.
        """
        ...

    @property
    def model_version(self) -> str:
        """Identificador ou versão do modelo em uso."""
        ...

