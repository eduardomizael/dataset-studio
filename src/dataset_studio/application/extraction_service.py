"""Execução assíncrona e monitorável da extração de frames."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from dataset_studio.domain import WorkflowError


ProgressCallback = Callable[[dict[str, Any]], None]
ExtractionRunner = Callable[[ProgressCallback], Any]


class ExtractionJobManager:
    """Mantém uma única extração ativa por origem e expõe seu progresso."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in job.items()
            if not key.startswith("_")
        }

    def start(self, source_id: str, runner: ExtractionRunner) -> dict[str, Any]:
        """Inicia a extração ou rejeita uma segunda execução concorrente."""
        with self._lock:
            current = self._jobs.get(source_id)
            if current and current["status"] in {"queued", "running"}:
                raise WorkflowError(
                    "A extração de frames desta origem já está em execução."
                )
            job = {
                "id": uuid.uuid4().hex[:10],
                "source_id": source_id,
                "status": "queued",
                "stage": "queued",
                "message": "Extração aguardando início.",
                "percent": 0,
                "processed_units": 0,
                "total_units": 0,
                "frames_saved": 0,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "_runner": runner,
            }
            self._jobs[source_id] = job
            thread = threading.Thread(
                target=self._run,
                args=(source_id,),
                daemon=True,
                name=f"frame-extraction-{source_id}",
            )
            job["_thread"] = thread
            thread.start()
            return self._public(job)

    def _run(self, source_id: str) -> None:
        with self._lock:
            job = self._jobs[source_id]
            job.update(
                status="running",
                stage="initializing",
                message="Preparando vídeos e diretórios.",
                started_at=self._now(),
            )
            runner = job["_runner"]

        def report(update: dict[str, Any]) -> None:
            with self._lock:
                current = self._jobs.get(source_id)
                if current is None or current["status"] != "running":
                    return
                for key in (
                    "stage",
                    "message",
                    "percent",
                    "processed_units",
                    "total_units",
                    "frames_saved",
                    "current_unit",
                ):
                    if key in update:
                        current[key] = update[key]

        try:
            manifest = runner(report)
            with self._lock:
                job = self._jobs[source_id]
                job.update(
                    status="completed",
                    stage="completed",
                    message="Extração concluída com sucesso.",
                    percent=100,
                    manifest=str(manifest),
                    finished_at=self._now(),
                )
        except Exception as exc:
            with self._lock:
                job = self._jobs[source_id]
                job.update(
                    status="failed",
                    stage="failed",
                    message="A extração não pôde ser concluída.",
                    error=str(exc),
                    finished_at=self._now(),
                )

    def get(self, source_id: str) -> dict[str, Any]:
        """Retorna o último estado conhecido da extração da origem."""
        with self._lock:
            job = self._jobs.get(source_id)
            if job is None:
                return {
                    "source_id": source_id,
                    "status": "idle",
                    "stage": "idle",
                    "message": "Nenhuma extração em execução.",
                    "percent": 0,
                    "processed_units": 0,
                    "total_units": 0,
                    "frames_saved": 0,
                    "error": None,
                }
            return self._public(job)
