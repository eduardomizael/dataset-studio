"""Gerenciador de tarefas e processos em segundo plano com suporte a persistência e encerramento cooperativo."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_studio.adapters.label_studio.process_supervisor import (
    process_group_options,
    terminate_process_tree,
)
from dataset_studio.domain.errors import WorkflowError


class JobManager:
    """Executa processos longos sem bloquear a aplicação, gerenciando logs e sinalização de parada."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(
        self,
        command: list[str],
        *,
        kind: str,
        target: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        cooperative_stop: bool = False,
        metadata: dict[str, Any] | None = None,
        log_path: Path | None = None,
    ) -> dict[str, Any]:
        """Inicia um novo processo em segundo plano e registra seu acompanhamento."""

        with self._lock:
            for job in self._jobs.values():
                self._refresh(job)
                if job["target"] == target and job["status"] == "running":
                    raise WorkflowError(
                        f"Já existe uma operação em andamento para {target}."
                    )
            job_id = uuid.uuid4().hex[:10]
            shutdown_file = None
            if cooperative_stop:
                runtime_root = (
                    Path(tempfile.gettempdir()) / "dataset_studio_runtime"
                )
                runtime_root.mkdir(parents=True, exist_ok=True)
                shutdown_file = runtime_root / f"{job_id}.stop"
                shutdown_file.unlink(missing_ok=True)
                command = [*command, "--shutdown-file", str(shutdown_file)]

            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                **process_group_options(hidden=True),
            )
            job = {
                "id": job_id,
                "kind": kind,
                "target": target,
                "command": subprocess.list2cmdline(command),
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "returncode": None,
                "shutdown_file": shutdown_file,
                "metadata": metadata or {},
                "log_path": str(log_path) if log_path is not None else None,
                "_log_path": log_path,
                "lines": [],
                "process": process,
            }
            self._jobs[job_id] = job
            self._persist(job)
            threading.Thread(target=self._collect, args=(job,), daemon=True).start()
            return self._public(job)

    def _collect(self, job: dict[str, Any]) -> None:
        process = job["process"]
        assert process.stdout is not None
        log_path = job.get("_log_path")
        log_handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
        try:
            for line in process.stdout:
                if log_handle is not None:
                    log_handle.write(line)
                    log_handle.flush()
                with self._lock:
                    job["lines"].append(line.rstrip())
                    job["lines"] = job["lines"][-500:]
        finally:
            if log_handle is not None:
                log_handle.close()
        process.wait()
        with self._lock:
            self._refresh(job)

    @staticmethod
    def _refresh(job: dict[str, Any]) -> None:
        returncode = job["process"].poll()
        if returncode is not None and job["status"] in {"running", "stopping"}:
            job["returncode"] = returncode
            if job["status"] == "stopping":
                job["status"] = "stopped"
            else:
                job["status"] = "completed" if returncode == 0 else "failed"
            shutdown_file = job.get("shutdown_file")
            if shutdown_file is not None:
                shutdown_file.unlink(missing_ok=True)
            JobManager._persist(job)

    @staticmethod
    def _persist(job: dict[str, Any]) -> None:
        log_path = job.get("_log_path")
        if log_path is None:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: value
            for key, value in job.items()
            if key not in {"process", "shutdown_file", "lines", "_log_path"}
        }
        (log_path.parent / "workflow_job.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in job.items()
            if key not in {"process", "shutdown_file", "_log_path"}
        } | {"log": "\n".join(job["lines"])}

    def get(self, job_id: str) -> dict[str, Any]:
        """Obtém o status atualizado e os logs de um job específico pelo ID."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise WorkflowError(f"Job {job_id} não encontrado.")
            self._refresh(job)
            return self._public(job)

    def list(self) -> list[dict[str, Any]]:
        """Retorna a lista de todos os jobs registrados no gerenciador."""


        with self._lock:
            for job in self._jobs.values():
                self._refresh(job)
            return [self._public(job) for job in reversed(self._jobs.values())]

    def stop(self, job_id: str) -> dict[str, Any]:
        """Solicita a interrupção graciosa ou forçada de um job em execução."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise WorkflowError("Operação não encontrada.")
            self._refresh(job)
            if job["status"] == "running":
                job["status"] = "stopping"
                shutdown_file = job.get("shutdown_file")
                if shutdown_file is not None:
                    shutdown_file.write_text("stop\n", encoding="utf-8")
                    threading.Thread(
                        target=self._force_after_grace,
                        args=(job,),
                        daemon=True,
                    ).start()
                else:
                    threading.Thread(
                        target=terminate_process_tree,
                        args=(job["process"],),
                        daemon=True,
                    ).start()
            return self._public(job)

    def stop_target(self, target: str) -> dict[str, Any]:
        with self._lock:
            job_id = next(
                (
                    job["id"]
                    for job in self._jobs.values()
                    if job["target"] == target
                    and job["status"] in {"running", "stopping"}
                ),
                None,
            )
        if job_id is None:
            raise WorkflowError(f"Nenhuma operação ativa para {target}.")
        return self.stop(job_id)

    def stop_all(self, *, wait: bool = False, timeout: float = 12.0) -> None:
        """Interrompe todos os jobs ativos no gerenciador."""

        with self._lock:
            job_ids = [
                job["id"]
                for job in self._jobs.values()
                if job["status"] in {"running", "stopping"}
            ]
        for job_id in job_ids:
            self.stop(job_id)
        if not wait or not job_ids:
            return

        deadline = time.monotonic() + timeout
        active: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            with self._lock:
                active = []
                for job_id in job_ids:
                    job = self._jobs[job_id]
                    self._refresh(job)
                    if job["status"] in {"running", "stopping"}:
                        active.append(job)
            if not active:
                return
            time.sleep(0.1)

        for job in active:
            terminate_process_tree(job["process"])

    def _force_after_grace(self, job: dict[str, Any], grace: float = 8.0) -> None:
        try:
            job["process"].wait(timeout=grace)
        except subprocess.TimeoutExpired:
            terminate_process_tree(job["process"])
