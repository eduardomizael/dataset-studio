"""Orquestração idempotente de projetos do Label Studio por origem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_studio.adapters.label_studio.api_client import LabelStudioClient
from dataset_studio.adapters.label_studio.credentials import (
    load_label_studio_credentials,
    public_credentials_status,
)
from dataset_studio.domain import Workspace, load_source
from dataset_studio.domain.errors import WorkflowError
from dataset_studio.domain.workspace import sha256, utc_now


@dataclass(frozen=True)
class PredictionPlan:
    total_tasks: int
    selected_version: str | None
    covered_tasks: int
    nonempty_tasks: int
    versions: tuple[dict[str, Any], ...]

    @property
    def uses_predictions(self) -> bool:
        return self.selected_version is not None

    @property
    def full_coverage(self) -> bool:
        return not self.uses_predictions or self.covered_tasks == self.total_tasks

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "uses_predictions": self.uses_predictions,
            "selected_version": self.selected_version,
            "covered_tasks": self.covered_tasks,
            "nonempty_tasks": self.nonempty_tasks,
            "full_coverage": self.full_coverage,
            "versions": list(self.versions),
        }


def integration_state_path(ws: Workspace, source_id: str) -> Path:
    return ws.source_root(source_id) / "label_studio" / "integration.json"


def labeling_config_path(ws: Workspace, source_id: str) -> Path:
    return ws.source_root(source_id) / "label_studio" / "labeling_config.xml"


def import_tasks_file(ws: Workspace, source_id: str) -> Path:
    return ws.source_root(source_id) / "label_studio" / "import_tasks.json"


def load_integration_state(ws: Workspace, source_id: str) -> dict[str, Any] | None:
    path = integration_state_path(ws, source_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"Estado de integração inválido: {path}") from exc
    if not isinstance(payload, dict):
        raise WorkflowError(f"Estado de integração inválido: {path}")
    return payload


def _write_integration_state(
    ws: Workspace, source_id: str, payload: dict[str, Any]
) -> Path:
    path = integration_state_path(ws, source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def load_import_tasks(ws: Workspace, source_id: str) -> list[dict[str, Any]]:
    path = import_tasks_file(ws, source_id)
    if not path.is_file():
        raise WorkflowError(
            "Gere import_tasks.json antes de preparar o projeto do Label Studio."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise WorkflowError(f"Arquivo de tarefas vazio ou inválido: {path}")
    if not all(isinstance(task, dict) and isinstance(task.get("data"), dict) for task in payload):
        raise WorkflowError(f"Formato de tarefas inválido: {path}")
    return payload


def build_prediction_plan(tasks: list[dict[str, Any]]) -> PredictionPlan:
    versions: dict[str, dict[str, int]] = {}
    for task in tasks:
        seen_in_task: set[str] = set()
        nonempty_in_task: set[str] = set()
        for prediction in task.get("predictions") or []:
            if not isinstance(prediction, dict):
                continue
            version = str(prediction.get("model_version") or "").strip()
            if not version:
                continue
            seen_in_task.add(version)
            if prediction.get("result"):
                nonempty_in_task.add(version)
        for version in seen_in_task:
            stats = versions.setdefault(version, {"covered_tasks": 0, "nonempty_tasks": 0})
            stats["covered_tasks"] += 1
            if version in nonempty_in_task:
                stats["nonempty_tasks"] += 1

    ordered = sorted(
        (
            {
                "model_version": version,
                "covered_tasks": stats["covered_tasks"],
                "nonempty_tasks": stats["nonempty_tasks"],
            }
            for version, stats in versions.items()
        ),
        key=lambda item: (
            item["covered_tasks"] == len(tasks),
            item["covered_tasks"],
            item["nonempty_tasks"],
            item["model_version"],
        ),
        reverse=True,
    )
    selected = ordered[0] if ordered else None
    return PredictionPlan(
        total_tasks=len(tasks),
        selected_version=selected["model_version"] if selected else None,
        covered_tasks=selected["covered_tasks"] if selected else 0,
        nonempty_tasks=selected["nonempty_tasks"] if selected else 0,
        versions=tuple(ordered),
    )


def label_studio_integration_status(
    ws: Workspace, source_id: str
) -> dict[str, Any]:
    load_source(ws, source_id)
    state = load_integration_state(ws, source_id)
    tasks_path = import_tasks_file(ws, source_id)
    plan = None
    if tasks_path.is_file():
        plan = build_prediction_plan(load_import_tasks(ws, source_id)).as_dict()
    credentials = public_credentials_status()
    if not credentials["configured"]:
        status = "needs-token"
        message = (
            "Configure uma única vez o token do Label Studio. "
            "As próximas origens serão preparadas automaticamente."
        )
    elif state:
        status = "ready"
        message = "Projeto do Label Studio vinculado e configurado automaticamente."
    elif not tasks_path.is_file():
        status = "waiting-import"
        message = "Gere import_tasks.json antes de preparar o Label Studio."
    elif plan and not plan["full_coverage"]:
        status = "partial-predictions"
        message = (
            f"A melhor predição cobre {plan['covered_tasks']} de "
            f"{plan['total_tasks']} tarefas."
        )
    else:
        status = "ready-to-prepare"
        message = "Integração pronta para criar ou reconhecer o projeto automaticamente."
    return {
        "status": status,
        "message": message,
        "credentials": credentials,
        "prediction_plan": plan,
        "integration": state,
    }


def _project_task_count(project: dict[str, Any]) -> int:
    for key in ("task_number", "tasks_count", "total_task_number"):
        value = project.get(key)
        if isinstance(value, int):
            return value
    return 0


def _task_source_id(task: dict[str, Any]) -> str | None:
    data = task.get("data")
    if not isinstance(data, dict):
        return None
    value = data.get("source_id") or data.get("campaign_id")
    return str(value) if value else None


def _find_existing_project(
    client: LabelStudioClient,
    *,
    source_id: str,
    total_tasks: int,
) -> dict[str, Any] | None:
    marker = f"dataset-studio-source:{source_id}"
    candidates: list[dict[str, Any]] = []
    for project in client.list_projects():
        project_id = project.get("id")
        if not isinstance(project_id, int):
            continue
        details = project
        if _project_task_count(details) == 0:
            details = client.get_project(project_id) or project
        description = str(project.get("description") or "")
        if marker in description:
            candidates.append(details)
            continue
        if _project_task_count(details) != total_tasks:
            continue
        first = client.list_tasks(project_id, page_size=1)
        if first and _task_source_id(first[0]) == source_id:
            candidates.append(details)
    unique = {int(project["id"]): project for project in candidates}
    if len(unique) > 1:
        ids = ", ".join(str(value) for value in sorted(unique))
        raise WorkflowError(
            f"Mais de um projeto do Label Studio corresponde à origem {source_id}: "
            f"{ids}. Vincule explicitamente o projeto correto."
        )
    return next(iter(unique.values()), None)


def ensure_label_studio_project(
    ws: Workspace,
    source_id: str,
    *,
    allow_partial_predictions: bool = False,
    ml_backend_url: str | None = None,
    client: LabelStudioClient | None = None,
) -> dict[str, Any]:
    """Cria, reconhece e configura um projeto sem duplicar importações."""

    load_source(ws, source_id)
    tasks = load_import_tasks(ws, source_id)
    plan = build_prediction_plan(tasks)
    if plan.uses_predictions and not plan.full_coverage and not allow_partial_predictions:
        raise WorkflowError(
            f"A versão de predição mais abrangente cobre {plan.covered_tasks} de "
            f"{plan.total_tasks} tarefas. Confirme explicitamente a cobertura parcial "
            "ou gere predições para todas as tarefas."
        )

    credentials = load_label_studio_credentials() if client is None else None
    if client is None:
        if credentials is None:
            raise WorkflowError(
                "Configure uma única vez o token do Label Studio antes de continuar."
            )
        client = LabelStudioClient(credentials.base_url, credentials.api_key)
    base_url = (
        credentials.base_url
        if credentials is not None
        else getattr(client, "base_url", "http://127.0.0.1:8080")
    )
    client.authenticate()

    config_path = labeling_config_path(ws, source_id)
    if not config_path.is_file():
        raise WorkflowError(f"Configuração de rotulação não encontrada: {config_path}")
    label_config = config_path.read_text(encoding="utf-8")
    import_path = import_tasks_file(ws, source_id)
    import_hash = sha256(import_path)
    config_hash = sha256(config_path)
    state = load_integration_state(ws, source_id)
    project = None
    created = False
    imported = False

    if state and isinstance(state.get("project_id"), int):
        if state.get("import_tasks_sha256") != import_hash:
            raise WorkflowError(
                "import_tasks.json diverge do arquivo vinculado ao Label Studio. "
                "A origem fixada não pode ser reimportada silenciosamente."
            )
        project = client.get_project(int(state["project_id"]))

    if project is None:
        project = _find_existing_project(
            client, source_id=source_id, total_tasks=plan.total_tasks
        )

    marker = f"dataset-studio-source:{source_id}"
    if project is None:
        project = client.create_project(
            {
                "title": f"Dataset Studio - {source_id}"[:50],
                "description": (
                    f"Projeto gerenciado automaticamente pelo Dataset Studio. [{marker}]"
                ),
                "label_config": label_config,
                "sampling": "Sequential sampling",
                "maximum_annotations": 1,
                "show_collab_predictions": plan.uses_predictions,
                "model_version": plan.selected_version,
                "reveal_preannotations_interactively": False,
            }
        )
        created = True

    project_id = int(project["id"])
    task_count = _project_task_count(project)
    if created or task_count == 0:
        response = client.import_tasks(project_id, tasks)
        imported_count = response.get("task_count")
        if isinstance(imported_count, int) and imported_count != plan.total_tasks:
            raise WorkflowError(
                f"O Label Studio importou {imported_count} de {plan.total_tasks} tarefas."
            )
        imported = True
    elif task_count != plan.total_tasks:
        raise WorkflowError(
            f"O projeto {project_id} possui {task_count} tarefas, mas a origem possui "
            f"{plan.total_tasks}. A importação automática foi interrompida para evitar duplicação."
        )

    configured = client.update_project(
        project_id,
        {
            "sampling": "Sequential sampling",
            "maximum_annotations": 1,
            "show_collab_predictions": plan.uses_predictions,
            "model_version": plan.selected_version,
            "reveal_preannotations_interactively": False,
            "evaluate_predictions_automatically": bool(ml_backend_url),
        },
    )
    ml_backend = None
    if ml_backend_url:
        normalized_backend_url = ml_backend_url.rstrip("/")
        existing_backends = client.list_ml_backends(project_id)
        ml_backend = next(
            (
                backend
                for backend in existing_backends
                if str(backend.get("url") or "").rstrip("/")
                == normalized_backend_url
            ),
            None,
        )
        backend_payload = {
            "url": normalized_backend_url,
            "title": f"Dataset Studio - {source_id}"[:50],
            "auto_update": False,
            "is_interactive": False,
        }
        if ml_backend and isinstance(ml_backend.get("id"), int):
            ml_backend = client.update_ml_backend(
                int(ml_backend["id"]), backend_payload
            )
        else:
            ml_backend = client.create_ml_backend(
                project_id,
                normalized_backend_url,
                title=f"Dataset Studio - {source_id}"[:50],
            )
    payload = {
        "schema_version": 1,
        "source_id": source_id,
        "base_url": base_url,
        "project_id": project_id,
        "project_title": configured.get("title") or project.get("title"),
        "import_tasks_sha256": import_hash,
        "labeling_config_sha256": config_hash,
        "selected_prediction_version": plan.selected_version,
        "prediction_coverage": plan.as_dict(),
        "settings": {
            "sampling": "Sequential sampling",
            "maximum_annotations": 1,
            "show_collab_predictions": plan.uses_predictions,
            "reveal_preannotations_interactively": False,
            "evaluate_predictions_automatically": bool(ml_backend_url),
        },
        "ml_backend": (
            {
                "id": ml_backend.get("id"),
                "url": ml_backend.get("url") or ml_backend_url,
                "state": ml_backend.get("state"),
            }
            if isinstance(ml_backend, dict)
            else None
        ),
        "last_synced_at": utc_now(),
    }
    _write_integration_state(ws, source_id, payload)
    return {
        "status": "ready",
        "created": created,
        "imported": imported,
        "project_id": project_id,
        "url": f"{base_url}/projects/{project_id}/data",
        "prediction_plan": plan.as_dict(),
        "ml_backend": payload["ml_backend"],
        "integration": payload,
    }
