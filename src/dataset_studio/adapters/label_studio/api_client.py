"""Cliente HTTP mínimo para a API oficial do Label Studio."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dataset_studio.domain.errors import WorkflowError


class LabelStudioApiError(WorkflowError):
    """Erro retornado pela API do Label Studio."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LabelStudioClient:
    """Cliente sem dependência do SDK, compatível com token legado e PAT."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self._authorization: str | None = None

    def _raw_request(
        self,
        method: str,
        path: str,
        *,
        payload: Any | None = None,
        authorization: str | None = None,
    ) -> tuple[int, Any]:
        url = path if path.startswith(("http://", "https://")) else f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if authorization:
            headers["Authorization"] = authorization
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return response.status, json.loads(raw) if raw else {}
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                body: Any = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {"detail": raw}
            return exc.code, body
        except URLError as exc:
            raise LabelStudioApiError(
                f"Não foi possível acessar o Label Studio em {self.base_url}: {exc.reason}"
            ) from exc

    @staticmethod
    def _error_detail(payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("detail", "message", "error"):
                if payload.get(key):
                    return str(payload[key])
        return str(payload)

    def authenticate(self) -> dict[str, Any]:
        """Detecta automaticamente token legado ou PAT e valida o usuário."""

        if not self.api_key:
            raise LabelStudioApiError("Informe um token de acesso do Label Studio.")

        legacy = f"Token {self.api_key}"
        status, payload = self._raw_request(
            "GET", "/api/current-user/whoami", authorization=legacy
        )
        if status == 200:
            self._authorization = legacy
            return payload

        status, refreshed = self._raw_request(
            "POST", "/api/token/refresh", payload={"refresh": self.api_key}
        )
        access = refreshed.get("access") if isinstance(refreshed, dict) else None
        if status == 200 and access:
            bearer = f"Bearer {access}"
            verify_status, user = self._raw_request(
                "GET", "/api/current-user/whoami", authorization=bearer
            )
            if verify_status == 200:
                self._authorization = bearer
                return user

        raise LabelStudioApiError(
            "Token rejeitado pelo Label Studio. Copie um token válido em "
            "Account & Settings > Access Token.",
            status_code=401,
        )

    def request(
        self, method: str, path: str, *, payload: Any | None = None
    ) -> Any:
        if self._authorization is None:
            self.authenticate()
        status, body = self._raw_request(
            method, path, payload=payload, authorization=self._authorization
        )
        if status == 401:
            self._authorization = None
            self.authenticate()
            status, body = self._raw_request(
                method, path, payload=payload, authorization=self._authorization
            )
        if not 200 <= status < 300:
            raise LabelStudioApiError(
                f"Label Studio respondeu {status}: {self._error_detail(body)}",
                status_code=status,
            )
        return body

    def get_project(self, project_id: int) -> dict[str, Any] | None:
        try:
            payload = self.request("GET", f"/api/projects/{project_id}")
        except LabelStudioApiError as exc:
            if exc.status_code == 404:
                return None
            raise
        return payload if isinstance(payload, dict) else None

    def list_projects(self) -> list[dict[str, Any]]:
        payload = self.request("GET", "/api/projects/?page_size=100")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            values = payload.get("results") or payload.get("projects") or []
            return [item for item in values if isinstance(item, dict)]
        return []

    def create_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.request("POST", "/api/projects/", payload=payload)
        if not isinstance(result, dict) or not result.get("id"):
            raise LabelStudioApiError("O Label Studio não retornou o ID do novo projeto.")
        return result

    def update_project(
        self, project_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        result = self.request(
            "PATCH", f"/api/projects/{project_id}", payload=payload
        )
        if not isinstance(result, dict):
            raise LabelStudioApiError("Resposta inválida ao configurar o projeto.")
        return result

    def import_tasks(
        self, project_id: int, tasks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        result = self.request(
            "POST",
            f"/api/projects/{project_id}/import?commit_to_project=true",
            payload=tasks,
        )
        if not isinstance(result, dict):
            raise LabelStudioApiError("Resposta inválida ao importar as tarefas.")
        return result

    def list_tasks(
        self, project_id: int, *, page_size: int = 1
    ) -> list[dict[str, Any]]:
        query = urlencode({"project": project_id, "page_size": page_size})
        payload = self.request("GET", f"/api/tasks?{query}")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            values = payload.get("tasks") or payload.get("results") or []
            return [item for item in values if isinstance(item, dict)]
        return []

    def list_ml_backends(self, project_id: int) -> list[dict[str, Any]]:
        query = urlencode({"project": project_id})
        payload = self.request("GET", f"/api/ml?{query}")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            values = payload.get("results") or payload.get("ml_backends") or []
            return [item for item in values if isinstance(item, dict)]
        return []

    def create_ml_backend(
        self, project_id: int, url: str, *, title: str
    ) -> dict[str, Any]:
        payload = self.request(
            "POST",
            "/api/ml/",
            payload={
                "project": project_id,
                "url": url.rstrip("/"),
                "title": title,
                "auth_method": "NONE",
                "auto_update": False,
                "is_interactive": False,
            },
        )
        if not isinstance(payload, dict) or not payload.get("id"):
            raise LabelStudioApiError(
                "O Label Studio não confirmou a conexão com o ML Backend."
            )
        return payload

    def update_ml_backend(
        self, backend_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        result = self.request("PATCH", f"/api/ml/{backend_id}", payload=payload)
        if not isinstance(result, dict):
            raise LabelStudioApiError("Resposta inválida ao atualizar o ML Backend.")
        return result
