from __future__ import annotations

import json
from pathlib import Path

import yaml

from dataset_studio.adapters.label_studio.credentials import (
    load_label_studio_credentials,
    public_credentials_status,
    save_label_studio_credentials,
)
from dataset_studio.application.label_studio_service import (
    build_prediction_plan,
    ensure_label_studio_project,
    label_studio_integration_status,
    load_integration_state,
)
from dataset_studio.domain import Workspace


def _task(source_id: str, version: str | None, regions: int = 1) -> dict:
    predictions = []
    if version:
        predictions = [
            {
                "model_version": version,
                "score": 0.5,
                "result": [{"id": f"r{index}"} for index in range(regions)],
            }
        ]
    return {
        "data": {
            "image": f"/data/local-files/?d={source_id}.jpg",
            "source_id": source_id,
        },
        "predictions": predictions,
    }


def _source_workspace(tmp_path: Path, tasks: list[dict]) -> Workspace:
    ws = Workspace.from_path(tmp_path)
    root = ws.source_root("source_a")
    (root / "label_studio").mkdir(parents=True)
    (root / "source.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source_id": "source_a",
                "campaign_id": "source_a",
                "videos": {"directory": "videos", "files": []},
                "extraction": {"mode": "uniform"},
                "annotation": {"classes": ["peixe"], "backend": "none"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "label_studio" / "labeling_config.xml").write_text(
        '<View><Image name="image" value="$image"/></View>',
        encoding="utf-8",
    )
    (root / "label_studio" / "import_tasks.json").write_text(
        json.dumps(tasks), encoding="utf-8"
    )
    return ws


class FakeClient:
    base_url = "http://label-studio.test"

    def __init__(self, projects=None, first_tasks=None):
        self.projects = list(projects or [])
        self.first_tasks = dict(first_tasks or {})
        self.created_payload = None
        self.updated_payload = None
        self.imported_tasks = None
        self.ml_backends = []

    def authenticate(self):
        return {"id": 1}

    def list_projects(self):
        return self.projects

    def list_tasks(self, project_id, page_size=1):
        return self.first_tasks.get(project_id, [])

    def get_project(self, project_id):
        return next((p for p in self.projects if p["id"] == project_id), None)

    def create_project(self, payload):
        self.created_payload = payload
        project = {"id": 7, "title": payload["title"], "task_number": 0}
        self.projects.append(project)
        return project

    def import_tasks(self, project_id, tasks):
        self.imported_tasks = list(tasks)
        project = self.get_project(project_id)
        project["task_number"] = len(tasks)
        return {"task_count": len(tasks)}

    def update_project(self, project_id, payload):
        self.updated_payload = payload
        project = self.get_project(project_id)
        project.update(payload)
        return project

    def list_ml_backends(self, project_id):
        return list(self.ml_backends)

    def create_ml_backend(self, project_id, url, title):
        backend = {
            "id": 9,
            "project": project_id,
            "url": url,
            "title": title,
            "state": "CO",
        }
        self.ml_backends.append(backend)
        return backend

    def update_ml_backend(self, backend_id, payload):
        backend = next(item for item in self.ml_backends if item["id"] == backend_id)
        backend.update(payload)
        return backend


def test_prediction_plan_prefers_full_coverage_over_newer_partial_version():
    tasks = [
        {
            "data": {"source_id": "source_a"},
            "predictions": [
                {"model_version": "baseline", "result": [{"id": "a"}]},
                {"model_version": "new", "result": [{"id": "b"}]},
            ],
        },
        _task("source_a", "baseline", regions=0),
    ]

    plan = build_prediction_plan(tasks)

    assert plan.selected_version == "baseline"
    assert plan.covered_tasks == 2
    assert plan.nonempty_tasks == 1
    assert plan.full_coverage is True


def test_prediction_plan_supports_intentional_manual_labeling():
    plan = build_prediction_plan([_task("source_a", None), _task("source_a", None)])

    assert plan.selected_version is None
    assert plan.uses_predictions is False
    assert plan.full_coverage is True


def test_ensure_project_creates_imports_and_persists_idempotent_state(tmp_path: Path):
    ws = _source_workspace(
        tmp_path,
        [_task("source_a", "baseline"), _task("source_a", "baseline", regions=0)],
    )
    client = FakeClient()

    result = ensure_label_studio_project(ws, "source_a", client=client)

    assert result["created"] is True
    assert result["imported"] is True
    assert result["project_id"] == 7
    assert client.created_payload["show_collab_predictions"] is True
    assert client.updated_payload["model_version"] == "baseline"
    assert len(client.imported_tasks) == 2
    state = load_integration_state(ws, "source_a")
    assert state["project_id"] == 7
    assert state["prediction_coverage"]["covered_tasks"] == 2

    client.imported_tasks = None
    repeated = ensure_label_studio_project(ws, "source_a", client=client)
    assert repeated["created"] is False
    assert repeated["imported"] is False
    assert client.imported_tasks is None


def test_ensure_project_recognizes_existing_source_without_duplicate_import(
    tmp_path: Path,
):
    tasks = [_task("source_a", "baseline"), _task("source_a", "baseline")]
    ws = _source_workspace(tmp_path, tasks)
    existing = {"id": 2, "title": "Projeto antigo", "task_number": 2}
    client = FakeClient(
        projects=[existing],
        first_tasks={2: [{"data": {"source_id": "source_a"}}]},
    )

    result = ensure_label_studio_project(ws, "source_a", client=client)

    assert result["project_id"] == 2
    assert result["created"] is False
    assert result["imported"] is False
    assert client.created_payload is None
    assert client.imported_tasks is None
    assert client.updated_payload["sampling"] == "Sequential sampling"


def test_ensure_project_connects_ml_backend_without_manual_project_settings(
    tmp_path: Path,
):
    ws = _source_workspace(tmp_path, [_task("source_a", "baseline")])
    client = FakeClient()

    result = ensure_label_studio_project(
        ws,
        "source_a",
        client=client,
        ml_backend_url="http://127.0.0.1:9090/",
    )

    assert result["ml_backend"]["id"] == 9
    assert result["ml_backend"]["url"] == "http://127.0.0.1:9090"
    assert client.updated_payload["evaluate_predictions_automatically"] is True


def test_credentials_are_saved_once_and_never_exposed(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("DATASET_STUDIO_CREDENTIALS_DIR", str(tmp_path))
    monkeypatch.delenv("DATASET_STUDIO_LABEL_STUDIO_API_KEY", raising=False)

    save_label_studio_credentials("http://127.0.0.1:8080/", "secret-token")

    loaded = load_label_studio_credentials()
    assert loaded.api_key == "secret-token"
    status = public_credentials_status()
    assert status["configured"] is True
    assert "api_key" not in status
    assert "secret-token" not in json.dumps(status)


def test_linked_project_still_requests_token_on_a_new_computer(
    tmp_path: Path, monkeypatch
):
    ws = _source_workspace(tmp_path, [_task("source_a", "baseline")])
    ensure_label_studio_project(ws, "source_a", client=FakeClient())
    monkeypatch.setenv(
        "DATASET_STUDIO_CREDENTIALS_DIR", str(tmp_path / "empty-credentials")
    )
    monkeypatch.delenv("DATASET_STUDIO_LABEL_STUDIO_API_KEY", raising=False)

    status = label_studio_integration_status(ws, "source_a")

    assert status["status"] == "needs-token"
    assert status["integration"]["project_id"] == 7
