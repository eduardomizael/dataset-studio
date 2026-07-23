from __future__ import annotations

import json
import threading
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from dataset_studio.application import JobManager
from dataset_studio.domain import (
    Workspace,
)
from dataset_studio.web.app import create_web_app


def test_job_manager_requests_cooperative_shutdown(tmp_path: Path):
    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    manager = JobManager()
    shutdown_file = tmp_path / "label-studio.stop"
    manager._jobs["label-job"] = {
        "id": "label-job",
        "kind": "label-studio",
        "target": "label-studio",
        "command": "start-labeling",
        "status": "running",
        "started_at": "2026-07-18T12:00:00+00:00",
        "returncode": None,
        "shutdown_file": shutdown_file,
        "lines": [],
        "process": FakeProcess(),
    }

    result = manager.stop_target("label-studio")

    assert result["status"] == "stopping"
    assert "shutdown_file" not in result
    assert shutdown_file.read_text(encoding="utf-8") == "stop\n"


def test_job_manager_persists_training_log_and_metadata(tmp_path: Path, monkeypatch):
    finished = threading.Event()
    registry_finished = threading.Event()
    completed = []

    class FakeProcess:
        pid = 12346

        def __init__(self):
            self.stdout = iter(["epoch 1/1\n", "Treinamento concluído.\n"])
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            finished.set()
            return 0

    monkeypatch.setattr(
        "dataset_studio.application.job_service.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    output = tmp_path / "run"
    manager = JobManager()

    manager.start(
        ["train"],
        kind="train",
        target="release:test",
        metadata={"training": {"model": "models/candidate.pt"}},
        log_path=output / "workflow.log",
        on_complete=lambda job: (
            completed.append(job["status"]),
            registry_finished.set(),
        ),
    )

    assert finished.wait(timeout=1)
    assert registry_finished.wait(timeout=1)
    state = manager.list()[0]
    payload = json.loads(
        (output / "workflow_job.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "completed"
    assert "epoch 1/1" in (output / "workflow.log").read_text(encoding="utf-8")
    assert payload["status"] == "completed"
    assert payload["metadata"]["training"]["model"] == "models/candidate.pt"
    assert completed == ["completed"]


def test_web_app_endpoints(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATASET_STUDIO_CREDENTIALS_DIR", str(tmp_path / "credentials"))
    ws = Workspace.from_path(tmp_path)
    app = create_web_app(ws)
    client = TestClient(app)

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "capture.mp4").write_bytes(b"video content")

    resp_create = client.post(
        "/api/sources",
        json={
            "source_id": "web_source",
            "videos_dir": str(videos),
            "video_files": ["capture.mp4"],
            "classes": ["peixe"],
        },
    )
    assert resp_create.status_code == 200
    assert resp_create.json()["status"] == "ok"

    resp_list = client.get("/api/sources")
    assert resp_list.status_code == 200
    assert "web_source" in resp_list.json()

    resp_status = client.get("/api/sources/web_source")
    assert resp_status.status_code == 200
    st_data = resp_status.json()
    assert st_data.get("source_id") == "web_source" or st_data.get("campaign_id") == "web_source"
    assert "video_details" in st_data
    assert len(st_data["video_details"]) == 1
    assert st_data["video_details"][0]["name"] == "capture.mp4"
    assert "size_human" in st_data["video_details"][0]
    assert "resolution" in st_data["video_details"][0]
    assert "fps" in st_data["video_details"][0]

    index = client.get("/")
    assert index.status_code == 200
    assert "Dataset Studio" in index.text
    settings = client.get("/api/label-studio/settings")
    assert settings.status_code == 200
    assert settings.json()["configured"] is False
    assert "api_key" not in settings.json()


def test_release_page_preserves_selected_annotation_revision(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    client = TestClient(create_web_app(ws))

    source_page = client.get("/source.html")
    release_page = client.get("/release.html")

    assert source_page.status_code == 200
    assert "data.revision_id" in source_page.text
    assert "&revision_id=${encodeURIComponent(revisionId)}" in source_page.text
    assert release_page.status_code == 200
    assert "let annotationRevisionId = urlParams.get('revision_id')" in release_page.text
    assert "revision_id: annotationRevisionId" in release_page.text
    assert (
        "annotation_revisions: annotationRevisionId ? { [campaignId]: annotationRevisionId } : {}"
        in release_page.text
    )


def test_completed_steps_locking(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    app = create_web_app(ws)
    client = TestClient(app)

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "video1.mp4").write_bytes(b"dummy video data")

    client.post(
        "/api/sources",
        json={
            "source_id": "locked_source",
            "videos_dir": str(videos),
            "video_files": ["video1.mp4"],
            "classes": ["peixe"],
        },
    )

    # Simular Etapa 2 Concluída criando manifesto
    src_dir = ws.source_root("locked_source")
    (src_dir / "frame_manifest.json").write_text(
        json.dumps({"frames": [{"frame_id": "f1", "image": "f1.jpg", "source_video": "video1.mp4", "frame_index": 0, "width": 100, "height": 100}]}),
        encoding="utf-8"
    )

    # Tentativa de re-executar Etapa 2 deve retornar 400
    resp_ext = client.post("/api/sources/locked_source/extract")
    assert resp_ext.status_code == 400
    assert "já foi concluída" in resp_ext.json()["detail"]

    # Simular Etapa 3 Concluída criando import_tasks.json
    (src_dir / "label_studio").mkdir(parents=True, exist_ok=True)
    (src_dir / "label_studio" / "import_tasks.json").write_text(
        json.dumps([{"id": 1}]), encoding="utf-8"
    )

    # Tentativa de re-executar Etapa 3 deve retornar 400
    resp_imp = client.post("/api/sources/locked_source/import-tasks")
    assert resp_imp.status_code == 400
    assert "já foi concluída" in resp_imp.json()["detail"]


def test_start_label_studio_endpoint_and_shutdown(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATASET_STUDIO_CREDENTIALS_DIR", str(tmp_path / "credentials"))
    started = []
    monkeypatch.setattr(
        "dataset_studio.web.app.start_label_studio_job",
        lambda *args, **kwargs: started.append("label-studio") or {"id": "ls-job", "status": "running"},
    )
    monkeypatch.setattr(
        "dataset_studio.web.app.start_ml_backend_job",
        lambda *args, **kwargs: started.append("ml-backend") or {"id": "ml-job", "status": "running"},
    )
    monkeypatch.setattr("dataset_studio.web.app.wait_for_port", lambda port, timeout=15.0: True)
    monkeypatch.setattr("dataset_studio.web.app.wait_for_ml_backend", lambda port, timeout=20.0: True)

    ws = Workspace.from_path(tmp_path)
    app = create_web_app(ws)
    client = TestClient(app)

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "sample.mp4").write_bytes(b"sample video data")

    client.post(
        "/api/sources",
        json={
            "source_id": "ls_source",
            "videos_dir": str(videos),
            "video_files": ["sample.mp4"],
            "classes": ["peixe"],
        },
    )

    model = ws.models_root / "model.pt"
    model.parent.mkdir()
    model.write_bytes(b"model")
    source_yaml = ws.source_config_path("ls_source")
    source = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source["annotation"].update({"backend": "local", "model": "models/model.pt"})
    source_yaml.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

    resp_start = client.post(
        "/api/sources/ls_source/start-label-studio",
        json={"enable_ml": True, "model": "models/model.pt"},
    )
    assert resp_start.status_code == 200
    data = resp_start.json()
    assert data["status"] == "ok"
    assert data["url"] == "http://127.0.0.1:8080"
    assert data["online"] is True

    assert started == ["ml-backend", "label-studio"]
