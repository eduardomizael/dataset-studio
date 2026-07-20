from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dataset_studio.application import JobManager
from dataset_studio.domain import (
    Workspace,
    campaign_root,
    create_campaign,
    frame_manifest_path,
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
    )

    assert finished.wait(timeout=1)
    state = manager.list()[0]
    payload = json.loads(
        (output / "workflow_job.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "completed"
    assert "epoch 1/1" in (output / "workflow.log").read_text(encoding="utf-8")
    assert payload["status"] == "completed"
    assert payload["metadata"]["training"]["model"] == "models/candidate.pt"


def test_web_app_endpoints(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    app = create_web_app(ws)
    client = TestClient(app)

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "capture.mp4").write_bytes(b"video")

    resp_create = client.post(
        "/api/campaigns",
        json={
            "campaign_id": "web_campaign",
            "videos_dir": str(videos),
            "video_files": ["capture.mp4"],
            "classes": ["objeto"],
        },
    )
    assert resp_create.status_code == 200

    resp_list = client.get("/api/campaigns")
    assert resp_list.status_code == 200
    assert "web_campaign" in resp_list.json()

    resp_status = client.get("/api/campaigns/web_campaign")
    assert resp_status.status_code == 200
    assert resp_status.json()["campaign_id"] == "web_campaign"

    index = client.get("/")
    assert index.status_code == 200
    assert "Dataset Studio" in index.text
