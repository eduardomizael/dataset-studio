from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dataset_studio.adapters.opencv.media import extract_source_frames
from dataset_studio.adapters.ultralytics.trainer import UltralyticsCommandTrainer
from dataset_studio.domain import (
    Workspace,
    WorkflowError,
    create_source,
    dump_yaml,
    load_frame_manifest,
    load_source,
    prediction_profile_sha256,
)
from dataset_studio.ports.trainer import TrainingParams
from dataset_studio.web.app import create_web_app


def _write_video(path: Path, frame_count: int = 10) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    for value in range(frame_count):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()


def test_uniform_extraction_uses_source_configuration(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    videos = tmp_path / "videos"
    videos.mkdir()
    _write_video(videos / "sample.avi")
    create_source(
        ws,
        source_id="uniform_source",
        videos_dir=videos,
        video_pattern="*.avi",
        extraction={"mode": "uniform", "uniform_frame_step": 3},
        annotation={"classes": ["objeto"]},
    )

    manifest_path = extract_source_frames(ws, "uniform_source")
    manifest = load_frame_manifest(ws, "uniform_source")

    assert manifest_path.is_file()
    assert manifest["mode"] == "uniform"
    assert [frame["frame_index"] for frame in manifest["frames"]] == [0, 3, 6, 9]


def test_upload_is_isolated_and_rejects_path_traversal(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    client = TestClient(create_web_app(ws))

    def upload(source_id: str, content: bytes, filename: str = "shared.mp4"):
        return client.post(
            "/api/sources/upload",
            data={"source_id": source_id, "classes": '["objeto"]'},
            files=[("videos", (filename, content, "video/mp4"))],
        )

    assert upload("source_one", b"first").status_code == 200
    assert upload("source_two", b"second").status_code == 200
    assert (ws.videos_root / "source_one" / "shared.mp4").read_bytes() == b"first"
    assert (ws.videos_root / "source_two" / "shared.mp4").read_bytes() == b"second"

    escaped = upload("source_three", b"escape", "../escaped.mp4")
    assert escaped.status_code == 400
    assert not (ws.videos_root / "escaped.mp4").exists()
    assert not ws.source_root("source_three").exists()


def test_deletion_requires_exact_confirmation_and_reports_dependencies(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "sample.mp4").write_bytes(b"video")
    create_source(
        ws,
        source_id="source_one",
        videos_dir=videos,
        video_pattern="*.mp4",
        annotation={"classes": ["objeto"]},
    )
    version_root = ws.version_root("version_one")
    dump_yaml(
        version_root / "version.yaml",
        {
            "version_id": "version_one",
            "sources": ["source_one"],
            "annotation_revisions": {"source_one": "rev_one"},
        },
    )
    run_root = ws.root / "runs" / "detect" / "training_one"
    run_root.mkdir(parents=True)
    (run_root / "workflow_job.json").write_text(
        json.dumps({"metadata": {"release_id": "version_one"}}), encoding="utf-8"
    )
    client = TestClient(create_web_app(ws))

    impact = client.get("/api/deletion-impact/source/source_one").json()
    assert impact["dependent_versions"] == ["version_one"]
    assert impact["dependent_trainings"] == ["training_one"]
    assert client.delete("/api/sources/source_one").status_code == 400

    deleted = client.delete(
        "/api/sources/source_one",
        params={"confirm": "source_one", "cascade": "true"},
    )
    assert deleted.status_code == 200
    assert not ws.source_root("source_one").exists()
    assert not (ws.registry_root / "sources" / "source_one.yaml").exists()
    assert not ws.version_root("version_one").exists()
    assert not run_root.exists()


def test_training_command_uses_current_environment(tmp_path: Path):
    command = UltralyticsCommandTrainer().build_command(
        tmp_path / "data.yaml", TrainingParams(device="cpu")
    )
    assert command[0] == sys.executable
    assert "fish_detection" not in " ".join(command)


def test_source_freezes_prediction_profile_and_detects_tampering(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    videos = ws.videos_root
    videos.mkdir()
    (videos / "sample.mp4").write_bytes(b"video")
    ws.models_root.mkdir()
    (ws.models_root / "model.pt").write_bytes(b"model")
    profile_path = ws.config_root / "prediction.yaml"
    dump_yaml(
        profile_path,
        {
            "detection": {
                "conf_threshold": 0.17,
                "device": "cpu",
                "img_size": 960,
                "iou_threshold": 0.5,
                "max_det": 100,
                "half_precision": False,
            },
            "roi": {"enabled": True, "points": [[1, 2], [3, 4]]},
            "tracking": {"ignored": True},
        },
    )
    create_source(
        ws,
        source_id="profile_source",
        videos_dir=videos,
        video_pattern="*.mp4",
        annotation={
            "classes": ["objeto"],
            "backend": "local",
            "model": "models/model.pt",
            "detection_config": "config/prediction.yaml",
        },
    )

    source = load_source(ws, "profile_source")
    profile = source["annotation"]["prediction_profile"]
    assert profile["detection"]["conf_threshold"] == 0.17
    assert "tracking" not in profile
    assert source["annotation"]["prediction_profile_sha256"] == (
        prediction_profile_sha256(profile)
    )

    profile_path.write_text("detection: {conf_threshold: 0.99}\n", encoding="utf-8")
    assert load_source(ws, "profile_source")["annotation"][
        "prediction_profile"
    ]["detection"]["conf_threshold"] == 0.17

    source["annotation"]["prediction_profile"]["detection"][
        "conf_threshold"
    ] = 0.99
    dump_yaml(ws.source_config_path("profile_source"), source)
    with pytest.raises(WorkflowError, match="perfil de predição"):
        load_source(ws, "profile_source")


def test_training_endpoint_persists_unique_identity(tmp_path: Path, monkeypatch):
    ws = Workspace.from_path(tmp_path)
    version_root = ws.version_root("version_one")
    dump_yaml(
        version_root / "version.yaml",
        {
            "version_id": "version_one",
            "sources": [],
            "assignments": {"train": [], "val": [], "test_normal": [], "test_stress": []},
        },
    )
    dump_yaml(
        version_root / "data.yaml",
        {"path": str(version_root), "train": "images/train", "val": "images/val", "names": {0: "objeto"}},
    )
    captured = []

    def fake_enqueue_training(**kwargs):
        captured.append(kwargs)
        return {"id": f"job-{len(captured)}", "metadata": kwargs["metadata"]}

    monkeypatch.setattr(
        "dataset_studio.web.app.job_manager.enqueue_training", fake_enqueue_training
    )
    client = TestClient(create_web_app(ws))
    payload = {"model": "yolo26n.pt", "epochs": 1, "imgsz": 64, "device": "cpu"}
    first = client.post("/api/versions/version_one/start-train", json=payload)
    second = client.post("/api/versions/version_one/start-train", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert captured[0]["metadata"]["release_id"] == "version_one"
    assert captured[0]["metadata"]["training_id"] != captured[1]["metadata"]["training_id"]
    assert captured[0]["command"][0] == sys.executable
    assert "exist_ok=True" in captured[0]["command"]
