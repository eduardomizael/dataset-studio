from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from dataset_studio.application import (
    TrainingParams,
    begin_training_record,
    finalize_training_record,
)
from dataset_studio.domain import (
    Workspace,
    dump_yaml,
    list_registered_models,
    load_yaml,
    run_registry_path,
    sha256,
    validate_registry,
)
from dataset_studio.web.app import create_web_app


def create_version_snapshot(ws: Workspace) -> None:
    root = ws.version_root("version_registry")
    dump_yaml(
        root / "version.yaml",
        {
            "schema_version": 2,
            "version_id": "version_registry",
            "sources": ["source_one"],
            "annotation_revisions": {"source_one": "rev_one"},
            "provisional": False,
            "assignments": {
                "train": ["source_one/train.mp4"],
                "val": ["source_one/val.mp4"],
                "test_normal": [],
                "test_stress": [],
            },
        },
    )
    dump_yaml(
        root / "data.yaml",
        {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "peixe"},
        },
    )
    (root / "manifest.csv").write_text(
        "frame_id,split,label_sha256\nf1,train,abc\nf2,val,def\n",
        encoding="utf-8",
    )
    (root / "build_report.json").write_text(
        json.dumps(
            {
                "images": 2,
                "boxes": 1,
                "excluded_frames": 0,
                "splits": {"train": 1, "val": 1},
                "manifest_sha256": sha256(root / "manifest.csv"),
                "version_config_sha256": sha256(root / "version.yaml"),
            }
        ),
        encoding="utf-8",
    )


def test_training_registry_captures_dataset_parent_and_output_hashes(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    create_version_snapshot(ws)
    model = ws.models_root / "yolo26n.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"initial-model")
    params = TrainingParams(
        model="models/yolo26n.pt",
        epochs=2,
        imgsz=640,
        project=str(ws.runs_root),
        name="training_one",
    )

    started = begin_training_record(
        ws, "training_one", "version_registry", params
    )
    run_root = ws.runs_root / "training_one"
    weights = run_root / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best-model")
    (weights / "last.pt").write_bytes(b"last-model")
    (run_root / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.8,0.7,0.9,0.6\n"
        "1,0.9,0.8,0.95,0.7\n",
        encoding="utf-8",
    )
    dump_yaml(run_root / "args.yaml", params.to_dict())

    completed = finalize_training_record(ws, "training_one", "completed")

    assert started["dataset_id"] == "version_registry"
    assert completed["metrics"]["best"]["map50_95"] == 0.7
    assert completed["artifacts"]["best"]["sha256"] == sha256(
        weights / "best.pt"
    )
    output = list_registered_models(ws)[completed["output_model_id"]]
    assert output["parent_model_id"] == started["initial_model_id"]
    assert output["dataset_id"] == "version_registry"
    assert load_yaml(run_registry_path(ws, "training_one"))["status"] == "completed"
    assert validate_registry(ws)["valid"] is True

    client = TestClient(create_web_app(ws))
    api_status = client.get("/api/registry/status")
    api_training = client.get("/api/trainings/training_one")
    assert api_status.status_code == 200
    assert api_status.json()["valid"] is True
    assert api_training.status_code == 200
    assert api_training.json()["registry"]["dataset_id"] == "version_registry"
    assert api_training.json()["registered_model"]["parent_model_id"] == started[
        "initial_model_id"
    ]
