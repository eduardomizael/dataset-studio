from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dataset_studio.application import (
    TrainingParams,
    begin_training_record,
    export_deployment_bundle,
    finalize_training_record,
    validate_deployment_bundle,
)
from dataset_studio.domain import (
    Workspace,
    WorkflowError,
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
    evaluation_root = run_root / "evaluations"
    evaluation_root.mkdir()
    (evaluation_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluations": {
                    "test_normal": {
                        "status": "completed",
                        "images": 10,
                        "boxes": 20,
                        "map50_95": 0.70,
                    },
                    "test_stress": {
                        "status": "completed",
                        "images": 5,
                        "boxes": 10,
                        "map50_95": 0.55,
                    },
                },
                "robustness": {
                    "status": "completed",
                    "map50_95": {
                        "drop_absolute": 0.15,
                        "drop_relative": 0.214285714,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    completed = finalize_training_record(ws, "training_one", "completed")

    assert started["dataset_id"] == "version_registry"
    assert completed["metrics"]["best"]["map50_95"] == 0.7
    assert completed["artifacts"]["best"]["sha256"] == sha256(
        weights / "best.pt"
    )
    assert completed["evaluations"]["test_stress"]["map50_95"] == 0.55
    assert completed["robustness"]["map50_95"]["drop_absolute"] == 0.15
    assert completed["artifacts"]["evaluation_summary"]["sha256"] == sha256(
        evaluation_root / "summary.json"
    )
    assert load_yaml(run_root / "run.yaml")["run_id"] == "training_one"
    output = list_registered_models(ws)[completed["output_model_id"]]
    assert output["parent_model_id"] == started["initial_model_id"]
    assert output["dataset_id"] == "version_registry"
    assert load_yaml(run_registry_path(ws, "training_one"))["status"] == "completed"
    assert validate_registry(ws)["valid"] is True

    client = TestClient(create_web_app(ws))
    api_status = client.get("/api/registry/status")
    api_sources = client.get("/api/registry/sources")
    api_training = client.get("/api/trainings/training_one")
    assert api_status.status_code == 200
    assert api_status.json()["valid"] is True
    assert api_sources.status_code == 200
    assert api_training.status_code == 200
    assert api_training.json()["registry"]["dataset_id"] == "version_registry"
    assert api_training.json()["registered_model"]["parent_model_id"] == started[
        "initial_model_id"
    ]

    deployment = export_deployment_bundle(
        ws, completed["output_model_id"], deployment_id="deploy_training_one"
    )
    deployment_root = ws.deployments_root / "deploy_training_one"
    deployed_model = deployment_root / deployment["artifact"]["path"]
    assert deployment["immutable"] is True
    assert deployment["model"]["dataset_id"] == "version_registry"
    assert deployment["artifact"]["sha256"] == sha256(deployed_model)
    assert load_yaml(
        deployment_root / "deployment_manifest.yaml"
    ) == deployment

    api_deployment = client.post(
        f"/api/models/{completed['output_model_id']}/deploy",
        json={"deployment_id": "deploy_training_one"},
    )
    assert api_deployment.status_code == 200
    assert api_deployment.json()["manifest_path"] == (
        "deployments/deploy_training_one/deployment_manifest.yaml"
    )
    deployed_model.write_bytes(b"tampered-after-export")
    with pytest.raises(WorkflowError, match="SHA-256 divergente"):
        validate_deployment_bundle(ws, "deploy_training_one")


def test_deployment_rejects_artifact_that_differs_from_registry(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    create_version_snapshot(ws)
    model = ws.models_root / "model.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"original")
    params = TrainingParams(
        model=str(model),
        epochs=1,
        project=str(ws.runs_root),
        name="training_tampered",
    )
    begin_training_record(ws, "training_tampered", "version_registry", params)
    weights = ws.runs_root / "training_tampered" / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    best.write_bytes(b"best")
    completed = finalize_training_record(ws, "training_tampered", "completed")
    best.write_bytes(b"tampered")

    with pytest.raises(WorkflowError, match="não corresponde ao SHA-256"):
        export_deployment_bundle(ws, completed["output_model_id"])
