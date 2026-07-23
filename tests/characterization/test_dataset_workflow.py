from __future__ import annotations

import json
import re
import csv
from pathlib import Path

import pytest

from dataset_studio.domain import (
    WorkflowError,
    Workspace,
    accept_native_export,
    annotation_revision_export_path,
    build_import_tasks,
    build_release,
    campaign_root,
    create_campaign,
    create_release,
    frame_manifest_path,
    inspect_native_export,
    label_studio_region_id,
    list_annotation_revisions,
    load_campaign,
    load_yaml,
    parse_native_export,
)


def defaults(tmp_path: Path) -> dict:
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"candidate-model")
    return {
        "schema_version": 2,
        "paths": {
            "campaigns_root": str(tmp_path / "dataset" / "campaigns"),
            "releases_root": str(tmp_path / "dataset" / "releases"),
            "videos_dir": str(tmp_path / "videos"),
            "models_root": str(tmp_path / "models"),
        },
        "extraction": {
            "model": str(model),
            "mode": "smart",
            "confidence": 0.1,
            "scan_step": 15,
            "dense_step": 30,
            "sparse_step": 90,
            "margin": 45,
            "max_negatives_per_video": 15,
            "uniform_frame_step": 30,
        },
        "annotation": {
            "classes": ["objeto"],
            "label_studio_port": 8080,
            "ml_backend_port": 9090,
            "model": str(model),
            "detection_config": "config/test_config.yaml",
        },
    }


def setup_dummy_config(tmp_path: Path):
    config_file = tmp_path / "config" / "test_config.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("detection:\n  confidence: 0.25\n", encoding="utf-8")


def create_fixture_campaign(tmp_path: Path, campaign_id: str = "capture_test") -> tuple[dict, Path]:
    setup_dummy_config(tmp_path)
    config = defaults(tmp_path)
    videos = Path(config["paths"]["videos_dir"])
    videos.mkdir(parents=True)
    (videos / "normal.mp4").write_bytes(b"video-normal")
    (videos / "validation.mp4").write_bytes(b"video-validation")
    create_campaign(
        config,
        campaign_id=campaign_id,
        videos_dir=videos,
        video_pattern="*.mp4",
    )
    root = campaign_root(config, campaign_id)
    images = root / "frames" / "raw" / "images"
    (images / "normal frame_f000001.jpg").write_bytes(b"normal-image")
    (images / "validation_f000001.jpg").write_bytes(b"validation-image")
    manifest = {
        "schema_version": 1,
        "model": "models/candidate.pt",
        "model_sha256": "a" * 64,
        "frames": [
            {
                "frame_id": "normal_f000001",
                "image": "normal frame_f000001.jpg",
                "source_video": "normal.mp4",
                "frame_index": 1,
                "width": 1920,
                "height": 1080,
                "predictions": [
                    {
                        "class_id": 0,
                        "xc": 0.5,
                        "yc": 0.5,
                        "width": 0.2,
                        "height": 0.1,
                    }
                ],
            },
            {
                "frame_id": "validation_f000001",
                "image": "validation_f000001.jpg",
                "source_video": "validation.mp4",
                "frame_index": 1,
                "width": 1920,
                "height": 1080,
                "predictions": [],
            },
        ],
    }
    frame_manifest_path(config, campaign_id).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return config, root


def native_export(
    *,
    include_second: bool = True,
    cancel_second: bool = False,
) -> list[dict]:
    tasks = [
        {
            "data": {"frame_id": "normal_f000001"},
            "annotations": [
                {
                    "was_cancelled": False,
                    "result": [
                        {
                            "type": "rectanglelabels",
                            "value": {
                                "x": 40.0,
                                "y": 45.0,
                                "width": 20.0,
                                "height": 10.0,
                                "rectanglelabels": ["objeto"],
                            },
                        }
                    ],
                }
            ],
        }
    ]
    if include_second:
        tasks.append(
            {
                "data": {"frame_id": "validation_f000001"},
                "annotations": [
                    {
                        "was_cancelled": cancel_second,
                        "result": [],
                    }
                ],
            }
        )
    return tasks


def test_workspace_abstraction(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    assert ws.sources_root == tmp_path / "dataset" / "sources"
    assert ws.versions_root == tmp_path / "dataset" / "versions"


def test_uniform_campaign_needs_no_detection_model(tmp_path: Path):
    config = defaults(tmp_path)
    videos = Path(config["paths"]["videos_dir"])
    videos.mkdir(parents=True, exist_ok=True)
    (videos / "bootstrap.mp4").write_bytes(b"video")
    extraction = dict(config["extraction"])
    extraction.update({"mode": "uniform", "model": None})
    annotation = dict(config["annotation"])
    annotation["model"] = None
    create_campaign(
        config,
        campaign_id="bootstrap",
        videos_dir=videos,
        video_pattern="*.mp4",
        extraction=extraction,
        annotation=annotation,
    )

    campaign = load_campaign(config, "bootstrap")
    assert campaign["annotation"]["backend"] == "none"


def test_import_json_contains_stable_ids_paths_and_predictions(tmp_path: Path):
    config, _ = create_fixture_campaign(tmp_path)

    output = build_import_tasks(config, "capture_test")
    tasks = json.loads(output.read_text(encoding="utf-8"))

    assert len(tasks) == 2
    assert tasks[0]["data"]["frame_id"] == "normal_f000001"
    assert tasks[0]["data"]["source_video"] == "normal.mp4"
    assert tasks[0]["data"]["image"].startswith("/data/local-files/?d=")
    assert "normal frame_f000001.jpg" in tasks[0]["data"]["image"]
    assert "%20" not in tasks[0]["data"]["image"]
    assert len(tasks[0]["predictions"][0]["result"]) == 1
    region_id = tasks[0]["predictions"][0]["result"][0]["id"]
    assert re.fullmatch(r"[A-Za-z0-9_-]+", region_id)
    assert tasks[1]["predictions"][0]["result"] == []


def test_label_studio_region_id_is_valid_and_stable_with_spaces():
    source = "2026-07-17 15-31-41_f001290"
    first = label_studio_region_id(source, 0)

    assert first == label_studio_region_id(source, 0)
    assert first != label_studio_region_id(source, 1)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", first)


def test_empty_human_annotation_is_confirmed_negative(tmp_path: Path):
    config, _ = create_fixture_campaign(tmp_path)
    export = tmp_path / "export.json"
    export.write_text(json.dumps(native_export()), encoding="utf-8")

    annotations, report = parse_native_export(config, "capture_test", export)

    assert annotations["validation_f000001"].is_negative
    assert annotations["validation_f000001"].boxes == ()
    assert report["confirmed_negatives"] == 1
    assert report["tasks_valid"] == 2


def test_missing_task_blocks_and_cancelled_task_is_excluded(tmp_path: Path):
    config, _ = create_fixture_campaign(tmp_path)
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(native_export(include_second=False)), encoding="utf-8")
    cancelled = tmp_path / "cancelled.json"
    cancelled.write_text(
        json.dumps(native_export(cancel_second=True)), encoding="utf-8"
    )

    with pytest.raises(WorkflowError, match="tasks ausentes"):
        parse_native_export(config, "capture_test", missing)
    annotations, report = parse_native_export(config, "capture_test", cancelled)
    assert annotations["validation_f000001"].excluded
    assert not annotations["validation_f000001"].is_negative
    assert report["tasks_excluded"] == 1
    assert report["exclusion_reasons"] == {"skipped_or_cancelled": 1}


def test_export_inspection_adjusts_roundoff_and_lists_real_boundary_errors(tmp_path: Path):
    config, _ = create_fixture_campaign(tmp_path)
    payload = native_export()
    task = payload[0]
    task.update({"id": 6132, "project": 2})
    task["data"]["image"] = "/data/local-files/?d=normal.jpg"
    region = task["annotations"][0]["result"][0]
    region["id"] = "region-at-border"
    region["value"]["x"] = -0.00005
    export = tmp_path / "roundoff.json"
    export.write_text(json.dumps(payload), encoding="utf-8")

    annotations, _ = parse_native_export(config, "capture_test", export)
    inspection = inspect_native_export(config, "capture_test", export)

    assert annotations["normal_f000001"].boxes[0].startswith("0 ")
    assert inspection["valid"]
    assert inspection["warnings"] == 1
    assert inspection["issues"][0]["code"] == "box_boundary_adjusted"
    assert inspection["issues"][0]["task_id"] == 6132

    region["value"]["x"] = -0.1
    export.write_text(json.dumps(payload), encoding="utf-8")
    invalid = inspect_native_export(config, "capture_test", export)

    assert not invalid["valid"]
    assert invalid["errors"] == 1
    assert invalid["issues"][0]["code"] == "box_out_of_bounds"
    assert "redimensione" in invalid["issues"][0]["resolution"]


def test_partial_annotation_revisions_are_append_only_and_reusable(tmp_path: Path):
    config, campaign = create_fixture_campaign(tmp_path)
    deferred_id = "normal_f000002"
    (campaign / "frames" / "raw" / "images" / f"{deferred_id}.jpg").write_bytes(
        b"deferred-image"
    )
    manifest_payload = json.loads(
        frame_manifest_path(config, "capture_test").read_text(encoding="utf-8")
    )
    manifest_payload["frames"].append(
        {
            "frame_id": deferred_id,
            "image": f"{deferred_id}.jpg",
            "source_video": "normal.mp4",
            "frame_index": 2,
            "width": 1920,
            "height": 1080,
            "predictions": [],
        }
    )
    frame_manifest_path(config, "capture_test").write_text(
        json.dumps(manifest_payload), encoding="utf-8"
    )
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps(native_export()), encoding="utf-8")

    accepted, report_path = accept_native_export(
        config,
        "capture_test",
        partial,
        revision_id="r001_partial",
        allow_pending=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert accepted == annotation_revision_export_path(
        config, "capture_test", "r001_partial"
    )
    assert report["snapshot_type"] == "provisional"
    assert report["tasks_completed"] == 2
    assert report["tasks_deferred"] == 1
    assert report["per_video"]["normal.mp4"]["deferred"] == 1
    assert list_annotation_revisions(config, "capture_test") == ["r001_partial"]

    create_release(
        config,
        release_id="partial_release",
        campaign_ids=["capture_test"],
        annotation_revisions={"capture_test": "r001_partial"},
        assignments={
            "train": ["capture_test/normal.mp4"],
            "val": ["capture_test/validation.mp4"],
            "test_normal": [],
            "test_stress": [],
        },
        evaluation_level="pilot",
    )
    partial_manifest = build_release(config, "partial_release")
    partial_report = json.loads(
        (partial_manifest.parent / "build_report.json").read_text(encoding="utf-8")
    )
    assert partial_report["exclusion_reasons"] == {"deferred": 1}
    assert deferred_id in partial_manifest.read_text(encoding="utf-8")


def test_annotation_revisions_are_ordered_by_validation_time_not_name(tmp_path: Path):
    config, _campaign = create_fixture_campaign(tmp_path)
    export = tmp_path / "annotations.json"
    export.write_text(json.dumps(native_export()), encoding="utf-8")

    _, older_report_path = accept_native_export(
        config, "capture_test", export, revision_id="rev_z_older"
    )
    _, newer_report_path = accept_native_export(
        config, "capture_test", export, revision_id="rev_a_newer"
    )

    older_report = json.loads(older_report_path.read_text(encoding="utf-8"))
    older_report["validated_at"] = "2026-07-22T10:00:00+00:00"
    older_report_path.write_text(json.dumps(older_report), encoding="utf-8")
    newer_report = json.loads(newer_report_path.read_text(encoding="utf-8"))
    newer_report["validated_at"] = "2026-07-23T10:00:00+00:00"
    newer_report_path.write_text(json.dumps(newer_report), encoding="utf-8")

    assert list_annotation_revisions(config, "capture_test") == [
        "rev_z_older",
        "rev_a_newer",
    ]


def test_release_is_built_from_accepted_json_and_split_by_whole_video(tmp_path: Path):
    config, campaign = create_fixture_campaign(tmp_path)
    extra_id = "normal_f000002"
    (campaign / "frames" / "raw" / "images" / f"{extra_id}.jpg").write_bytes(
        b"excluded-image"
    )
    manifest_payload = json.loads(
        frame_manifest_path(config, "capture_test").read_text(encoding="utf-8")
    )
    manifest_payload["frames"].append(
        {
            "frame_id": extra_id,
            "image": f"{extra_id}.jpg",
            "source_video": "normal.mp4",
            "frame_index": 2,
            "width": 1920,
            "height": 1080,
            "predictions": [],
        }
    )
    frame_manifest_path(config, "capture_test").write_text(
        json.dumps(manifest_payload), encoding="utf-8"
    )
    export = tmp_path / "export.json"
    tasks = native_export()
    tasks.append(
        {
            "data": {"frame_id": extra_id},
            "annotations": [
                {
                    "was_cancelled": True,
                    "result": [],
                }
            ],
        }
    )
    export.write_text(json.dumps(tasks), encoding="utf-8")
    accept_native_export(config, "capture_test", export)
    create_release(
        config,
        release_id="dataset_v1",
        campaign_ids=["capture_test"],
        assignments={
            "train": ["capture_test/normal.mp4"],
            "val": ["capture_test/validation.mp4"],
            "test_normal": [],
            "test_stress": [],
        },
        evaluation_level="pilot",
    )

    manifest = build_release(config, "dataset_v1")
    release = manifest.parent

    assert (release / "images" / "train" / "capture_test__normal_f000001.jpg").exists()
    train_label = release / "labels" / "train" / "capture_test__normal_f000001.txt"
    val_label = release / "labels" / "val" / "capture_test__validation_f000001.txt"
    assert train_label.read_text(encoding="utf-8").startswith("0 0.500000 0.500000")
    assert val_label.read_text(encoding="utf-8") == ""
    assert not (release / "images" / "train" / f"capture_test__{extra_id}.jpg").exists()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        excluded_row = next(
            row for row in csv.DictReader(handle) if row["frame_id"] == extra_id
        )
    assert excluded_row["source_video"] == "normal.mp4"
    assert excluded_row["split"] == "train"
    assert excluded_row["included"] == "false"
    assert excluded_row["exclusion_reason"] == "skipped_or_cancelled"
    report = json.loads((release / "build_report.json").read_text(encoding="utf-8"))
    assert report["excluded_frames"] == 1
    assert (release / "data.yaml").exists()
    with pytest.raises(WorkflowError, match="imutavel"):
        build_release(config, "dataset_v1")


def test_standard_release_requires_independent_train_val_and_test_units(
    tmp_path: Path,
):
    config, _campaign = create_fixture_campaign(tmp_path)
    export = tmp_path / "export.json"
    export.write_text(json.dumps(native_export()), encoding="utf-8")
    accept_native_export(config, "capture_test", export)

    with pytest.raises(WorkflowError, match="train, val e test_normal"):
        create_release(
            config,
            release_id="invalid_standard",
            campaign_ids=["capture_test"],
            assignments={
                "train": ["capture_test/normal.mp4"],
                "val": ["capture_test/validation.mp4"],
                "test_normal": [],
                "test_stress": [],
            },
        )


def test_single_capture_unit_can_materialize_an_explicit_pilot(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)
    videos = ws.videos_root
    videos.mkdir()
    (videos / "single.mp4").write_bytes(b"single-video")
    create_campaign(
        ws,
        campaign_id="single_source",
        videos_dir=videos,
        video_pattern="*.mp4",
        annotation={"classes": ["objeto"]},
    )
    image_root = ws.source_root("single_source") / "frames" / "raw" / "images"
    (image_root / "single_f000001.jpg").write_bytes(b"image")
    frame_manifest_path(ws, "single_source").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frames": [
                    {
                        "frame_id": "single_f000001",
                        "image": "single_f000001.jpg",
                        "source_video": "single.mp4",
                        "frame_index": 1,
                        "width": 100,
                        "height": 100,
                        "predictions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    export = tmp_path / "single_export.json"
    export.write_text(
        json.dumps(
            [
                {
                    "data": {"frame_id": "single_f000001"},
                    "annotations": [{"was_cancelled": False, "result": []}],
                }
            ]
        ),
        encoding="utf-8",
    )
    accept_native_export(
        ws,
        "single_source",
        export,
        revision_id="single_revision",
    )
    create_release(
        ws,
        release_id="single_pilot",
        campaign_ids=["single_source"],
        annotation_revisions={"single_source": "single_revision"},
        assignments={
            "train": ["single_source/single.mp4"],
            "val": [],
            "test_normal": [],
            "test_stress": [],
        },
        evaluation_level="pilot",
    )

    manifest = build_release(ws, "single_pilot")
    data = load_yaml(manifest.parent / "data.yaml")
    report = json.loads(
        (manifest.parent / "build_report.json").read_text(encoding="utf-8")
    )
    assert data["val"] == "images/train"
    assert report["evaluation_level"] == "pilot"
    assert report["provisional"] is True
    assert any("não comprovam generalização" in item for item in report["warnings"])
