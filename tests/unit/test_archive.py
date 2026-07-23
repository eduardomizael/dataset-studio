from __future__ import annotations

from pathlib import Path

import pytest

from dataset_studio.application import (
    archive_status,
    attach_archive_to_dataset,
    attach_archive_to_run,
    import_archive_snapshot,
    materialize_archive_snapshot,
    verify_archive_snapshot,
)
from dataset_studio.domain import (
    Workspace,
    WorkflowError,
    dump_yaml,
    load_yaml,
    register_dataset,
    register_run,
    validate_registry,
)


def create_source(root: Path) -> Path:
    source = root / "source"
    (source / "images").mkdir(parents=True)
    (source / "empty").mkdir()
    (source / "images" / "one.jpg").write_bytes(b"same-content")
    (source / "images" / "two.jpg").write_bytes(b"same-content")
    (source / "label.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return source


def test_archive_deduplicates_verifies_and_materializes(tmp_path: Path):
    ws = Workspace.from_path(tmp_path / "workspace")
    source = create_source(tmp_path)

    snapshot = import_archive_snapshot(ws, "legacy-one", source)

    assert snapshot["files"] == 3
    assert snapshot["objects_added"] == 2
    assert snapshot["objects_reused"] == 1
    assert verify_archive_snapshot(
        ws, "legacy-one", source_root=source
    )["valid"] is True
    status = archive_status(ws)
    assert status["snapshot_count"] == 1
    assert status["object_count"] == 2
    assert status["physical_bytes"] < status["logical_bytes"]

    destination = tmp_path / "restored"
    materialize_archive_snapshot(ws, "legacy-one", destination)
    assert (destination / "images" / "one.jpg").read_bytes() == b"same-content"
    assert (destination / "images" / "two.jpg").read_bytes() == b"same-content"
    assert (destination / "empty").is_dir()


def test_archive_is_immutable_and_detects_changes(tmp_path: Path):
    ws = Workspace.from_path(tmp_path / "workspace")
    source = create_source(tmp_path)
    import_archive_snapshot(ws, "legacy-one", source)
    (source / "label.txt").write_text("changed", encoding="utf-8")

    verification = verify_archive_snapshot(
        ws, "legacy-one", source_root=source
    )

    assert verification["valid"] is False
    assert any("divergente na origem" in item for item in verification["errors"])
    with pytest.raises(WorkflowError, match="diverge da origem"):
        import_archive_snapshot(ws, "legacy-one", source)


def test_archive_detects_corrupted_object(tmp_path: Path):
    ws = Workspace.from_path(tmp_path / "workspace")
    source = create_source(tmp_path)
    import_archive_snapshot(ws, "legacy-one", source)
    objects_root = ws.archive_root / "objects"
    object_path = next(path for path in objects_root.rglob("*") if path.is_file())
    object_path.write_bytes(b"corrupted")

    verification = verify_archive_snapshot(ws, "legacy-one")

    assert verification["valid"] is False
    with pytest.raises(WorkflowError, match="integridade"):
        materialize_archive_snapshot(ws, "legacy-one", tmp_path / "restored")


def test_archive_can_replace_external_dataset_paths(tmp_path: Path):
    ws = Workspace.from_path(tmp_path / "workspace")
    source = create_source(tmp_path)
    import_archive_snapshot(ws, "legacy-one", source)
    dataset_path = ws.registry_root / "datasets" / "dataset-one.yaml"
    dump_yaml(
        dataset_path,
        {
            "schema_version": 1,
            "dataset_id": "dataset-one",
            "paths": {"snapshot": "C:/old/location"},
            "provenance": {
                "origin": "reconstructed",
                "confidence": "probable",
                "evidence": [],
            },
        },
    )

    attached = attach_archive_to_dataset(
        ws,
        "dataset-one",
        "legacy-one",
        subpaths=["images"],
    )

    assert attached["legacy_paths"]["snapshot"] == "C:/old/location"
    assert attached["paths"]["archive_manifest"].endswith("manifest.csv")
    assert load_yaml(dataset_path)["physical_archive"]["snapshot_id"] == "legacy-one"
    assert validate_registry(ws)["valid"] is True


def test_archive_can_be_attached_to_run(tmp_path: Path):
    ws = Workspace.from_path(tmp_path / "workspace")
    source = create_source(tmp_path)
    import_archive_snapshot(ws, "legacy-one", source)
    register_dataset(
        ws,
        {
            "schema_version": 1,
            "dataset_id": "dataset-one",
            "provenance": {
                "origin": "reconstructed",
                "confidence": "confirmed",
                "evidence": [],
            },
        },
    )
    register_run(
        ws,
        {
            "schema_version": 1,
            "run_id": "run-one",
            "dataset_id": "dataset-one",
            "provenance": {
                "origin": "reconstructed",
                "confidence": "confirmed",
                "evidence": [],
            },
        },
    )

    attached = attach_archive_to_run(
        ws,
        "run-one",
        "legacy-one",
        subpaths=["images"],
    )

    assert attached["physical_archive"]["subpaths"] == ["images"]
    assert load_yaml(
        ws.registry_root / "runs" / "run-one.yaml"
    )["physical_archive"]["snapshot_id"] == "legacy-one"
    assert validate_registry(ws)["valid"] is True

    first_record = load_yaml(ws.registry_root / "runs" / "run-one.yaml")
    attach_archive_to_run(
        ws,
        "run-one",
        "legacy-one",
        subpaths=["images"],
    )
    assert load_yaml(ws.registry_root / "runs" / "run-one.yaml") == first_record
