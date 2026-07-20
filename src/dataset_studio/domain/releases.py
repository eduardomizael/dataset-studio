"""Regras de domínio para criação, validação e materialização de releases."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_studio.domain.annotations import parse_native_export
from dataset_studio.domain.campaigns import (
    annotation_source_paths,
    campaign_root,
    load_annotation_revision_report,
    load_campaign,
    load_frame_manifest,
)
from dataset_studio.domain.errors import WorkflowError
from dataset_studio.domain.workspace import (
    SPLITS,
    Workspace,
    dump_yaml,
    load_yaml,
    sha256,
    utc_now,
    validate_id,
)


def releases_root(defaults_or_ws: dict[str, Any] | Workspace) -> Path:
    if isinstance(defaults_or_ws, Workspace):
        return defaults_or_ws.releases_root
    p = Path(defaults_or_ws["paths"]["releases_root"])
    return p if p.is_absolute() else p.resolve()


def release_root(defaults_or_ws: dict[str, Any] | Workspace, release_id: str) -> Path:
    return releases_root(defaults_or_ws) / release_id


def release_config_path(defaults_or_ws: dict[str, Any] | Workspace, release_id: str) -> Path:
    return release_root(defaults_or_ws, release_id) / "release.yaml"


def list_releases(defaults_or_ws: dict[str, Any] | Workspace) -> list[str]:
    root = releases_root(defaults_or_ws)
    if not root.exists():
        return []
    return sorted(
        path.name for path in root.iterdir() if (path / "release.yaml").is_file()
    )


def campaign_video_keys(defaults_or_ws: dict[str, Any] | Workspace, campaign_id: str) -> list[str]:
    manifest = load_frame_manifest(defaults_or_ws, campaign_id)
    videos = sorted({frame["source_video"] for frame in manifest["frames"]})
    return [f"{campaign_id}/{video}" for video in videos]


def create_release(
    defaults_or_ws: dict[str, Any] | Workspace,
    *,
    release_id: str,
    campaign_ids: list[str],
    assignments: dict[str, list[str]],
    annotation_revisions: dict[str, str] | None = None,
) -> Path:
    validate_id(release_id, "release_id")
    root = release_root(defaults_or_ws, release_id)
    if root.exists():
        raise WorkflowError(f"A release ja existe: {root}")
    if not campaign_ids:
        raise WorkflowError("Selecione pelo menos uma campanha.")
    if len(campaign_ids) != len(set(campaign_ids)):
        raise WorkflowError("Uma campanha foi selecionada mais de uma vez.")
    if annotation_revisions is not None and set(annotation_revisions) != set(
        campaign_ids
    ):
        raise WorkflowError(
            "annotation_revisions deve indicar exatamente uma revisao por campanha."
        )

    expected: set[str] = set()
    first_campaign = load_campaign(defaults_or_ws, campaign_ids[0])
    defaults_classes = list(first_campaign["annotation"]["classes"])

    selected_revisions: dict[str, str] = {}
    provisional = False
    for campaign_id in campaign_ids:
        campaign = load_campaign(defaults_or_ws, campaign_id)
        if list(campaign["annotation"]["classes"]) != defaults_classes:
            raise WorkflowError(f"Classes divergentes na campanha: {campaign_id}")

        revision_id = (
            annotation_revisions.get(campaign_id)
            if annotation_revisions is not None
            else "legacy"
        )
        if not revision_id:
            raise WorkflowError(
                f"Selecione uma revisao de anotacao para {campaign_id}."
            )
        report = load_annotation_revision_report(defaults_or_ws, campaign_id, revision_id)
        selected_revisions[campaign_id] = revision_id
        provisional = provisional or report.get("snapshot_type") == "provisional"
        expected.update(campaign_video_keys(defaults_or_ws, campaign_id))

    assigned = [key for split in SPLITS for key in assignments.get(split, [])]
    if len(assigned) != len(set(assigned)):
        raise WorkflowError("Um video foi atribuido a mais de um split.")
    if set(assigned) != expected:
        raise WorkflowError("Todos os videos devem ser atribuidos exatamente uma vez.")
    if not assignments.get("train") or not assignments.get("val"):
        raise WorkflowError("A release exige ao menos um video em train e val.")

    root.mkdir(parents=True)
    payload = {
        "schema_version": 2,
        "release_id": release_id,
        "created_at": utc_now(),
        "campaigns": campaign_ids,
        "annotation_revisions": selected_revisions,
        "provisional": provisional,
        "assignments": {split: assignments.get(split, []) for split in SPLITS},
        "materialization": "copy",
        "classes": defaults_classes,
    }
    path = root / "release.yaml"
    dump_yaml(path, payload)
    return path


def build_release(defaults_or_ws: dict[str, Any] | Workspace, release_id: str) -> Path:
    root = release_root(defaults_or_ws, release_id)
    release = load_yaml(root / "release.yaml")
    manifest_path = root / "manifest.csv"
    if manifest_path.exists():
        raise WorkflowError("A release ja foi materializada e e imutavel.")
    split_by_video = {
        key: split
        for split, keys in release["assignments"].items()
        for key in keys
    }

    rows: list[dict[str, str]] = []
    operations: list[tuple[Path, Path, str]] = []
    for campaign_id in release["campaigns"]:
        manifest = load_frame_manifest(defaults_or_ws, campaign_id)
        revision_id = release.get("annotation_revisions", {}).get(
            campaign_id, "legacy"
        )
        export_path, _ = annotation_source_paths(
            defaults_or_ws, campaign_id, revision_id
        )
        revision_report = load_annotation_revision_report(
            defaults_or_ws, campaign_id, revision_id
        )
        annotations, _ = parse_native_export(
            defaults_or_ws,
            campaign_id,
            export_path,
            allow_pending=bool(revision_report.get("allow_pending", False)),
        )
        images_dir = campaign_root(defaults_or_ws, campaign_id) / "frames" / "raw" / "images"
        for frame in manifest["frames"]:
            frame_id = frame["frame_id"]
            split = split_by_video[f"{campaign_id}/{frame['source_video']}"]
            source_image = images_dir / frame["image"]
            if not source_image.exists():
                raise WorkflowError(f"Imagem ausente ao materializar: {source_image}")
            annotation = annotations[frame_id]
            if annotation.excluded:
                rows.append(
                    {
                        "campaign_id": campaign_id,
                        "frame_id": frame_id,
                        "source_video": frame["source_video"],
                        "frame_index": str(frame["frame_index"]),
                        "split": split,
                        "included": "false",
                        "exclusion_reason": str(annotation.exclusion_reason),
                        "image": "",
                        "label": "",
                        "boxes": "0",
                        "source_image_sha256": sha256(source_image),
                    }
                )
                continue
            output_stem = f"{campaign_id}__{frame_id}"
            destination_image = root / "images" / split / f"{output_stem}.jpg"
            destination_label = root / "labels" / split / f"{output_stem}.txt"
            label_text = "\n".join(annotation.boxes)
            if label_text:
                label_text += "\n"
            operations.append((source_image, destination_image, label_text))
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "frame_id": frame_id,
                    "source_video": frame["source_video"],
                    "frame_index": str(frame["frame_index"]),
                    "split": split,
                    "included": "true",
                    "exclusion_reason": "",
                    "image": destination_image.relative_to(root).as_posix(),
                    "label": destination_label.relative_to(root).as_posix(),
                    "boxes": str(len(annotation.boxes)),
                    "source_image_sha256": sha256(source_image),
                }
            )

    included_rows = [row for row in rows if row["included"] == "true"]
    included_splits = Counter(row["split"] for row in included_rows)
    for required_split in ("train", "val"):
        if not included_splits[required_split]:
            raise WorkflowError(
                f"O split {required_split} ficou sem frames utilizaveis apos as exclusoes."
            )

    for source_image, destination_image, label_text in operations:
        destination_label = (
            root
            / "labels"
            / destination_image.parent.name
            / f"{destination_image.stem}.txt"
        )
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination_image)
        destination_label.write_text(label_text, encoding="utf-8")

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    names = {index: name for index, name in enumerate(release["classes"])}
    data_yaml = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    if release["assignments"].get("test_normal"):
        data_yaml["test"] = "images/test_normal"
    dump_yaml(root / "data.yaml", data_yaml)
    if release["assignments"].get("test_stress"):
        stress_yaml = dict(data_yaml)
        stress_yaml["val"] = "images/test_stress"
        stress_yaml["test"] = "images/test_stress"
        dump_yaml(root / "data_test_stress.yaml", stress_yaml)
    summary = {
        "schema_version": 1,
        "release_id": release_id,
        "provisional": bool(release.get("provisional", False)),
        "annotation_revisions": release.get("annotation_revisions", {}),
        "built_at": utc_now(),
        "source_frames": len(rows),
        "images": len(included_rows),
        "excluded_frames": len(rows) - len(included_rows),
        "exclusion_reasons": dict(
            Counter(
                row["exclusion_reason"]
                for row in rows
                if row["included"] == "false"
            )
        ),
        "boxes": sum(int(row["boxes"]) for row in included_rows),
        "splits": dict(included_splits),
        "manifest_sha256": sha256(manifest_path),
    }
    (root / "build_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path
