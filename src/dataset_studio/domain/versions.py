"""Regras de domínio para criação, validação e materialização de versões de dataset (versions)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_studio.domain.annotations import parse_native_export
from dataset_studio.domain.sources import (
    annotation_source_paths,
    list_annotation_revisions,
    load_annotation_revision_report,
    load_frame_manifest,
    load_source,
    source_root,
)
from dataset_studio.domain.errors import WorkflowError
from dataset_studio.domain.registry import unregister_dataset
from dataset_studio.domain.workspace import (
    SPLITS,
    Workspace,
    dump_yaml,
    load_yaml,
    sha256,
    utc_now,
    validate_id,
)


def versions_root(defaults_or_ws: dict[str, Any] | Workspace) -> Path:
    """Retorna o diretório raiz de versões/releases salvas no workspace."""
    if isinstance(defaults_or_ws, Workspace):
        return defaults_or_ws.versions_root
    p = Path(defaults_or_ws["paths"].get("versions_root") or defaults_or_ws["paths"].get("releases_root"))
    return p if p.is_absolute() else p.resolve()



def version_root(defaults_or_ws: dict[str, Any] | Workspace, version_id: str) -> Path:
    return versions_root(defaults_or_ws) / version_id


def version_config_path(defaults_or_ws: dict[str, Any] | Workspace, version_id: str) -> Path:
    p = version_root(defaults_or_ws, version_id)
    if (p / "version.yaml").is_file():
        return p / "version.yaml"
    return p / "release.yaml"


def list_versions(defaults_or_ws: dict[str, Any] | Workspace) -> list[str]:
    """Lista todos os identificadores de versões ativas no workspace."""
    root = versions_root(defaults_or_ws)
    if not root.exists():
        return []
    return sorted(
        path.name for path in root.iterdir()
        if (path / "version.yaml").is_file() or (path / "release.yaml").is_file()
    )



def source_video_keys(defaults_or_ws: dict[str, Any] | Workspace, source_id: str) -> list[str]:
    """Retorna as chaves únicas identificadoras dos vídeos contidos em uma fonte."""
    manifest = load_frame_manifest(defaults_or_ws, source_id)
    videos = sorted({frame["source_video"] for frame in manifest["frames"]})
    return [f"{source_id}/{video}" for video in videos]



def create_version(
    defaults_or_ws: dict[str, Any] | Workspace,
    *,
    version_id: str | None = None,
    release_id: str | None = None,
    source_ids: list[str] | None = None,
    campaign_ids: list[str] | None = None,
    assignments: dict[str, list[str]],
    annotation_revisions: dict[str, str] | None = None,
) -> Path:
    """Cria o manifesto inicial de configuração para uma nova versão de dataset."""
    target_id = version_id or release_id
    if not target_id:
        raise WorkflowError("Identificador da versao obrigatorio.")
    validate_id(target_id, "version_id")

    targets = source_ids or campaign_ids
    if not targets:
        raise WorkflowError("Selecione pelo menos uma origem.")
    if len(targets) != len(set(targets)):
        raise WorkflowError("Uma origem foi selecionada mais de uma vez.")
    if annotation_revisions is not None and set(annotation_revisions) != set(targets):
        raise WorkflowError(
            "annotation_revisions deve indicar exatamente uma revisao por origem."
        )

    root = version_root(defaults_or_ws, target_id)
    if root.exists():
        raise WorkflowError(f"A versao ja existe: {root}")

    expected: set[str] = set()
    first_source = load_source(defaults_or_ws, targets[0])
    defaults_classes = list(first_source["annotation"]["classes"])

    selected_revisions: dict[str, str] = {}
    provisional = False
    for src_id in targets:
        source = load_source(defaults_or_ws, src_id)
        if list(source["annotation"]["classes"]) != defaults_classes:
            raise WorkflowError(f"Classes divergentes na origem: {src_id}")

        revs = list_annotation_revisions(defaults_or_ws, src_id)
        if annotation_revisions is not None and annotation_revisions.get(src_id):
            revision_id = annotation_revisions[src_id]
        elif revs:
            revision_id = revs[-1]
        else:
            revision_id = "legacy"

        if not revision_id:
            raise WorkflowError(
                f"Selecione uma revisao de anotacao para {src_id}."
            )
        report = load_annotation_revision_report(defaults_or_ws, src_id, revision_id)
        selected_revisions[src_id] = revision_id
        provisional = provisional or report.get("snapshot_type") == "provisional"
        expected.update(source_video_keys(defaults_or_ws, src_id))

    assigned = [key for split in SPLITS for key in assignments.get(split, [])]
    if len(assigned) != len(set(assigned)):
        raise WorkflowError("Um video foi atribuido a mais de um split.")
    if set(assigned) != expected:
        raise WorkflowError("Todos os videos devem ser atribuidos exatamente uma vez.")
    if not assignments.get("train") or not assignments.get("val"):
        raise WorkflowError("A versao exige ao menos um video em train e val.")

    root.mkdir(parents=True)
    payload = {
        "schema_version": 2,
        "version_id": target_id,
        "release_id": target_id,
        "created_at": utc_now(),
        "sources": targets,
        "campaigns": targets,
        "annotation_revisions": selected_revisions,
        "provisional": provisional,
        "assignments": {split: assignments.get(split, []) for split in SPLITS},
        "materialization": "copy",
        "classes": defaults_classes,
    }
    path = root / "version.yaml"
    dump_yaml(path, payload)
    return path


def _materialize_version(
    defaults_or_ws: dict[str, Any] | Workspace,
    version_id: str,
    *,
    root: Path,
    config_p: Path,
    final_root: Path,
) -> Path:
    version = load_yaml(config_p)
    manifest_path = root / "manifest.csv"
    split_by_video = {
        key: split
        for split, keys in version["assignments"].items()
        for key in keys
    }

    rows: list[dict[str, str]] = []
    operations: list[tuple[Path, Path, str]] = []
    sources_list = version.get("sources") or version.get("campaigns", [])
    for src_id in sources_list:
        manifest = load_frame_manifest(defaults_or_ws, src_id)
        revision_id = version.get("annotation_revisions", {}).get(src_id, "legacy")
        export_path, _ = annotation_source_paths(defaults_or_ws, src_id, revision_id)
        revision_report = load_annotation_revision_report(defaults_or_ws, src_id, revision_id)
        annotations, _ = parse_native_export(
            defaults_or_ws,
            src_id,
            export_path,
            allow_pending=bool(revision_report.get("allow_pending", False)),
        )
        images_dir = source_root(defaults_or_ws, src_id) / "frames" / "raw" / "images"
        for frame in manifest["frames"]:
            frame_id = frame["frame_id"]
            split = split_by_video[f"{src_id}/{frame['source_video']}"]
            source_image = images_dir / frame["image"]
            if not source_image.exists():
                raise WorkflowError(f"Imagem ausente ao materializar: {source_image}")
            annotation = annotations[frame_id]
            if annotation.excluded:
                rows.append(
                    {
                        "source_id": src_id,
                        "campaign_id": src_id,
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
                        "image_sha256": "",
                        "label_sha256": "",
                    }
                )
                continue
            output_stem = f"{src_id}__{frame_id}"
            destination_image = root / "images" / split / f"{output_stem}.jpg"
            destination_label = root / "labels" / split / f"{output_stem}.txt"
            label_text = "\n".join(annotation.boxes)
            if label_text:
                label_text += "\n"
            operations.append((source_image, destination_image, label_text))
            rows.append(
                {
                    "source_id": src_id,
                    "campaign_id": src_id,
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
                    "image_sha256": sha256(source_image),
                    "label_sha256": hashlib.sha256(label_text.encode("utf-8")).hexdigest(),
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

    names = {index: name for index, name in enumerate(version["classes"])}
    data_yaml = {
        "path": str(final_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    if version["assignments"].get("test_normal"):
        data_yaml["test"] = "images/test_normal"
    dump_yaml(root / "data.yaml", data_yaml)
    if version["assignments"].get("test_stress"):
        stress_yaml = dict(data_yaml)
        stress_yaml["val"] = "images/test_stress"
        stress_yaml["test"] = "images/test_stress"
        dump_yaml(root / "data_test_stress.yaml", stress_yaml)
    summary = {
        "schema_version": 1,
        "version_id": version_id,
        "release_id": version_id,
        "provisional": bool(version.get("provisional", False)),
        "annotation_revisions": version.get("annotation_revisions", {}),
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
        "version_config_sha256": sha256(config_p),
    }
    (root / "build_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


def build_version(defaults_or_ws: dict[str, Any] | Workspace, version_id: str) -> Path:
    """Materializa a versão em staging e a publica somente após sucesso integral."""
    final_root = version_root(defaults_or_ws, version_id)
    config_p = version_config_path(defaults_or_ws, version_id)
    if (final_root / "manifest.csv").exists():
        raise WorkflowError("A versao ja foi materializada e e imutavel.")
    if not config_p.is_file():
        raise WorkflowError(f"Configuracao da versao nao encontrada: {config_p}")

    parent = final_root.parent
    token = uuid.uuid4().hex
    staging_root = parent / f".{version_id}.build-{token}"
    backup_root = parent / f".{version_id}.backup-{token}"
    staging_root.mkdir(parents=True)
    staging_config = staging_root / "version.yaml"
    shutil.copy2(config_p, staging_config)
    try:
        _materialize_version(
            defaults_or_ws,
            version_id,
            root=staging_root,
            config_p=staging_config,
            final_root=final_root,
        )
        final_root.rename(backup_root)
        try:
            staging_root.rename(final_root)
        except Exception:
            backup_root.rename(final_root)
            raise
        shutil.rmtree(backup_root, ignore_errors=False)
        return final_root / "manifest.csv"
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise


def delete_version(defaults_or_ws: dict[str, Any] | Workspace, version_id: str) -> None:
    """Remove completamente a versão; treinamentos são recursos independentes."""
    root = version_root(defaults_or_ws, version_id)
    if not root.exists():
        raise WorkflowError(f"Versão não encontrada: {version_id}")
    shutil.rmtree(root, ignore_errors=False)
    if isinstance(defaults_or_ws, Workspace):
        unregister_dataset(defaults_or_ws, version_id)


# Aliases de retrocompatibilidade para release -> version
releases_root = versions_root
release_root = version_root
release_config_path = version_config_path
list_releases = list_versions
create_release = create_version
delete_release = delete_version
campaign_video_keys = source_video_keys
create_release = create_version
build_release = build_version
