"""Arquivo físico deduplicado para snapshots históricos de datasets."""

from __future__ import annotations

import csv
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

from dataset_studio.domain import (
    Workspace,
    WorkflowError,
    dataset_registry_path,
    dump_yaml,
    load_yaml,
    register_dataset,
    sha256,
    utc_now,
    validate_id,
)

MANIFEST_FIELDS = ("kind", "path", "size_bytes", "sha256")


def _snapshot_root(ws: Workspace, snapshot_id: str) -> Path:
    validate_id(snapshot_id, "snapshot_id")
    return ws.archive_root / "snapshots" / snapshot_id


def _object_path(ws: Workspace, digest: str) -> Path:
    return ws.archive_root / "objects" / digest[:2] / digest


def _iter_entries(source_root: Path) -> Iterable[tuple[str, Path]]:
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise WorkflowError(f"Links simbólicos não são aceitos: {path}")
        kind = "directory" if path.is_dir() else "file" if path.is_file() else None
        if kind:
            yield kind, path


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise WorkflowError(f"Manifest de arquivo ausente: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if not set(MANIFEST_FIELDS).issubset(row):
            raise WorkflowError(f"Manifest de arquivo inválido: {path}")
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _copy_object(
    source: Path,
    destination: Path,
    digest: str,
    verified_objects: set[str],
) -> bool:
    if destination.is_file():
        if destination.stat().st_size != source.stat().st_size:
            raise WorkflowError(f"Objeto existente com tamanho divergente: {destination}")
        if digest not in verified_objects and sha256(destination) != digest:
            raise WorkflowError(f"Objeto existente com hash divergente: {destination}")
        verified_objects.add(digest)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{digest}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copy2(source, temporary)
        if sha256(temporary) != digest:
            raise WorkflowError(f"Falha de integridade ao copiar {source}.")
        try:
            temporary.replace(destination)
        except FileExistsError:
            if destination.stat().st_size != source.stat().st_size:
                raise WorkflowError(
                    f"Conflito concorrente no objeto {destination}."
                )
        verified_objects.add(digest)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def import_archive_snapshot(
    ws: Workspace,
    snapshot_id: str,
    source_root: str | Path,
    *,
    source_label: str | None = None,
) -> dict[str, Any]:
    """Importa uma árvore sem sobrescrever snapshots existentes."""
    source = Path(source_root).resolve()
    if not source.is_dir():
        raise WorkflowError(f"Diretório de origem não encontrado: {source}")
    if source == ws.archive_root.resolve() or source.is_relative_to(
        ws.archive_root.resolve()
    ):
        raise WorkflowError("O arquivo do Dataset Studio não pode importar a si mesmo.")
    snapshot = _snapshot_root(ws, snapshot_id)
    if snapshot.exists():
        verification = verify_archive_snapshot(
            ws, snapshot_id, source_root=source
        )
        if not verification["valid"]:
            raise WorkflowError(
                f"O snapshot {snapshot_id} existe, mas diverge da origem."
            )
        return load_yaml(snapshot / "snapshot.yaml")

    staging = (
        ws.archive_root
        / "staging"
        / f"{snapshot_id}.{uuid.uuid4().hex}.tmp"
    )
    staging.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    logical_bytes = 0
    copied_bytes = 0
    copied_objects = 0
    reused_objects = 0
    file_count = 0
    directory_count = 0
    verified_objects: set[str] = set()
    try:
        for kind, path in _iter_entries(source):
            relative = path.relative_to(source).as_posix()
            if kind == "directory":
                directory_count += 1
                rows.append(
                    {
                        "kind": kind,
                        "path": relative,
                        "size_bytes": 0,
                        "sha256": "",
                    }
                )
                continue
            digest = sha256(path)
            size = path.stat().st_size
            copied = _copy_object(
                path,
                _object_path(ws, digest),
                digest,
                verified_objects,
            )
            file_count += 1
            logical_bytes += size
            copied_bytes += size if copied else 0
            copied_objects += int(copied)
            reused_objects += int(not copied)
            rows.append(
                {
                    "kind": kind,
                    "path": relative,
                    "size_bytes": size,
                    "sha256": digest,
                }
            )

        manifest_path = staging / "manifest.csv"
        _write_manifest(manifest_path, rows)
        record = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "immutable": True,
            "imported_at": utc_now(),
            "source": {
                "label": source_label or source.name,
                "original_path": str(source),
            },
            "files": file_count,
            "directories": directory_count,
            "logical_bytes": logical_bytes,
            "unique_bytes_added": copied_bytes,
            "objects_added": copied_objects,
            "objects_reused": reused_objects,
            "manifest_sha256": sha256(manifest_path),
        }
        dump_yaml(staging / "snapshot.yaml", record)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(snapshot)
        return record
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_archive_snapshot(
    ws: Workspace,
    snapshot_id: str,
    *,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Confere manifest, objetos e, opcionalmente, a árvore original."""
    snapshot = _snapshot_root(ws, snapshot_id)
    config_path = snapshot / "snapshot.yaml"
    manifest_path = snapshot / "manifest.csv"
    config = load_yaml(config_path)
    rows = _read_manifest(manifest_path)
    errors: list[str] = []
    if config.get("manifest_sha256") != sha256(manifest_path):
        errors.append("SHA-256 do manifest divergente.")

    source = Path(source_root).resolve() if source_root else None
    expected_paths = {row["path"] for row in rows}
    for row in rows:
        relative = row["path"]
        if row["kind"] == "directory":
            if source and not (source / relative).is_dir():
                errors.append(f"Diretório ausente na origem: {relative}")
            continue
        digest = row["sha256"]
        object_path = _object_path(ws, digest)
        if not object_path.is_file():
            errors.append(f"Objeto ausente: {digest}")
        elif object_path.stat().st_size != int(row["size_bytes"]):
            errors.append(f"Tamanho divergente no objeto: {digest}")
        elif sha256(object_path) != digest:
            errors.append(f"Hash divergente no objeto: {digest}")
        if source:
            source_file = source / relative
            if not source_file.is_file():
                errors.append(f"Arquivo ausente na origem: {relative}")
            elif source_file.stat().st_size != int(row["size_bytes"]):
                errors.append(f"Tamanho divergente na origem: {relative}")
            elif sha256(source_file) != digest:
                errors.append(f"Hash divergente na origem: {relative}")

    if source:
        actual_paths = {
            path.relative_to(source).as_posix()
            for _, path in _iter_entries(source)
        }
        for unexpected in sorted(actual_paths - expected_paths):
            errors.append(f"Entrada nova na origem: {unexpected}")
    return {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "verified_at": utc_now(),
        "valid": not errors,
        "files": config.get("files"),
        "logical_bytes": config.get("logical_bytes"),
        "source_checked": source is not None,
        "errors": errors,
    }


def materialize_archive_snapshot(
    ws: Workspace,
    snapshot_id: str,
    destination: str | Path,
) -> dict[str, Any]:
    """Reconstrói uma cópia física exata a partir do arquivo deduplicado."""
    target = Path(destination).resolve()
    if target.exists() and any(target.iterdir()):
        raise WorkflowError(f"O destino deve estar vazio: {target}")
    verification = verify_archive_snapshot(ws, snapshot_id)
    if not verification["valid"]:
        raise WorkflowError(
            f"O snapshot {snapshot_id} falhou na verificação de integridade."
        )
    rows = _read_manifest(_snapshot_root(ws, snapshot_id) / "manifest.csv")
    target.mkdir(parents=True, exist_ok=True)
    for row in rows:
        output = (target / row["path"]).resolve()
        if not output.is_relative_to(target):
            raise WorkflowError(f"Caminho inválido no manifest: {row['path']}")
        if row["kind"] == "directory":
            output.mkdir(parents=True, exist_ok=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_object_path(ws, row["sha256"]), output)
    return {
        "snapshot_id": snapshot_id,
        "destination": str(target),
        "files": verification["files"],
        "logical_bytes": verification["logical_bytes"],
        "materialized_at": utc_now(),
    }


def attach_archive_to_dataset(
    ws: Workspace,
    dataset_id: str,
    snapshot_id: str,
    *,
    subpaths: list[str],
    manifest_subpath: str | None = None,
) -> dict[str, Any]:
    """Relaciona um registro de dataset a conteúdo físico já arquivado."""
    dataset_path = dataset_registry_path(ws, dataset_id)
    dataset = load_yaml(dataset_path)
    snapshot_root = _snapshot_root(ws, snapshot_id)
    snapshot = load_yaml(snapshot_root / "snapshot.yaml")
    rows = _read_manifest(snapshot_root / "manifest.csv")
    files = {row["path"]: row for row in rows if row["kind"] == "file"}
    all_paths = {row["path"] for row in rows}
    normalized_subpaths = [Path(value).as_posix().strip("/") for value in subpaths]
    for prefix in normalized_subpaths:
        if prefix not in all_paths and not any(
            path.startswith(f"{prefix}/") for path in all_paths
        ):
            raise WorkflowError(
                f"O subcaminho {prefix!r} não existe no snapshot {snapshot_id}."
            )

    current_paths = dict(dataset.get("paths") or {})
    archived_paths = {
        "snapshot": (
            snapshot_root / "snapshot.yaml"
        ).relative_to(ws.root).as_posix(),
        "archive_manifest": (
            snapshot_root / "manifest.csv"
        ).relative_to(ws.root).as_posix(),
    }
    if manifest_subpath:
        normalized_manifest = Path(manifest_subpath).as_posix().strip("/")
        row = files.get(normalized_manifest)
        if not row:
            raise WorkflowError(
                f"Manifest original ausente no snapshot: {normalized_manifest}"
            )
        archived_paths["manifest"] = _object_path(
            ws, row["sha256"]
        ).relative_to(ws.root).as_posix()

    evidence = list((dataset.get("provenance") or {}).get("evidence") or [])
    archive_manifest = archived_paths["archive_manifest"]
    if archive_manifest not in evidence:
        evidence.append(archive_manifest)
    updated = {
        **dataset,
        "legacy_paths": current_paths,
        "paths": archived_paths,
        "physical_archive": {
            "snapshot_id": snapshot_id,
            "subpaths": normalized_subpaths,
            "manifest_sha256": snapshot["manifest_sha256"],
            "attached_at": utc_now(),
        },
        "provenance": {
            **(dataset.get("provenance") or {}),
            "evidence": evidence,
        },
    }
    register_dataset(ws, updated, replace=True)
    return updated


def archive_status(ws: Workspace) -> dict[str, Any]:
    snapshots_root = ws.archive_root / "snapshots"
    snapshots = []
    if snapshots_root.is_dir():
        for path in sorted(item for item in snapshots_root.iterdir() if item.is_dir()):
            snapshots.append(load_yaml(path / "snapshot.yaml"))
    object_files = (
        [path for path in (ws.archive_root / "objects").rglob("*") if path.is_file()]
        if (ws.archive_root / "objects").is_dir()
        else []
    )
    return {
        "schema_version": 1,
        "snapshots": snapshots,
        "snapshot_count": len(snapshots),
        "object_count": len(object_files),
        "physical_bytes": sum(path.stat().st_size for path in object_files),
        "logical_bytes": sum(int(item.get("logical_bytes") or 0) for item in snapshots),
    }
