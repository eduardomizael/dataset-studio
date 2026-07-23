"""Exportação imutável de modelos para aplicações consumidoras."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from dataset_studio.domain import (
    Workspace,
    WorkflowError,
    dataset_registry_path,
    dump_yaml,
    list_registered_models,
    load_yaml,
    run_registry_path,
    sha256,
    utc_now,
    validate_id,
)


def _stored_path(ws: Workspace, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ws.root).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_artifact(
    ws: Workspace,
    model: dict[str, Any],
    artifact_path: str | Path | None,
) -> Path:
    candidates = [artifact_path] if artifact_path else model.get("paths") or []
    for value in candidates:
        path = ws.resolve_path(str(value))
        if path.is_file() or path.is_dir():
            return path
    raise WorkflowError(
        f"Nenhum artefato físico disponível para o modelo {model['model_id']}."
    )


def _directory_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files[path.relative_to(root).as_posix()] = sha256(path)
    if not files:
        raise WorkflowError(f"O diretório de modelo está vazio: {root}")
    return files


def _dataset_context(ws: Workspace, dataset_id: str | None) -> dict[str, Any]:
    if not dataset_id:
        return {}
    path = dataset_registry_path(ws, dataset_id)
    if not path.is_file():
        return {"dataset_id": dataset_id}
    dataset = load_yaml(path)
    context: dict[str, Any] = {
        "dataset_id": dataset_id,
        "images": dataset.get("images"),
        "boxes": dataset.get("boxes"),
        "splits": dataset.get("splits"),
        "manifest_sha256": dataset.get("manifest_sha256"),
        "provisional": dataset.get("provisional"),
    }
    data_value = (dataset.get("paths") or {}).get("data")
    if data_value:
        data_path = ws.resolve_path(data_value)
        if data_path.is_file():
            context["classes"] = load_yaml(data_path).get("names", {})
    return {key: value for key, value in context.items() if value is not None}


def _run_context(ws: Workspace, run_id: str | None) -> dict[str, Any]:
    if not run_id:
        return {}
    path = run_registry_path(ws, run_id)
    if not path.is_file():
        return {"run_id": run_id}
    run = load_yaml(path)
    training = run.get("training") or {}
    return {
        "run_id": run_id,
        "dataset_id": run.get("dataset_id"),
        "status": run.get("status"),
        "metrics": run.get("metrics") or {},
        "training": {
            key: training.get(key)
            for key in (
                "imgsz",
                "epochs",
                "batch",
                "device",
                "seed",
                "optimizer",
            )
            if training.get(key) is not None
        },
    }


def validate_deployment_bundle(
    ws: Workspace,
    deployment_id: str,
) -> dict[str, Any]:
    """Valida hashes e confinamento de um bundle já materializado."""
    validate_id(deployment_id, "deployment_id")
    root = (ws.deployments_root / deployment_id).resolve()
    manifest_path = root / "deployment_manifest.yaml"
    if not manifest_path.is_file():
        raise WorkflowError(f"Manifest de implantação ausente: {manifest_path}")
    manifest = load_yaml(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("immutable") is not True:
        raise WorkflowError(f"Manifest de implantação inválido: {manifest_path}")
    artifact = manifest.get("artifact") or {}
    relative = artifact.get("path")
    if not isinstance(relative, str) or not relative:
        raise WorkflowError("Manifest sem artifact.path.")
    artifact_path = (root / relative).resolve()
    if not artifact_path.is_relative_to(root):
        raise WorkflowError("artifact.path tenta sair do bundle.")

    if artifact.get("kind") == "file":
        expected = artifact.get("sha256")
        if not artifact_path.is_file() or not isinstance(expected, str):
            raise WorkflowError("Arquivo ou SHA-256 ausente no bundle.")
        if sha256(artifact_path) != expected:
            raise WorkflowError(f"SHA-256 divergente no deployment {deployment_id}.")
    elif artifact.get("kind") == "directory":
        expected_files = artifact.get("files")
        if not artifact_path.is_dir() or not isinstance(expected_files, dict):
            raise WorkflowError("Diretório ou lista de hashes ausente no bundle.")
        actual_files = _directory_files(artifact_path)
        if actual_files != expected_files:
            raise WorkflowError(
                f"Conteúdo divergente no deployment {deployment_id}."
            )
    else:
        raise WorkflowError("Tipo de artefato inválido no bundle.")
    return manifest


def export_deployment_bundle(
    ws: Workspace,
    model_id: str,
    *,
    deployment_id: str | None = None,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Materializa um bundle imutável e autocontido para inferência."""
    models = list_registered_models(ws)
    model = models.get(model_id)
    if not model:
        raise WorkflowError(f"Modelo não registrado: {model_id}")

    target_id = deployment_id or model_id
    validate_id(target_id, "deployment_id")
    artifact = _resolve_artifact(ws, model, artifact_path)
    expected_hash = model.get("sha256")
    if artifact.is_file() and expected_hash and sha256(artifact) != expected_hash:
        raise WorkflowError(
            f"O artefato de {model_id} não corresponde ao SHA-256 do registry."
        )

    destination = ws.deployments_root / target_id
    if destination.exists():
        current = validate_deployment_bundle(ws, target_id)
        if current.get("model", {}).get("model_id") != model_id:
            raise WorkflowError(
                f"O deployment_id {target_id} já pertence a outro modelo."
            )
        return current

    temporary = ws.deployments_root / f".{target_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        if artifact.is_file():
            artifact_name = f"model{artifact.suffix.lower()}"
            copied_artifact = temporary / artifact_name
            shutil.copy2(artifact, copied_artifact)
            artifact_record: dict[str, Any] = {
                "kind": "file",
                "path": artifact_name,
                "sha256": sha256(copied_artifact),
                "size_bytes": copied_artifact.stat().st_size,
            }
        else:
            artifact_name = "model_bundle"
            copied_artifact = temporary / artifact_name
            shutil.copytree(artifact, copied_artifact)
            files = _directory_files(copied_artifact)
            artifact_record = {
                "kind": "directory",
                "path": artifact_name,
                "files": files,
                "size_bytes": sum(
                    path.stat().st_size
                    for path in copied_artifact.rglob("*")
                    if path.is_file()
                ),
            }

        warnings: list[str] = []
        if model.get("state") not in {"promoted", "baseline"}:
            warnings.append(
                "O modelo foi exportado sem estado promoted ou baseline; "
                "a decisão de implantação é responsabilidade do usuário."
            )
        provenance = model.get("provenance") or {}
        if provenance.get("confidence") != "confirmed":
            warnings.append(
                "A proveniência do modelo não está classificada como confirmed."
            )

        run_context = _run_context(ws, model.get("source_run_id"))
        dataset_id = model.get("dataset_id") or run_context.get("dataset_id")
        manifest = {
            "schema_version": 1,
            "deployment_id": target_id,
            "created_at": utc_now(),
            "immutable": True,
            "model": {
                "model_id": model_id,
                "architecture": model.get("architecture"),
                "role": model.get("role"),
                "state": model.get("state"),
                "sha256": model.get("sha256"),
                "parent_model_id": model.get("parent_model_id"),
                "source_run_id": model.get("source_run_id"),
                "dataset_id": dataset_id,
                "source_artifact": _stored_path(ws, artifact),
            },
            "artifact": artifact_record,
            "dataset": _dataset_context(ws, dataset_id),
            "run": run_context,
            "warnings": warnings,
        }
        dump_yaml(temporary / "deployment_manifest.yaml", manifest)
        ws.deployments_root.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
