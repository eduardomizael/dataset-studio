"""Registro estruturado de datasets, treinamentos e modelos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_studio.domain.errors import WorkflowError
from dataset_studio.domain.workspace import (
    Workspace,
    dump_yaml,
    load_yaml,
    sha256,
    utc_now,
    validate_id,
)

MODEL_STATES = {
    "external",
    "base",
    "experimental",
    "baseline",
    "candidate",
    "promoted",
    "discarded",
    "legacy",
}
CONFIDENCE_LEVELS = {"confirmed", "probable", "incomplete"}


def models_registry_path(ws: Workspace) -> Path:
    return ws.registry_root / "models.yaml"


def aliases_registry_path(ws: Workspace) -> Path:
    return ws.registry_root / "aliases.yaml"


def dataset_registry_path(ws: Workspace, dataset_id: str) -> Path:
    validate_id(dataset_id, "dataset_id")
    return ws.registry_root / "datasets" / f"{dataset_id}.yaml"


def run_registry_path(ws: Workspace, run_id: str) -> Path:
    validate_id(run_id, "run_id")
    return ws.registry_root / "runs" / f"{run_id}.yaml"


def _load_or_default(path: Path, key: str) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, key: {}}
    payload = load_yaml(path)
    if not isinstance(payload.get(key), dict):
        raise WorkflowError(f"Registry inválido em {path}: campo '{key}'.")
    return payload


def _stored_path(ws: Workspace, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ws.root).as_posix()
    except ValueError:
        return str(resolved)


def resolve_registered_path(ws: Workspace, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ws.resolve_path(path)


def list_registered_models(ws: Workspace) -> dict[str, dict[str, Any]]:
    return _load_or_default(models_registry_path(ws), "models")["models"]


def list_registered_aliases(ws: Workspace) -> dict[str, str]:
    return _load_or_default(aliases_registry_path(ws), "aliases")["aliases"]


def register_dataset(
    ws: Workspace,
    record: dict[str, Any],
    *,
    replace: bool = False,
) -> Path:
    dataset_id = str(record.get("dataset_id") or "")
    validate_id(dataset_id, "dataset_id")
    provenance = record.get("provenance") or {}
    confidence = provenance.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        raise WorkflowError(
            f"Confiança inválida para {dataset_id}: {confidence!r}."
        )
    path = dataset_registry_path(ws, dataset_id)
    if path.exists() and not replace:
        current = load_yaml(path)
        if current != record:
            raise WorkflowError(f"Dataset já registrado com conteúdo diferente: {dataset_id}")
        return path
    dump_yaml(path, record)
    return path


def register_run(
    ws: Workspace,
    record: dict[str, Any],
    *,
    replace: bool = False,
) -> Path:
    run_id = str(record.get("run_id") or "")
    validate_id(run_id, "run_id")
    provenance = record.get("provenance") or {}
    confidence = provenance.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        raise WorkflowError(f"Confiança inválida para {run_id}: {confidence!r}.")
    path = run_registry_path(ws, run_id)
    if path.exists() and not replace:
        current = load_yaml(path)
        if current != record:
            raise WorkflowError(f"Treinamento já registrado com conteúdo diferente: {run_id}")
        return path
    dump_yaml(path, record)
    return path


def register_model(
    ws: Workspace,
    record: dict[str, Any],
    *,
    aliases: list[Path] | None = None,
    replace: bool = False,
) -> Path:
    model_id = str(record.get("model_id") or "")
    validate_id(model_id, "model_id")
    state = record.get("state")
    if state not in MODEL_STATES:
        raise WorkflowError(f"Estado inválido para {model_id}: {state!r}.")

    registry = _load_or_default(models_registry_path(ws), "models")
    current = registry["models"].get(model_id)
    if current and not replace and current != record:
        current_hash = current.get("sha256")
        new_hash = record.get("sha256")
        if current_hash != new_hash:
            raise WorkflowError(
                f"model_id {model_id} já aponta para outro SHA-256."
            )
    registry["models"][model_id] = record
    registry["models"] = dict(sorted(registry["models"].items()))
    dump_yaml(models_registry_path(ws), registry)

    alias_registry = _load_or_default(aliases_registry_path(ws), "aliases")
    for alias in aliases or []:
        alias_key = _stored_path(ws, alias)
        owner = alias_registry["aliases"].get(alias_key)
        if owner and owner != model_id:
            raise WorkflowError(
                f"Alias {alias_key} já pertence ao modelo {owner}."
            )
        alias_registry["aliases"][alias_key] = model_id
    alias_registry["aliases"] = dict(sorted(alias_registry["aliases"].items()))
    dump_yaml(aliases_registry_path(ws), alias_registry)
    return models_registry_path(ws)


def find_model_by_hash(ws: Workspace, digest: str) -> tuple[str, dict[str, Any]] | None:
    for model_id, record in list_registered_models(ws).items():
        if record.get("sha256") == digest:
            return model_id, record
    return None


def find_model_by_path(ws: Workspace, path: Path) -> tuple[str, dict[str, Any]] | None:
    alias = _stored_path(ws, path)
    model_id = list_registered_aliases(ws).get(alias)
    if model_id:
        record = list_registered_models(ws).get(model_id)
        if record:
            return model_id, record
    if path.is_file():
        return find_model_by_hash(ws, sha256(path))
    return None


def validate_registry(ws: Workspace) -> dict[str, Any]:
    """Valida referências, aliases e hashes sem alterar o workspace."""
    errors: list[str] = []
    warnings: list[str] = []
    models = list_registered_models(ws)
    aliases = list_registered_aliases(ws)

    for model_id, model in models.items():
        parent = model.get("parent_model_id")
        if parent and parent not in models:
            errors.append(f"{model_id}: parent_model_id inexistente: {parent}")
        expected_hash = model.get("sha256")
        paths = model.get("paths") or []
        existing_paths = 0
        for value in paths:
            path = resolve_registered_path(ws, str(value))
            if not path.is_file():
                warnings.append(f"{model_id}: arquivo ausente: {value}")
                continue
            existing_paths += 1
            if expected_hash and sha256(path) != expected_hash:
                errors.append(f"{model_id}: SHA-256 divergente em {value}")
        if paths and not existing_paths:
            warnings.append(f"{model_id}: nenhum arquivo físico disponível")

    for alias, model_id in aliases.items():
        if model_id not in models:
            errors.append(f"Alias {alias}: model_id inexistente: {model_id}")
            continue
        path = resolve_registered_path(ws, alias)
        if not path.exists():
            warnings.append(f"Alias ausente: {alias}")
        elif path.is_file():
            expected_hash = models[model_id].get("sha256")
            if expected_hash and sha256(path) != expected_hash:
                errors.append(f"Alias {alias}: SHA-256 divergente")

    run_paths = sorted((ws.registry_root / "runs").glob("*.yaml"))
    dataset_paths = sorted((ws.registry_root / "datasets").glob("*.yaml"))
    dataset_ids = {path.stem for path in dataset_paths}
    for path in dataset_paths:
        dataset = load_yaml(path)
        manifest_value = (dataset.get("paths") or {}).get("manifest")
        manifest_hash = dataset.get("manifest_sha256")
        if manifest_value and manifest_hash:
            manifest_path = resolve_registered_path(ws, str(manifest_value))
            if not manifest_path.is_file():
                warnings.append(
                    f"{dataset.get('dataset_id', path.stem)}: manifest ausente"
                )
            elif sha256(manifest_path) != manifest_hash:
                errors.append(
                    f"{dataset.get('dataset_id', path.stem)}: manifest alterado"
                )
    for path in run_paths:
        run = load_yaml(path)
        run_id = run.get("run_id", path.stem)
        dataset_id = run.get("dataset_id")
        if dataset_id and dataset_id not in dataset_ids:
            errors.append(f"{run_id}: dataset_id inexistente: {dataset_id}")
        initial_model_id = run.get("initial_model_id")
        if initial_model_id and initial_model_id not in models:
            errors.append(
                f"{run_id}: initial_model_id inexistente: {initial_model_id}"
            )
        output_model_id = run.get("output_model_id")
        if output_model_id and output_model_id not in models:
            errors.append(
                f"{run_id}: output_model_id inexistente: {output_model_id}"
            )
        for artifact_name, artifact in (run.get("artifacts") or {}).items():
            artifact_path = resolve_registered_path(ws, str(artifact.get("path") or ""))
            expected_hash = artifact.get("sha256")
            if not artifact_path.is_file():
                warnings.append(f"{run_id}: artefato ausente: {artifact_name}")
            elif expected_hash and sha256(artifact_path) != expected_hash:
                errors.append(
                    f"{run_id}: artefato alterado: {artifact_name}"
                )

    run_ids = {path.stem for path in run_paths}
    for model_id, model in models.items():
        source_run_id = model.get("source_run_id")
        if source_run_id and source_run_id not in run_ids:
            warnings.append(
                f"{model_id}: source_run_id não registrado: {source_run_id}"
            )

    return {
        "schema_version": 1,
        "validated_at": utc_now(),
        "valid": not errors,
        "models": len(models),
        "aliases": len(aliases),
        "datasets": len(dataset_ids),
        "runs": len(run_paths),
        "errors": errors,
        "warnings": warnings,
    }
