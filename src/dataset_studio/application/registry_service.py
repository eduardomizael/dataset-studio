"""Serviços de proveniência para versões, treinamentos e modelos."""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_studio.domain import (
    Workspace,
    WorkflowError,
    dump_yaml,
    find_model_by_hash,
    find_model_by_path,
    list_registered_models,
    load_yaml,
    register_dataset,
    register_model,
    register_run,
    run_registry_path,
    sha256,
    utc_now,
    validate_registry,
    version_config_path,
    version_root,
)
from dataset_studio.ports.trainer import TrainingParams


def _stored_path(ws: Workspace, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ws.root).as_posix()
    except ValueError:
        return str(resolved)


def _safe_id(value: str, prefix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return f"{prefix}-{normalized}" if normalized else prefix


def _resolve_model_path(ws: Workspace, value: str) -> Path:
    direct = ws.resolve_path(value)
    if direct.is_file():
        return direct
    candidate = ws.models_root / value
    return candidate.resolve()


def resolve_model_reference(ws: Workspace, value: str) -> str:
    """Aceita caminho legado ou model_id e retorna um caminho utilizável."""
    model = list_registered_models(ws).get(value)
    if not model:
        return value
    for stored in model.get("paths") or []:
        path = Path(str(stored))
        resolved = path if path.is_absolute() else ws.resolve_path(path)
        if resolved.is_file() and resolved.suffix.lower() == ".pt":
            try:
                return resolved.relative_to(ws.root).as_posix()
            except ValueError:
                return str(resolved)
    raise WorkflowError(
        f"O modelo registrado {value} não possui um arquivo .pt disponível."
    )


def _ensure_initial_model(
    ws: Workspace, model_value: str
) -> tuple[str, dict[str, Any]]:
    model_path = _resolve_model_path(ws, model_value)
    existing = find_model_by_path(ws, model_path)
    if existing:
        model_id, record = existing
        paths = sorted(
            set([*(record.get("paths") or []), _stored_path(ws, model_path)])
        )
        updated = {**record, "paths": paths}
        register_model(ws, updated, aliases=[model_path], replace=True)
        return model_id, updated

    if model_path.is_file():
        digest = sha256(model_path)
        by_hash = find_model_by_hash(ws, digest)
        if by_hash:
            model_id, record = by_hash
            paths = sorted(
                set([*(record.get("paths") or []), _stored_path(ws, model_path)])
            )
            updated = {**record, "paths": paths}
            register_model(ws, updated, aliases=[model_path], replace=True)
            return model_id, updated
        model_id = f"model-sha256-{digest[:16]}"
        record = {
            "model_id": model_id,
            "architecture": model_path.stem,
            "role": "initial_model",
            "state": "base",
            "sha256": digest,
            "paths": [_stored_path(ws, model_path)],
            "parent_model_id": None,
            "source_run_id": None,
            "created_at": utc_now(),
            "provenance": {
                "origin": "generated",
                "confidence": "confirmed",
                "evidence": [_stored_path(ws, model_path)],
            },
        }
        register_model(ws, record, aliases=[model_path])
        return model_id, record

    model_id = _safe_id(model_value, "model-external")
    record = {
        "model_id": model_id,
        "architecture": Path(model_value).stem,
        "role": "initial_model",
        "state": "external",
        "sha256": None,
        "paths": [model_value],
        "parent_model_id": None,
        "source_run_id": None,
        "created_at": utc_now(),
        "provenance": {
            "origin": "generated",
            "confidence": "incomplete",
            "evidence": [],
            "notes": ["O modelo não estava disponível localmente no início do registro."],
        },
    }
    current = list_registered_models(ws).get(model_id)
    register_model(ws, current or record, replace=bool(current))
    return model_id, current or record


def snapshot_version_dataset(ws: Workspace, version_id: str) -> dict[str, Any]:
    """Registra a identidade de uma versão materializada como dataset."""
    root = version_root(ws, version_id)
    config_path = version_config_path(ws, version_id)
    build_report_path = root / "build_report.json"
    manifest_path = root / "manifest.csv"
    data_path = root / "data.yaml"
    config = load_yaml(config_path)
    report = (
        json.loads(build_report_path.read_text(encoding="utf-8"))
        if build_report_path.is_file()
        else {}
    )
    evidence = [
        _stored_path(ws, path)
        for path in (config_path, build_report_path, manifest_path, data_path)
        if path.is_file()
    ]
    record = {
        "schema_version": 1,
        "dataset_id": version_id,
        "kind": "materialized_version",
        "version_id": version_id,
        "sources": list(config.get("sources") or config.get("campaigns") or []),
        "annotation_revisions": config.get("annotation_revisions", {}),
        "provisional": bool(config.get("provisional", False)),
        "evaluation_level": config.get("evaluation_level", "legacy"),
        "quality_assessment": config.get("quality_assessment", {}),
        "images": report.get("images"),
        "boxes": report.get("boxes"),
        "excluded_frames": report.get("excluded_frames"),
        "splits": report.get("splits", {}),
        "manifest_sha256": (
            report.get("manifest_sha256")
            or (sha256(manifest_path) if manifest_path.is_file() else None)
        ),
        "version_config_sha256": (
            report.get("version_config_sha256")
            or (sha256(config_path) if config_path.is_file() else None)
        ),
        "paths": {
            "version": _stored_path(ws, config_path),
            "manifest": _stored_path(ws, manifest_path),
            "data": _stored_path(ws, data_path),
            "build_report": _stored_path(ws, build_report_path),
        },
        "canonical_manifest": {
            "path": _stored_path(ws, config_path),
            "sha256": (
                report.get("version_config_sha256")
                or (sha256(config_path) if config_path.is_file() else None)
            ),
        },
        "provenance": {
            "origin": "generated",
            "confidence": "confirmed",
            "reconstructed_at": None,
            "evidence": evidence,
        },
    }
    register_dataset(ws, record, replace=True)
    return record


def _copy_training_evidence(ws: Workspace, training_id: str, version_id: str) -> list[str]:
    run_root = ws.runs_root / training_id
    evidence_root = run_root / "provenance"
    evidence_root.mkdir(parents=True, exist_ok=True)
    version_dir = version_root(ws, version_id)
    copied: list[str] = []
    for name in ("version.yaml", "data.yaml", "manifest.csv", "build_report.json"):
        source = version_dir / name
        if not source.is_file():
            continue
        destination = evidence_root / name
        shutil.copy2(source, destination)
        copied.append(_stored_path(ws, destination))
    return copied


def begin_training_record(
    ws: Workspace,
    training_id: str,
    version_id: str,
    params: TrainingParams,
) -> dict[str, Any]:
    """Cria o snapshot de proveniência antes de iniciar o subprocesso."""
    dataset = snapshot_version_dataset(ws, version_id)
    initial_model_id, initial_model = _ensure_initial_model(ws, params.model)
    evidence = _copy_training_evidence(ws, training_id, version_id)
    record = {
        "schema_version": 1,
        "run_id": training_id,
        "status": "queued",
        "state": "experimental",
        "created_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "dataset_id": dataset["dataset_id"],
        "initial_model_id": initial_model_id,
        "initial_model_sha256": initial_model.get("sha256"),
        "output_model_id": None,
        "training": params.to_dict(),
        "metrics": {},
        "artifacts": {},
        "provenance": {
            "origin": "generated",
            "confidence": "confirmed",
            "reconstructed_at": None,
            "evidence": evidence,
        },
    }
    register_run(ws, record, replace=True)
    return record


def _read_metrics(csv_path: Path) -> dict[str, Any]:
    if not csv_path.is_file():
        return {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def metric_value(row: dict[str, str], token: str) -> float | None:
        for key, raw in row.items():
            if token in key.strip().lower():
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return None
        return None

    scored = [
        (metric_value(row, "map50-95"), index, row)
        for index, row in enumerate(rows)
    ]
    scored = [item for item in scored if item[0] is not None]
    _, best_index, best_row = max(scored, default=(None, len(rows) - 1, rows[-1]))
    final_row = rows[-1]
    return {
        "completed_epochs": len(rows),
        "best_epoch": int(float(best_row.get("epoch", best_index))),
        "best": {
            "precision": metric_value(best_row, "precision"),
            "recall": metric_value(best_row, "recall"),
            "map50": next(
                (
                    metric_value(best_row, token)
                    for token in ("map50(b)", "map50")
                    if metric_value(best_row, token) is not None
                ),
                None,
            ),
            "map50_95": metric_value(best_row, "map50-95"),
        },
        "final": {
            "precision": metric_value(final_row, "precision"),
            "recall": metric_value(final_row, "recall"),
            "map50": next(
                (
                    metric_value(final_row, token)
                    for token in ("map50(b)", "map50")
                    if metric_value(final_row, token) is not None
                ),
                None,
            ),
            "map50_95": metric_value(final_row, "map50-95"),
        },
    }


def finalize_training_record(
    ws: Workspace,
    training_id: str,
    status: str,
    *,
    state: str | None = None,
) -> dict[str, Any]:
    """Consolida métricas, hashes e checkpoint resultante após o treinamento."""
    registry_path = run_registry_path(ws, training_id)
    if registry_path.is_file():
        record = load_yaml(registry_path)
    else:
        record = {
            "schema_version": 1,
            "run_id": training_id,
            "dataset_id": None,
            "initial_model_id": None,
            "provenance": {
                "origin": "generated",
                "confidence": "incomplete",
                "evidence": [],
            },
        }
    run_root = ws.runs_root / training_id
    best = run_root / "weights" / "best.pt"
    last = run_root / "weights" / "last.pt"
    args = run_root / "args.yaml"
    results = run_root / "results.csv"
    evaluation_summary = run_root / "evaluations" / "summary.json"
    artifacts: dict[str, Any] = dict(record.get("artifacts") or {})
    for key, path in (
        ("args", args),
        ("results", results),
        ("best", best),
        ("last", last),
        ("evaluation_summary", evaluation_summary),
    ):
        if path.is_file():
            artifacts[key] = {
                "path": _stored_path(ws, path),
                "sha256": sha256(path),
            }

    output_model_id = record.get("output_model_id")
    if best.is_file():
        digest = sha256(best)
        existing = find_model_by_hash(ws, digest)
        output_model_id = existing[0] if existing else f"model-{training_id}-best"
        output_record = (
            existing[1]
            if existing
            else {
                "model_id": output_model_id,
                "architecture": Path(
                    str((record.get("training") or {}).get("model") or "YOLO")
                ).stem,
                "role": "trained_checkpoint",
                "state": state or "experimental",
                "sha256": digest,
                "paths": [_stored_path(ws, best)],
                "parent_model_id": record.get("initial_model_id"),
                "source_run_id": training_id,
                "dataset_id": record.get("dataset_id"),
                "created_at": datetime.fromtimestamp(
                    best.stat().st_mtime, timezone.utc
                ).isoformat(),
                "provenance": {
                    "origin": record.get("provenance", {}).get(
                        "origin", "generated"
                    ),
                    "confidence": record.get("provenance", {}).get(
                        "confidence", "confirmed"
                    ),
                    "evidence": [
                        value["path"] for value in artifacts.values()
                    ],
                },
            }
        )
        register_model(ws, output_record, aliases=[best], replace=bool(existing))

    evaluations: dict[str, Any] = dict(record.get("evaluations") or {})
    robustness: dict[str, Any] = dict(record.get("robustness") or {})
    if evaluation_summary.is_file():
        try:
            evaluation_payload = json.loads(
                evaluation_summary.read_text(encoding="utf-8")
            )
            evaluations = dict(evaluation_payload.get("evaluations") or {})
            robustness = dict(evaluation_payload.get("robustness") or {})
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            evaluations = {
                **evaluations,
                "_summary": {
                    "status": "failed",
                    "error": f"Relatório de avaliação inválido: {exc}",
                },
            }

    record.update(
        {
            "status": status,
            "state": state or record.get("state", "experimental"),
            "completed_at": (
                record.get("completed_at")
                if record.get("provenance", {}).get("origin") == "reconstructed"
                and record.get("completed_at")
                else utc_now()
            )
            if status in {"completed", "failed", "stopped", "cancelled"}
            else None,
            "output_model_id": output_model_id,
            "metrics": _read_metrics(results) or record.get("metrics", {}),
            "evaluations": evaluations,
            "robustness": robustness,
            "artifacts": artifacts,
        }
    )
    register_run(ws, record, replace=True)
    run_root.mkdir(parents=True, exist_ok=True)
    dump_yaml(run_root / "run.yaml", record)
    return record


def promote_registered_model(
    ws: Workspace,
    training_id: str,
    promoted_path: Path,
) -> dict[str, Any]:
    """Registra a promoção como novo alias do checkpoint, sem mudar sua identidade."""
    record = finalize_training_record(ws, training_id, "completed")
    model_id = record.get("output_model_id")
    if not model_id:
        raise WorkflowError(f"Treinamento {training_id} não produziu best.pt.")
    model = list_registered_models(ws)[model_id]
    paths = sorted(set([*(model.get("paths") or []), _stored_path(ws, promoted_path)]))
    promoted = {**model, "paths": paths, "state": "promoted"}
    register_model(ws, promoted, aliases=[promoted_path], replace=True)
    record["state"] = "promoted"
    register_run(ws, record, replace=True)
    dump_yaml(ws.runs_root / training_id / "run.yaml", record)
    return promoted


def registry_status(ws: Workspace) -> dict[str, Any]:
    return validate_registry(ws)
