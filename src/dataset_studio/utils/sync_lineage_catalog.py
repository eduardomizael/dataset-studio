"""Sincroniza o catálogo derivado com os manifestos canônicos do workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_studio.application.registry_service import snapshot_version_dataset
from dataset_studio.domain import (
    Workspace,
    WorkflowError,
    dump_yaml,
    list_sources,
    list_versions,
    load_yaml,
    prediction_profile_from_config,
    prediction_profile_sha256,
    register_source_manifest,
    sha256,
)


def freeze_legacy_prediction_profile(
    ws: Workspace, source_id: str
) -> bool:
    """Congela o perfil de uma origem antiga sem mudar sua semântica."""
    path = ws.source_config_path(source_id)
    source = load_yaml(path)
    annotation = source.get("annotation") or {}
    if annotation.get("prediction_profile") is not None:
        return False
    config_value = annotation.get("detection_config")
    if annotation.get("backend") != "local" or not config_value:
        return False
    config_path = ws.resolve_path(str(config_value))
    expected = annotation.get("detection_config_sha256")
    if expected and sha256(config_path) != expected:
        raise WorkflowError(
            f"{source_id}: o perfil global mudou desde a criação da origem; "
            "não é seguro congelá-lo automaticamente."
        )
    profile = prediction_profile_from_config(load_yaml(config_path))
    annotation["prediction_profile"] = profile
    annotation["prediction_profile_sha256"] = prediction_profile_sha256(profile)
    source["annotation"] = annotation
    dump_yaml(path, source)
    return True


def sync_lineage_catalog(ws: Workspace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "sources": [],
        "profiles_frozen": [],
        "versions": [],
        "skipped_versions": [],
    }
    for source_id in list_sources(ws):
        if freeze_legacy_prediction_profile(ws, source_id):
            report["profiles_frozen"].append(source_id)
        register_source_manifest(ws, source_id)
        report["sources"].append(source_id)

    for version_id in list_versions(ws):
        root = ws.version_root(version_id)
        required = (
            root / "version.yaml",
            root / "manifest.csv",
            root / "data.yaml",
            root / "build_report.json",
        )
        if not all(path.is_file() for path in required):
            report["skipped_versions"].append(version_id)
            continue
        snapshot_version_dataset(ws, version_id)
        report["versions"].append(version_id)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = sync_lineage_catalog(Workspace.from_path(args.workspace))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
