"""Relaciona os registros históricos ao snapshot físico do fish_detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_studio.application import (
    attach_archive_to_dataset,
    verify_archive_snapshot,
)
from dataset_studio.domain import Workspace, utc_now, validate_registry

FISH_DATASET_LINKS = {
    "ds-d02-605": {
        "subpaths": ["split"],
    },
    "ds-d03-variable": {
        "subpaths": ["03_prepared/yolo", "splits"],
    },
    "ds-d03-fixed-test": {
        "subpaths": ["03_prepared/yolo", "05_test_fixed"],
    },
    "ds-d03-fixed-valtest": {
        "subpaths": [
            "03_prepared/yolo",
            "04_splits",
            "05_val_fixed",
            "06_test_fixed",
        ],
    },
    "ds-d04-r260719001": {
        "subpaths": [
            "campaigns/canaleta_pvc_260717",
            "releases/dataset_260719001",
        ],
        "manifest_subpath": "releases/dataset_260719001/manifest.csv",
    },
}


def link_fish_dataset_archive(
    ws: Workspace,
    snapshot_id: str,
    *,
    source_root: Path | None = None,
) -> dict:
    verification = verify_archive_snapshot(
        ws,
        snapshot_id,
        source_root=source_root,
    )
    if not verification["valid"]:
        raise ValueError(f"Snapshot inválido: {verification['errors']}")
    linked = {}
    for dataset_id, options in FISH_DATASET_LINKS.items():
        linked[dataset_id] = attach_archive_to_dataset(
            ws,
            dataset_id,
            snapshot_id,
            subpaths=options["subpaths"],
            manifest_subpath=options.get("manifest_subpath"),
        )["physical_archive"]
    report = {
        "schema_version": 1,
        "linked_at": utc_now(),
        "snapshot_id": snapshot_id,
        "snapshot_verification": verification,
        "datasets": linked,
        "registry_validation": validate_registry(ws),
    }
    report_path = ws.registry_root / "fish_dataset_archive_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args(argv)
    report = link_fish_dataset_archive(
        Workspace.from_path(args.workspace),
        args.snapshot_id,
        source_root=args.source,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["registry_validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
