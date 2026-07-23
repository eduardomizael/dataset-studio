"""Relaciona runs históricos ao snapshot físico do fish_detection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dataset_studio.application import (
    attach_archive_to_run,
    verify_archive_snapshot,
)
from dataset_studio.domain import (
    Workspace,
    load_yaml,
    utc_now,
    validate_registry,
)


def link_fish_runs_archive(
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

    snapshot_manifest = (
        ws.archive_root / "snapshots" / snapshot_id / "manifest.csv"
    )
    with snapshot_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        archived_paths = {
            row["path"]
            for row in csv.DictReader(handle)
            if row.get("path")
        }
    linked = {}
    unmatched = []
    for run_path in sorted((ws.registry_root / "runs").glob("*.yaml")):
        run = load_yaml(run_path)
        run_id = str(run.get("run_id") or run_path.stem)
        legacy_name = str(run.get("legacy_name") or run_id)
        subpath = f"detect/{legacy_name}"
        if subpath not in archived_paths and not any(
            value.startswith(f"{subpath}/") for value in archived_paths
        ):
            unmatched.append(run_id)
            continue
        linked[run_id] = attach_archive_to_run(
            ws,
            run_id,
            snapshot_id,
            subpaths=[subpath],
        )["physical_archive"]

    report = {
        "schema_version": 1,
        "linked_at": utc_now(),
        "snapshot_id": snapshot_id,
        "snapshot_verification": verification,
        "runs": linked,
        "unmatched_runs": unmatched,
        "registry_validation": validate_registry(ws),
    }
    report_path = ws.registry_root / "fish_runs_archive_report.json"
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
    report = link_fish_runs_archive(
        Workspace.from_path(args.workspace),
        args.snapshot_id,
        source_root=args.source,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["registry_validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
