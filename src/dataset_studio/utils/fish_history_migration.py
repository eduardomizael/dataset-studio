"""Migração aditiva do histórico de treinamento do projeto fish_detection."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_studio.application.registry_service import finalize_training_record
from dataset_studio.domain import (
    Workspace,
    WorkflowError,
    dump_yaml,
    load_yaml,
    list_registered_models,
    register_dataset,
    register_model,
    register_run,
    sha256,
    utc_now,
    validate_registry,
)


def _iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _copy_without_overwrite(source: Path, destination: Path) -> str:
    if destination.exists():
        if source.is_file() and destination.is_file():
            if sha256(source) != sha256(destination):
                raise WorkflowError(
                    f"Conflito: {destination} já existe com conteúdo diferente."
                )
            return "identical"
        if source.is_dir() and destination.is_dir():
            return "existing"
        raise WorkflowError(f"Conflito de tipo ao copiar {source} para {destination}.")
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return "copied"


def _split_variants(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    variants: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            seed = str(row.get("seed") or "unknown")
            split = str(row.get("split") or "unknown")
            variants.setdefault(seed, {})[split] = {
                "images": int(row.get("images") or 0),
                "boxes": int(row.get("boxes") or 0),
                "negatives": int(row.get("none") or 0),
            }
    return variants


def _dataset_records(source_root: Path) -> list[dict[str, Any]]:
    history = source_root / "docs" / "historico_treinamentos_modelos.md"
    experiments = source_root / "runs" / "experiments"
    d3a_summary = experiments / "20260611_175142" / "split_summary.csv"
    d3b_summary = experiments / "20260612_163350" / "split_summary.csv"
    d3c_summary = experiments / "20260612_201316" / "split_summary.csv"
    d4_release = source_root / "dataset" / "releases" / "dataset_260719001"
    common = {
        "schema_version": 1,
        "kind": "legacy_reconstruction",
    }
    return [
        {
            **common,
            "dataset_id": "ds-d00-train6-external",
            "generation": "D0",
            "images": None,
            "boxes": None,
            "negatives": None,
            "splits": {"train": 103, "val": 25},
            "provenance": {
                "origin": "reconstructed",
                "confidence": "incomplete",
                "reconstructed_at": utc_now(),
                "evidence": [str(history), str(source_root / "runs" / "detect" / "train6" / "args.yaml")],
                "notes": ["O conteúdo original de D:\\Peixes não foi preservado."],
            },
        },
        {
            **common,
            "dataset_id": "ds-d01-128",
            "generation": "D1",
            "images": 128,
            "boxes": 222,
            "negatives": 0,
            "splits": {"train": 103, "val": 25},
            "provenance": {
                "origin": "reconstructed",
                "confidence": "probable",
                "reconstructed_at": utc_now(),
                "evidence": [str(history), str(source_root / "runs" / "detect" / "yolo26n_baseline" / "args.yaml")],
            },
        },
        {
            **common,
            "dataset_id": "ds-d02-605",
            "generation": "D2",
            "images": 605,
            "boxes": 901,
            "negatives": 228,
            "splits": {"train": 484, "val": 121},
            "paths": {"snapshot": str(source_root / "dataset" / "split")},
            "provenance": {
                "origin": "reconstructed",
                "confidence": "probable",
                "reconstructed_at": utc_now(),
                "evidence": [str(history), str(source_root / "runs" / "detect" / "yolo26n_v2" / "args.yaml")],
                "notes": ["O snapshot atual converge com os artefatos, mas não há manifest imutável ligado diretamente ao run."],
            },
        },
        {
            **common,
            "dataset_id": "ds-d03-variable",
            "generation": "D3a",
            "images": 1599,
            "boxes": 3323,
            "negatives": 473,
            "split_variants": _split_variants(d3a_summary),
            "paths": {"prepared_snapshot": str(source_root / "dataset" / "03_prepared" / "yolo")},
            "provenance": {
                "origin": "reconstructed",
                "confidence": "confirmed",
                "reconstructed_at": utc_now(),
                "evidence": [str(history), str(d3a_summary), str(experiments / "20260611_175142" / "prepared_summary.json")],
            },
        },
        {
            **common,
            "dataset_id": "ds-d03-fixed-test",
            "generation": "D3b",
            "images": 1599,
            "boxes": 3323,
            "negatives": 473,
            "split_variants": _split_variants(d3b_summary),
            "provenance": {
                "origin": "reconstructed",
                "confidence": "confirmed",
                "reconstructed_at": utc_now(),
                "evidence": [str(history), str(d3b_summary), str(experiments / "20260612_163350" / "prepared_summary.json")],
            },
        },
        {
            **common,
            "dataset_id": "ds-d03-fixed-valtest",
            "generation": "D3c",
            "images": 1599,
            "boxes": 3323,
            "negatives": 473,
            "splits": {
                "train": {"images": 1040, "boxes": 2225, "negatives": 307},
                "val": {"images": 239, "boxes": 469, "negatives": 71},
                "test": {"images": 320, "boxes": 629, "negatives": 95},
            },
            "split_variants": _split_variants(d3c_summary),
            "provenance": {
                "origin": "reconstructed",
                "confidence": "confirmed",
                "reconstructed_at": utc_now(),
                "evidence": [str(history), str(d3c_summary), str(experiments / "20260612_201316" / "prepared_summary.json")],
            },
        },
        {
            **common,
            "dataset_id": "ds-d04-r260719001",
            "generation": "D4",
            "version_id": "dataset_260719001",
            "annotation_revisions": {"canaleta_pvc_260717": "r001_parcial"},
            "provisional": True,
            "images": 581,
            "boxes": 2374,
            "negatives": 189,
            "excluded_frames": 296,
            "splits": {
                "train": {"images": 373, "boxes": 1579},
                "val": {"images": 57, "boxes": 138},
                "test_normal": {"images": 98, "boxes": 155},
                "test_stress": {"images": 53, "boxes": 502},
            },
            "paths": {
                "release": str(d4_release),
                "manifest": str(d4_release / "manifest.csv"),
                "build_report": str(d4_release / "build_report.json"),
            },
            "manifest_sha256": sha256(d4_release / "manifest.csv") if (d4_release / "manifest.csv").is_file() else None,
            "provenance": {
                "origin": "reconstructed",
                "confidence": "confirmed",
                "reconstructed_at": utc_now(),
                "evidence": [
                    str(history),
                    str(d4_release / "release.yaml"),
                    str(d4_release / "manifest.csv"),
                    str(d4_release / "build_report.json"),
                ],
            },
        },
    ]


def _dataset_for_run(run_name: str) -> tuple[str, str]:
    if run_name == "train6":
        return "ds-d00-train6-external", "incomplete"
    if run_name == "yolo26n_baseline":
        return "ds-d01-128", "probable"
    if run_name == "yolo26n_v2":
        return "ds-d02-605", "probable"
    if run_name.startswith(("stat_seed_", "stat_w0_", "stat_e100_")):
        return "ds-d03-variable", "confirmed"
    if run_name.startswith("stat_fixed_e150_b16_") or run_name.startswith(
        "stat_fixed_yolo26n_base_"
    ):
        return "ds-d03-fixed-test", "probable"
    if run_name.startswith("stat_fixed_valtest_yolo26n_base_"):
        return "ds-d03-fixed-valtest", "confirmed"
    return "ds-d04-r260719001", "confirmed"


def _initial_model_for_run(run_name: str) -> str:
    if run_name == "train6":
        return "model-y11m-generic-external"
    if run_name in {"yolo26n_baseline", "yolo26n_v2"}:
        return "model-y26n-generic"
    if run_name.startswith(
        ("stat_seed_", "stat_w0_", "stat_e100_", "stat_fixed_e150_b16_")
    ):
        return "model-y26n-d02-s00-v2-best"
    if run_name.startswith(
        ("stat_fixed_yolo26n_base_", "stat_fixed_valtest_yolo26n_base_")
    ):
        return "model-y26n-generic"
    if run_name.startswith("dataset_260719001"):
        return "model-y26n-d03fixed-s43-best"
    if run_name == "release_canaleta_pvc_260717_export_selected2":
        return "model-y26n-d03fixed-s43-best"
    if run_name == "t_20260723T041718_508d56":
        return "model-y26n-d02-s43-best"
    return "model-y26n-generic"


def _register_known_models(ws: Workspace) -> None:
    specs = [
        ("model-y11n-generic", "yolo11n.pt", "base", None, None),
        ("model-y12n-generic", "yolo12n.pt", "base", None, None),
        ("model-y26n-generic", "yolo26n.pt", "base", None, None),
        ("model-y26n-d02-s00-v2-best", "yolo26n_best.pt", "promoted", "model-y26n-generic", "yolo26n_v2"),
        ("model-y26n-d02-s43-best", "yolo26n_seed_43.pt", "promoted", "model-y26n-d02-s00-v2-best", "stat_w0_seed_43"),
        ("model-y26n-d03fixed-s43-best", "yolo26n_fixed_valtest_seed43_best.pt", "baseline", "model-y26n-generic", "stat_fixed_valtest_yolo26n_base_e150_b16_seed_43"),
        ("model-y26n-d04r001-s42-best", "yolo26n_dataset_260719001_partial_best.pt", "candidate", "model-y26n-d03fixed-s43-best", "dataset_260719001__yolo26n_fixed_valtest_seed43_best__20260720_012436_571432"),
    ]
    for model_id, filename, state, parent, run_id in specs:
        path = ws.models_root / filename
        if not path.is_file():
            continue
        aliases = [candidate for candidate in ws.models_root.glob("*.pt") if sha256(candidate) == sha256(path)]
        record = {
            "model_id": model_id,
            "architecture": Path(filename).stem.split("_")[0],
            "role": "generic_pretrained" if parent is None else "trained_checkpoint",
            "state": state,
            "sha256": sha256(path),
            "paths": sorted(candidate.relative_to(ws.root).as_posix() for candidate in aliases),
            "parent_model_id": parent,
            "source_run_id": run_id,
            "created_at": _iso_mtime(path),
            "provenance": {
                "origin": "reconstructed",
                "confidence": "confirmed",
                "reconstructed_at": utc_now(),
                "evidence": [candidate.relative_to(ws.root).as_posix() for candidate in aliases],
            },
        }
        register_model(ws, record, aliases=aliases, replace=True)

    external = {
        "model_id": "model-y11m-generic-external",
        "architecture": "yolo11m",
        "role": "generic_pretrained",
        "state": "external",
        "sha256": None,
        "paths": ["models/yolo11m.pt"],
        "parent_model_id": None,
        "source_run_id": None,
        "created_at": None,
        "provenance": {
            "origin": "reconstructed",
            "confidence": "incomplete",
            "reconstructed_at": utc_now(),
            "evidence": [],
            "notes": ["O peso inicial de train6 não foi preservado."],
        },
    }
    register_model(ws, external, replace=True)


def _run_record(
    ws: Workspace,
    source_root: Path,
    run_name: str,
    *,
    source_path: Path,
    destination: Path,
) -> dict[str, Any]:
    args_path = destination / "args.yaml"
    results_path = destination / "results.csv"
    args = load_yaml(args_path) if args_path.is_file() else {}
    dataset_id, confidence = _dataset_for_run(run_name)
    state = (
        "discarded"
        if run_name == "t_20260723T041718_508d56"
        else "candidate"
        if run_name.startswith("dataset_260719001")
        else "experimental"
    )
    evidence = [str(source_root / "docs" / "historico_treinamentos_modelos.md")]
    if run_name.startswith("stat_"):
        evidence.extend(
            str(path)
            for path in (
                source_root / "runs" / "detect" / "stat_summary.csv",
                source_root / "runs" / "detect" / "stat_aggregate.csv",
            )
            if path.is_file()
        )
    evidence.extend(
        str(path)
        for path in (source_path / "args.yaml", source_path / "results.csv")
        if path.is_file()
    )
    record = {
        "schema_version": 1,
        "run_id": run_name,
        "legacy_name": run_name,
        "status": "completed",
        "state": state,
        "created_at": _iso_mtime(args_path) or _iso_mtime(destination),
        "started_at": None,
        "completed_at": _iso_mtime(destination / "weights" / "best.pt"),
        "dataset_id": dataset_id,
        "initial_model_id": _initial_model_for_run(run_name),
        "initial_model_sha256": (
            list_registered_models(ws)
            .get(_initial_model_for_run(run_name), {})
            .get("sha256")
        ),
        "output_model_id": None,
        "training": args,
        "metrics": {},
        "artifacts": {},
        "git_commit": None,
        "provenance": {
            "origin": "reconstructed",
            "confidence": confidence,
            "reconstructed_at": utc_now(),
            "evidence": evidence,
        },
    }
    if run_name.startswith("dataset_260719001"):
        record["output_model_id"] = "model-y26n-d04r001-s42-best"
        best_model = (
            source_root / "models" / "yolo26n_dataset_260719001_partial_best.pt"
        )
        record["training"] = {
            "model": "models/yolo26n_fixed_valtest_seed43_best.pt",
            "epochs": 50,
            "imgsz": 640,
            "batch": 12,
            "seed": 42,
            "patience": 20,
            "cos_lr": True,
            "close_mosaic": 5,
            "flipud": 0.3,
        }
        record["metrics"] = {
            "completed_epochs": 50,
            "best_epoch": 43,
            "best": {
                "precision": 0.99217,
                "recall": 0.95652,
                "map50": 0.99031,
                "map50_95": 0.89548,
            },
        }
        if best_model.is_file():
            record["artifacts"] = {
                "best": {
                    "path": str(best_model),
                    "sha256": sha256(best_model),
                }
            }
        record["provenance"]["evidence"].extend(
            [
                str(
                    source_root
                    / "models"
                    / "yolo26n_dataset_260719001_partial_best.pt"
                ),
                str(
                    source_root
                    / "runs"
                    / "experiments"
                    / "20260720_new_model_video_eval"
                    / "comparison.md"
                ),
            ]
        )
    if results_path.is_file():
        record["provenance"]["evidence"].append(str(results_path))
    return record


def migrate_fish_detection_history(
    ws: Workspace,
    source_root: Path,
    *,
    copy_runs: bool = True,
    copy_models: bool = True,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not (source_root / "docs" / "historico_treinamentos_modelos.md").is_file():
        raise WorkflowError("O diretório informado não contém o histórico esperado.")

    copied_models = 0
    skipped_models = 0
    if copy_models:
        ws.models_root.mkdir(parents=True, exist_ok=True)
        for source in (source_root / "models").iterdir():
            if source.name == "README.md":
                continue
            result = _copy_without_overwrite(source, ws.models_root / source.name)
            copied_models += result == "copied"
            skipped_models += result != "copied"

    for record in _dataset_records(source_root):
        register_dataset(ws, record, replace=True)
    _register_known_models(ws)

    source_runs = source_root / "runs" / "detect"
    ws.runs_root.mkdir(parents=True, exist_ok=True)
    copied_run_summaries = 0
    for source_file in sorted(source_runs.iterdir()):
        if not source_file.is_file():
            continue
        copied_run_summaries += (
            _copy_without_overwrite(source_file, ws.runs_root / source_file.name)
            == "copied"
        )
    migrated_runs: list[str] = []
    existing_runs: list[str] = []
    for source_run in sorted(source_runs.iterdir()):
        if not source_run.is_dir():
            continue
        destination = ws.runs_root / source_run.name
        if copy_runs:
            result = _copy_without_overwrite(source_run, destination)
        else:
            destination = source_run
            result = "external"
        (migrated_runs if result == "copied" else existing_runs).append(source_run.name)

    # Inclui também os treinamentos que já nasceram no Dataset Studio.
    all_run_names = sorted(
        {
            *migrated_runs,
            *existing_runs,
            *[path.name for path in ws.runs_root.iterdir() if path.is_dir()],
        }
    )
    for run_name in all_run_names:
        destination = ws.runs_root / run_name
        source_path = (
            source_runs / run_name
            if (source_runs / run_name).is_dir()
            else destination
        )
        record = _run_record(
            ws,
            source_root,
            run_name,
            source_path=source_path,
            destination=destination,
        )
        register_run(ws, record, replace=True)
        state = record["state"]
        finalized = finalize_training_record(
            ws,
            run_name,
            "completed",
            state=state,
        )
        if run_name.startswith("dataset_260719001"):
            finalized["output_model_id"] = "model-y26n-d04r001-s42-best"
            register_run(ws, finalized, replace=True)
        dump_yaml(destination / "run.yaml", finalized)
        workflow_path = destination / "workflow_job.json"
        if not workflow_path.exists():
            workflow = {
                "id": f"reconstructed-{run_name}",
                "kind": "training",
                "target": finalized.get("dataset_id"),
                "status": "completed",
                "created_at": finalized.get("created_at"),
                "started_at": finalized.get("started_at"),
                "returncode": 0,
                "metadata": {
                    "training_id": run_name,
                    "dataset_id": finalized.get("dataset_id"),
                    "initial_model_id": finalized.get("initial_model_id"),
                    "provenance_origin": "reconstructed",
                },
            }
            workflow_path.write_text(
                json.dumps(workflow, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # Aliases promovidos adicionais, reconhecidos por hash.
    _register_known_models(ws)
    validation = validate_registry(ws)
    report = {
        "schema_version": 1,
        "migrated_at": utc_now(),
        "source_root": str(source_root),
        "workspace": str(ws.root),
        "copied_models": copied_models,
        "existing_or_identical_models": skipped_models,
        "copied_runs": len(migrated_runs),
        "existing_runs": len(existing_runs),
        "registered_runs": len(all_run_names),
        "copied_run_summaries": copied_run_summaries,
        "registered_datasets": len(_dataset_records(source_root)),
        "validation": validation,
    }
    report_path = ws.registry_root / "migration_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migra o histórico do fish_detection para o Dataset Studio."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--no-copy-runs", action="store_true")
    parser.add_argument("--no-copy-models", action="store_true")
    args = parser.parse_args(argv)
    report = migrate_fish_detection_history(
        Workspace.from_path(args.workspace),
        args.source,
        copy_runs=not args.no_copy_runs,
        copy_models=not args.no_copy_models,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
