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
    """Retorna chaves de unidades experimentais, com fallback para vídeos legados."""
    manifest = load_frame_manifest(defaults_or_ws, source_id)
    units = sorted(
        {
            str(frame.get("unit_id") or frame["source_video"])
            for frame in manifest["frames"]
        }
    )
    return [f"{source_id}/{unit_id}" for unit_id in units]


def assess_split_sufficiency(
    assignments: dict[str, list[str]],
    unit_metrics: dict[str, dict[str, Any]],
    evaluation_level: str,
) -> dict[str, Any]:
    """Produz bloqueios técnicos e alertas heurísticos por split."""
    required = ["train"]
    if evaluation_level != "pilot":
        required.extend(["val", "test_normal"])
    if evaluation_level == "robust":
        required.append("test_stress")
    recommended_frames = {
        "train": 100,
        "val": 30,
        "test_normal": 30,
        "test_stress": 30,
    }
    recommended_boxes = {
        "train": 50,
        "val": 10,
        "test_normal": 10,
        "test_stress": 10,
    }
    splits: dict[str, Any] = {}
    blocking: list[str] = []
    warnings: list[str] = []
    for split in SPLITS:
        keys = assignments.get(split) or []
        frames = sum(
            int(
                (unit_metrics.get(key) or {}).get(
                    "included",
                    (unit_metrics.get(key) or {}).get("completed", 0),
                )
                or 0
            )
            for key in keys
        )
        boxes = sum(
            int((unit_metrics.get(key) or {}).get("boxes", 0) or 0)
            for key in keys
        )
        splits[split] = {
            "units": len(keys),
            "frames": frames,
            "boxes": boxes,
        }
        if split in required and not keys:
            blocking.append(f"{split}: nenhuma unidade experimental atribuída")
        elif split in required and frames == 0:
            blocking.append(f"{split}: nenhuma imagem utilizável")
        if keys and frames < recommended_frames[split]:
            warnings.append(
                f"{split}: somente {frames} frames; referência heurística "
                f"de {recommended_frames[split]} ou mais"
            )
        if keys and boxes < recommended_boxes[split]:
            warnings.append(
                f"{split}: somente {boxes} caixas; referência heurística "
                f"de {recommended_boxes[split]} ou mais"
            )
    if evaluation_level == "pilot":
        warnings.append(
            "Versão piloto: métricas não comprovam generalização independente."
        )
    elif not assignments.get("test_stress"):
        warnings.append(
            "Sem teste de estresse: o relatório não medirá robustez em condições difíceis."
        )
    return {
        "evaluation_level": evaluation_level,
        "splits": splits,
        "blocking": blocking,
        "warnings": warnings,
        "threshold_kind": "heuristic_advisory",
    }



def create_version(
    defaults_or_ws: dict[str, Any] | Workspace,
    *,
    version_id: str | None = None,
    release_id: str | None = None,
    source_ids: list[str] | None = None,
    campaign_ids: list[str] | None = None,
    assignments: dict[str, list[str]],
    annotation_revisions: dict[str, str] | None = None,
    evaluation_level: str = "standard",
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
    unit_metrics: dict[str, dict[str, Any]] = {}
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
        for unit_id, metrics in (
            report.get("per_unit") or report.get("per_video") or {}
        ).items():
            unit_metrics[f"{src_id}/{unit_id}"] = metrics

    assigned = [key for split in SPLITS for key in assignments.get(split, [])]
    if len(assigned) != len(set(assigned)):
        raise WorkflowError("Um video foi atribuido a mais de um split.")
    if set(assigned) != expected:
        raise WorkflowError("Todos os videos devem ser atribuidos exatamente uma vez.")
    if evaluation_level not in {"pilot", "standard", "robust"}:
        raise WorkflowError(
            "evaluation_level deve ser pilot, standard ou robust."
        )
    if not assignments.get("train"):
        raise WorkflowError(
            "A versão exige ao menos uma unidade experimental em train."
        )
    if evaluation_level != "pilot":
        missing = [
            split
            for split in ("val", "test_normal")
            if not assignments.get(split)
        ]
        if missing:
            raise WorkflowError(
                "Uma versão avaliável exige unidades independentes em train, "
                "val e test_normal. Ausentes: " + ", ".join(missing)
            )
    if evaluation_level == "robust" and not assignments.get("test_stress"):
        raise WorkflowError(
            "Uma versão robusta exige ao menos uma unidade em test_stress."
        )
    effective_level = (
        "pilot"
        if evaluation_level == "pilot"
        else "robust"
        if assignments.get("test_stress")
        else "standard"
    )
    provisional = provisional or effective_level == "pilot"
    quality_assessment = assess_split_sufficiency(
        assignments,
        unit_metrics,
        effective_level,
    )
    if quality_assessment["blocking"]:
        raise WorkflowError(
            "A versão não pode ser materializada: "
            + "; ".join(quality_assessment["blocking"])
        )

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
        "evaluation_level": effective_level,
        "quality_assessment": quality_assessment,
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
            unit_id = str(frame.get("unit_id") or frame["source_video"])
            split = split_by_video[f"{src_id}/{unit_id}"]
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
                        "unit_id": unit_id,
                        "frame_index": str(frame["frame_index"]),
                        "timestamp_seconds": str(
                            frame.get("timestamp_seconds", "")
                        ),
                        "unit_timestamp_seconds": str(
                            frame.get("unit_timestamp_seconds", "")
                        ),
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
                    "unit_id": unit_id,
                    "frame_index": str(frame["frame_index"]),
                    "timestamp_seconds": str(
                        frame.get("timestamp_seconds", "")
                    ),
                    "unit_timestamp_seconds": str(
                        frame.get("unit_timestamp_seconds", "")
                    ),
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
    required_splits = (
        ("train",)
        if version.get("evaluation_level") == "pilot"
        else ("train", "val", "test_normal")
    )
    for required_split in required_splits:
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
        "val": (
            "images/val"
            if included_splits.get("val")
            else "images/train"
        ),
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
        "evaluation_level": version.get("evaluation_level", "legacy"),
        "quality_assessment": version.get("quality_assessment", {}),
        "warnings": list(
            (version.get("quality_assessment") or {}).get("warnings") or []
        )
        + (
            [
                "Versão piloto: a validação reutiliza o split de treino e não "
                "mede generalização."
            ]
            if version.get("evaluation_level") == "pilot"
            and not included_splits.get("val")
            else []
        ),
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
