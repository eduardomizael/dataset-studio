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


def resolve_class_mapping(
    source_classes: dict[str, list[str]],
    source_class_counts: dict[str, dict[str, int]],
    *,
    class_mapping: dict[str, dict[str, str | None]] | None = None,
    final_classes: list[str] | None = None,
    acknowledged: bool = False,
) -> dict[str, Any]:
    """Valida e descreve a transformação de classes de uma release."""
    if not source_classes:
        raise WorkflowError("Nenhuma origem foi informada para o mapeamento de classes.")

    schemas = list(source_classes.values())
    exact_match = all(schema == schemas[0] for schema in schemas[1:])
    if class_mapping is None:
        if not exact_match:
            raise WorkflowError(
                "As origens possuem classes diferentes. Defina e confirme o "
                "mapeamento de classes da release."
            )
        final = list(schemas[0])
        mapping = {
            source_id: {class_name: class_name for class_name in classes}
            for source_id, classes in source_classes.items()
        }
    else:
        if set(class_mapping) != set(source_classes):
            raise WorkflowError(
                "class_mapping deve indicar exatamente um mapeamento por origem."
            )
        mapping: dict[str, dict[str, str | None]] = {}
        for source_id, classes in source_classes.items():
            raw_mapping = class_mapping[source_id]
            if set(raw_mapping) != set(classes):
                raise WorkflowError(
                    f"O mapeamento de {source_id} deve definir uma ação para "
                    "cada classe original."
                )
            normalized: dict[str, str | None] = {}
            for original_name in classes:
                target = raw_mapping[original_name]
                if target is None:
                    normalized[original_name] = None
                    continue
                if not isinstance(target, str) or not target:
                    raise WorkflowError(
                        f"Destino inválido para {source_id}/{original_name}."
                    )
                if target != target.strip():
                    raise WorkflowError(
                        f"A classe final '{target}' possui espaços nas extremidades."
                    )
                normalized[original_name] = target
            mapping[source_id] = normalized

        if final_classes is None:
            final = []
            for source_id in source_classes:
                for original_name in source_classes[source_id]:
                    target = mapping[source_id][original_name]
                    if target is not None and target not in final:
                        final.append(target)
        else:
            final = list(final_classes)

    if not final:
        raise WorkflowError("A release deve preservar ao menos uma classe final.")
    if any(not isinstance(name, str) or not name or name != name.strip() for name in final):
        raise WorkflowError("As classes finais devem possuir nomes não vazios e sem espaços externos.")
    if len(final) != len(set(final)):
        raise WorkflowError("A lista de classes finais possui nomes duplicados.")

    used_targets = {
        target
        for source_mapping in mapping.values()
        for target in source_mapping.values()
        if target is not None
    }
    unknown_targets = sorted(used_targets - set(final))
    unused_targets = sorted(set(final) - used_targets)
    if unknown_targets:
        raise WorkflowError(
            "Classes de destino ausentes em final_classes: "
            + ", ".join(unknown_targets)
        )
    if unused_targets:
        raise WorkflowError(
            "Classes finais sem nenhuma classe de origem associada: "
            + ", ".join(unused_targets)
        )

    warnings: list[str] = []
    renamed_boxes = 0
    ignored_boxes = 0
    affected_boxes = 0
    target_origins: dict[str, set[str]] = {name: set() for name in final}
    for source_id, classes in source_classes.items():
        targets_in_source: set[str] = set()
        for original_name in classes:
            target = mapping[source_id][original_name]
            boxes = int((source_class_counts.get(source_id) or {}).get(original_name, 0))
            if target is None:
                ignored_boxes += boxes
                affected_boxes += boxes
                warnings.append(
                    f"{source_id}: a classe '{original_name}' será ignorada "
                    f"e {boxes} caixa(s) serão removidas."
                )
                continue
            targets_in_source.add(target)
            target_origins[target].add(original_name)
            if target != original_name:
                renamed_boxes += boxes
                affected_boxes += boxes
                warnings.append(
                    f"{source_id}: '{original_name}' será convertida para "
                    f"'{target}' em {boxes} caixa(s)."
                )
        missing = [name for name in final if name not in targets_in_source]
        if missing:
            warnings.append(
                f"{source_id}: não possui classes associadas a "
                + ", ".join(f"'{name}'" for name in missing)
                + "; confirme que não são objetos presentes sem anotação."
            )
        ordered_targets = [
            mapping[source_id][name]
            for name in classes
            if mapping[source_id][name] is not None
        ]
        if ordered_targets != [name for name in final if name in ordered_targets]:
            warnings.append(
                f"{source_id}: a ordem dos IDs será remapeada para o esquema final."
            )

    fused_classes: dict[str, list[str]] = {}
    for target, originals in target_origins.items():
        if len(originals) > 1:
            fused_classes[target] = sorted(originals)
            warnings.append(
                f"As classes {', '.join(repr(name) for name in sorted(originals))} "
                f"serão fundidas em '{target}'."
            )

    requires_acknowledgement = not exact_match or bool(warnings)
    if requires_acknowledgement and not acknowledged:
        raise WorkflowError(
            "O mapeamento altera ou combina esquemas de classes. "
            "Confirme explicitamente os avisos antes de criar a release."
        )

    return {
        "original_classes": source_classes,
        "final_classes": final,
        "mapping": mapping,
        "warnings": warnings,
        "requires_acknowledgement": requires_acknowledgement,
        "acknowledged": bool(acknowledged),
        "acknowledged_at": utc_now() if acknowledged else None,
        "affected_boxes": affected_boxes,
        "renamed_boxes": renamed_boxes,
        "ignored_boxes": ignored_boxes,
        "fused_classes": fused_classes,
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
    class_mapping: dict[str, dict[str, str | None]] | None = None,
    final_classes: list[str] | None = None,
    class_mapping_acknowledged: bool = False,
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
    selected_revisions: dict[str, str] = {}
    unit_metrics: dict[str, dict[str, Any]] = {}
    source_classes: dict[str, list[str]] = {}
    source_class_counts: dict[str, dict[str, int]] = {}
    provisional = False
    for src_id in targets:
        source = load_source(defaults_or_ws, src_id)
        source_classes[src_id] = list(source["annotation"]["classes"])

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
        source_class_counts[src_id] = {
            str(name): int(count)
            for name, count in (report.get("class_counts") or {}).items()
        }
        provisional = provisional or report.get("snapshot_type") == "provisional"
        expected.update(source_video_keys(defaults_or_ws, src_id))
        for unit_id, metrics in (
            report.get("per_unit") or report.get("per_video") or {}
        ).items():
            unit_metrics[f"{src_id}/{unit_id}"] = metrics

    class_resolution = resolve_class_mapping(
        source_classes,
        source_class_counts,
        class_mapping=class_mapping,
        final_classes=final_classes,
        acknowledged=class_mapping_acknowledged,
    )

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
        "class_resolution": class_resolution,
        "class_mapping": class_resolution["mapping"],
        "assignments": {split: assignments.get(split, []) for split in SPLITS},
        "materialization": "copy",
        "classes": class_resolution["final_classes"],
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
    final_class_ids = {
        class_name: index for index, class_name in enumerate(version["classes"])
    }
    for src_id in sources_list:
        source = load_source(defaults_or_ws, src_id)
        original_classes = list(source["annotation"]["classes"])
        source_mapping = (version.get("class_mapping") or {}).get(src_id) or {
            class_name: class_name for class_name in original_classes
        }
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
            remapped_boxes: list[str] = []
            for box in annotation.boxes:
                parts = box.split()
                if len(parts) != 5:
                    raise WorkflowError(
                        f"Anotação YOLO inválida em {src_id}/{frame_id}."
                    )
                original_id = int(parts[0])
                if original_id < 0 or original_id >= len(original_classes):
                    raise WorkflowError(
                        f"ID de classe inválido em {src_id}/{frame_id}: "
                        f"{original_id}."
                    )
                original_name = original_classes[original_id]
                target_name = source_mapping.get(original_name)
                if target_name is None:
                    continue
                if target_name not in final_class_ids:
                    raise WorkflowError(
                        f"Classe final desconhecida no mapeamento: {target_name}."
                    )
                remapped_boxes.append(
                    " ".join([str(final_class_ids[target_name]), *parts[1:]])
                )
            label_text = "\n".join(remapped_boxes)
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
                    "boxes": str(len(remapped_boxes)),
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
        "class_resolution": version.get("class_resolution", {}),
        "warnings": list(
            (version.get("quality_assessment") or {}).get("warnings") or []
        )
        + list((version.get("class_resolution") or {}).get("warnings") or [])
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
