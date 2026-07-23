"""Executa treinamento Ultralytics e avaliações finais dos splits de teste."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _metric_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_validation_metrics(result: Any) -> dict[str, Any]:
    """Converte o objeto de métricas da Ultralytics em dados serializáveis."""
    box = getattr(result, "box", None)
    speed = getattr(result, "speed", None)
    return {
        "precision": _metric_value(getattr(box, "mp", None)),
        "recall": _metric_value(getattr(box, "mr", None)),
        "map50": _metric_value(getattr(box, "map50", None)),
        "map50_95": _metric_value(getattr(box, "map", None)),
        "speed_ms": {
            str(key): _metric_value(value)
            for key, value in (speed or {}).items()
        },
    }


def split_inventory(version_root: Path, split: str) -> dict[str, int]:
    """Conta imagens e caixas do split usando o manifest imutável da versão."""
    manifest = version_root / "manifest.csv"
    inventory = {"images": 0, "boxes": 0}
    if not manifest.is_file():
        return inventory
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split or row.get("included", "true") != "true":
                continue
            inventory["images"] += 1
            try:
                inventory["boxes"] += int(row.get("boxes") or 0)
            except ValueError:
                pass
    return inventory


def calculate_robustness(evaluations: dict[str, Any]) -> dict[str, Any]:
    """Calcula a queda do teste normal para o teste de estresse."""
    normal = evaluations.get("test_normal") or {}
    stress = evaluations.get("test_stress") or {}
    if normal.get("status") != "completed" or stress.get("status") != "completed":
        return {"status": "not_available"}
    output: dict[str, Any] = {"status": "completed"}
    for metric in ("precision", "recall", "map50", "map50_95"):
        normal_value = _metric_value(normal.get(metric))
        stress_value = _metric_value(stress.get(metric))
        if normal_value is None or stress_value is None:
            continue
        absolute = normal_value - stress_value
        output[metric] = {
            "drop_absolute": absolute,
            "drop_relative": absolute / normal_value if normal_value else None,
        }
    return output


def _write_summary(run_root: Path, summary: dict[str, Any]) -> None:
    output = run_root / "evaluations" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)


def _evaluate_split(
    yolo_class: Any,
    best_path: Path,
    *,
    split_name: str,
    data_path: Path,
    run_root: Path,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    inventory = split_inventory(data_path.parent, split_name)
    if inventory["images"] == 0 or not data_path.is_file():
        return {
            "status": "not_available",
            **inventory,
            "reason": f"O split {split_name} não está disponível nesta versão.",
        }
    try:
        model = yolo_class(str(best_path))
        result = model.val(
            data=str(data_path),
            split="test",
            imgsz=imgsz,
            device=device,
            project=str(run_root / "evaluations"),
            name=split_name,
            exist_ok=True,
            plots=True,
        )
        return {
            "status": "completed",
            **inventory,
            **normalize_validation_metrics(result),
            "evaluated_at": _utc_now(),
        }
    except Exception as exc:  # avaliação não invalida pesos treinados
        return {
            "status": "failed",
            **inventory,
            "error": f"{type(exc).__name__}: {exc}",
            "evaluated_at": _utc_now(),
        }


def _parse_args() -> tuple[dict[str, Any], dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters", nargs="*")
    parsed = parser.parse_args()
    values: dict[str, Any] = {}
    for item in parsed.parameters:
        if "=" not in item:
            raise ValueError(f"Parâmetro inválido: {item}")
        key, value = item.split("=", 1)
        values[key] = _coerce_value(value)
    required = {"data", "model", "project", "name"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"Parâmetros obrigatórios ausentes: {', '.join(missing)}")
    runner = {
        "data": Path(str(values.pop("data"))).resolve(),
        "project": Path(str(values.get("project"))).resolve(),
        "name": str(values.get("name")),
    }
    return runner, values


def main() -> int:
    runner, train_args = _parse_args()
    from ultralytics import YOLO

    model = YOLO(str(train_args.pop("model")))
    model.train(data=str(runner["data"]), **train_args)

    run_root = runner["project"] / runner["name"]
    best_path = run_root / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Checkpoint best.pt não encontrado em {best_path}")

    imgsz = int(train_args.get("imgsz", 640))
    device = str(train_args.get("device", "cpu"))
    evaluations = {
        "test_normal": _evaluate_split(
            YOLO,
            best_path,
            split_name="test_normal",
            data_path=runner["data"],
            run_root=run_root,
            imgsz=imgsz,
            device=device,
        ),
        "test_stress": _evaluate_split(
            YOLO,
            best_path,
            split_name="test_stress",
            data_path=runner["data"].with_name("data_test_stress.yaml"),
            run_root=run_root,
            imgsz=imgsz,
            device=device,
        ),
    }
    summary = {
        "schema_version": 1,
        "checkpoint": str(best_path),
        "generated_at": _utc_now(),
        "evaluations": evaluations,
        "robustness": calculate_robustness(evaluations),
    }
    _write_summary(run_root, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
