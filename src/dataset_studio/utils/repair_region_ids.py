"""Utilitário de migração para reparar IDs de regiões no banco SQLite3 do Label Studio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

VALID_REGION_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def default_database_path() -> Path:
    """Localiza o caminho padrão da base SQLite3 do Label Studio no sistema Windows."""

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA não está definido.")
    return (
        Path(local_app_data)
        / "label-studio"
        / "label-studio"
        / "label_studio.sqlite3"
    )


def safe_region_id(scope: str, original_id: str) -> str:
    """Gera um ID seguro de região compativel com o Label Studio a partir do hash do ID original."""

    digest = hashlib.sha1(f"{scope}:{original_id}".encode("utf-8")).hexdigest()[:20]
    return f"region_{digest}"


def repair_result(raw_result: str | None, scope: str) -> tuple[str | None, int]:
    """Repara e substitui IDs inválidos de região em um campo de resultado do banco de dados."""

    if not raw_result:
        return raw_result, 0
    result = json.loads(raw_result)
    if not isinstance(result, list):
        raise ValueError(f"Resultado inválido em {scope}: era esperada uma lista.")

    replacements: dict[str, str] = {}
    changed = 0
    for item in result:
        if not isinstance(item, dict):
            continue
        original_id = item.get("id")
        if not isinstance(original_id, str) or VALID_REGION_ID.fullmatch(original_id):
            continue
        replacement = replacements.setdefault(
            original_id, safe_region_id(scope, original_id)
        )
        item["id"] = replacement
        changed += 1
    if not changed:
        return raw_result, 0
    return json.dumps(result, ensure_ascii=False), changed


def candidate_rows(
    connection: sqlite3.Connection, project_id: int
) -> list[tuple[str, int, str | None]]:
    rows: list[tuple[str, int, str | None]] = []
    rows.extend(
        ("prediction", row_id, result)
        for row_id, result in connection.execute(
            "SELECT id, result FROM prediction WHERE project_id = ?", (project_id,)
        )
    )
    rows.extend(
        ("task_completion", row_id, result)
        for row_id, result in connection.execute(
            "SELECT id, result FROM task_completion WHERE project_id = ?",
            (project_id,),
        )
    )
    rows.extend(
        ("tasks_annotationdraft", row_id, result)
        for row_id, result in connection.execute(
            """
            SELECT draft.id, draft.result
            FROM tasks_annotationdraft AS draft
            JOIN task ON task.id = draft.task_id
            WHERE task.project_id = ?
            """,
            (project_id,),
        )
    )
    return rows


def inspect_or_repair(
    database: Path, project_id: int, *, apply: bool = False
) -> dict[str, Any]:
    database = database.resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Banco do Label Studio não encontrado: {database}")

    connection = sqlite3.connect(database)
    backup_path: Path | None = None
    try:
        updates: list[tuple[str, int, str]] = []
        regions_by_table: dict[str, int] = {}
        for table, row_id, raw_result in candidate_rows(connection, project_id):
            repaired, changed = repair_result(raw_result, f"{table}:{row_id}")
            if changed and repaired is not None:
                updates.append((table, row_id, repaired))
                regions_by_table[table] = regions_by_table.get(table, 0) + changed

        if apply and updates:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = database.with_name(f"{database.stem}.backup_{stamp}.sqlite3")
            backup = sqlite3.connect(backup_path)
            try:
                connection.backup(backup)
            finally:
                backup.close()

            with connection:
                for table, row_id, repaired in updates:
                    connection.execute(
                        f"UPDATE {table} SET result = ? WHERE id = ?",
                        (repaired, row_id),
                    )

        return {
            "project_id": project_id,
            "records": len(updates),
            "regions": sum(regions_by_table.values()),
            "regions_by_table": regions_by_table,
            "applied": bool(apply and updates),
            "backup": str(backup_path) if backup_path else None,
        }
    finally:
        connection.close()
