import sqlite3
from pathlib import Path

from dataset_studio.utils.repair_region_ids import inspect_or_repair, repair_result


def test_repair_result_replaces_invalid_ids_consistently():
    invalid = '[{"id": "inv alid", "type": "rectanglelabels"}]'
    repaired, changed = repair_result(invalid, "test:1")

    assert changed == 1
    assert "region_" in repaired
    assert "inv alid" not in repaired


def test_inspect_and_apply_repairs_project_and_creates_backup(tmp_path: Path):
    db_path = tmp_path / "label_studio.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE prediction (id INTEGER PRIMARY KEY, project_id INTEGER, result TEXT)"
    )
    conn.execute(
        "CREATE TABLE task_completion (id INTEGER PRIMARY KEY, project_id INTEGER, result TEXT)"
    )
    conn.execute(
        "CREATE TABLE tasks_annotationdraft (id INTEGER PRIMARY KEY, task_id INTEGER, result TEXT)"
    )
    conn.execute("CREATE TABLE task (id INTEGER PRIMARY KEY, project_id INTEGER)")
    conn.execute(
        "INSERT INTO prediction VALUES (1, 10, '[{\"id\": \"bad id\", \"type\": \"rectanglelabels\"}]')"
    )
    conn.commit()
    conn.close()

    res = inspect_or_repair(db_path, 10, apply=True)
    assert res["records"] == 1
    assert res["applied"] is True
    assert res["backup"] is not None
