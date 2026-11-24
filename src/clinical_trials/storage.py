"""Transactional run-scoped storage and collision-resistant output paths."""

import hashlib
from pathlib import Path
import re
import sqlite3

from .models import Summary


def filename(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")[:60] or "item"
    return stem + "-" + hashlib.sha256(value.encode()).hexdigest()[:10]


class SummaryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS summaries (
                    run_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, id)
                )"""
            )

    def save(self, run_id: str, summaries: list[Summary]) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.executemany(
                """INSERT INTO summaries (run_id, id, original_text, summary, error)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, id) DO UPDATE SET
                       original_text=excluded.original_text,
                       summary=excluded.summary, error=excluded.error""",
                [
                    (run_id, item.patient_id, item.text, item.summary, item.error)
                    for item in summaries
                ],
            )

    def rows(self, run_id: str) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, original_text, summary, error FROM summaries WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]
