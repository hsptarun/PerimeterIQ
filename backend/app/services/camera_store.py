"""SQLite storage for registered cameras."""

import sqlite3
from pathlib import Path
from typing import Any


class CameraStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def list_cameras(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cameras ORDER BY id"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_camera(self, camera_id: int):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cameras WHERE id = ?", (camera_id,)
            ).fetchone()
            return dict(row) if row else None

    def create_camera(self, name: str, source_type: str, source: str):
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO cameras (name, source_type, source)
                   VALUES (?, ?, ?)""",
                (name, source_type, source),
            )
            camera_id = cursor.lastrowid
        return self.get_camera(camera_id)

    def update_camera(self, camera_id: int, name: str, enabled: bool):
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE cameras SET name = ?, enabled = ?
                   WHERE id = ?""",
                (name, int(enabled), camera_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_camera(camera_id)

    def delete_camera(self, camera_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM cameras WHERE id = ?", (camera_id,)
            )
            return cursor.rowcount > 0
