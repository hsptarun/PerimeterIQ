"""Persistent Supabase camera storage with local SQLite fallback."""

import sqlite3
from pathlib import Path
from typing import Any

import httpx


class CameraStore:
    def __init__(
        self,
        db_path: Path,
        supabase_url: str = "",
        supabase_key: str = "",
    ):
        self.db_path = db_path
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.use_supabase = bool(self.supabase_url and self.supabase_key)

        if not self.use_supabase:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    def _headers(self, prefer: str = "") -> dict:
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def _request(self, method: str, params=None, json=None, prefer=""):
        response = httpx.request(
            method,
            f"{self.supabase_url}/rest/v1/cameras",
            headers=self._headers(prefer),
            params=params,
            json=json,
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json() if response.content else []

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
        if self.use_supabase:
            return self._request("GET", {"select": "*", "order": "id.asc"})
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cameras ORDER BY id"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_camera(self, camera_id: int):
        if self.use_supabase:
            rows = self._request(
                "GET",
                {"select": "*", "id": f"eq.{camera_id}"},
            )
            return rows[0] if rows else None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cameras WHERE id = ?", (camera_id,)
            ).fetchone()
            return dict(row) if row else None

    def create_camera(self, name: str, source_type: str, source: str):
        if self.use_supabase:
            rows = self._request(
                "POST",
                json={
                    "name": name,
                    "source_type": source_type,
                    "source": source,
                },
                prefer="return=representation",
            )
            if not rows:
                raise RuntimeError("Supabase did not return the created camera")
            return rows[0]

        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO cameras (name, source_type, source)
                   VALUES (?, ?, ?)""",
                (name, source_type, source),
            )
            camera_id = cursor.lastrowid
        return self.get_camera(camera_id)

    def update_camera(self, camera_id: int, name: str, enabled: bool):
        if self.use_supabase:
            rows = self._request(
                "PATCH",
                params={"id": f"eq.{camera_id}"},
                json={"name": name, "enabled": enabled},
                prefer="return=representation",
            )
            return rows[0] if rows else None

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
        if self.use_supabase:
            rows = self._request(
                "DELETE",
                params={"id": f"eq.{camera_id}", "select": "id"},
                prefer="return=representation",
            )
            return bool(rows)

        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM cameras WHERE id = ?", (camera_id,)
            )
            return cursor.rowcount > 0
