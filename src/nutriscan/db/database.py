"""Thread-safe SQLite WAL database for NutriScan logs and offline sync queue."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from nutriscan.core.logger import get_logger

logger = get_logger("Database")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_uuid TEXT NOT NULL UNIQUE,
    device_id TEXT NOT NULL,
    location TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    weight_total_g REAL,
    detection_count INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    energi_kcal REAL,
    protein_g REAL,
    lemak_g REAL,
    karbohidrat_g REAL,
    image_path TEXT,
    sync_status TEXT NOT NULL DEFAULT 'pending',
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    next_retry_at REAL NOT NULL DEFAULT 0.0,
    synced_at TEXT,
    created_at_epoch REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_sync_due
    ON scans(sync_status, next_retry_at);
"""


class Database:
    """Thread-safe offline-first SQLite WAL storage adapted from DICA."""

    def __init__(self, db_path: Path | str = "data/nutriscan.db") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create database directory, enable WAL mode, and ensure schema exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        logger.info("Database SQLite siap: %s", self.db_path)

    def save_scan(
        self,
        payload: dict[str, Any],
        image_path: Path | str | None = None,
        sync_status: str = "pending",
    ) -> int:
        """Insert a scan record safely using parameterized queries."""
        nutrition = payload.get("total_gizi") or {}
        img_str = str(image_path) if image_path else None
        payload_str = json.dumps(payload, ensure_ascii=False)

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scans (
                    scan_uuid, device_id, location, timestamp_utc,
                    weight_total_g, detection_count, payload_json,
                    energi_kcal, protein_g, lemak_g, karbohidrat_g,
                    image_path, sync_status, next_retry_at, created_at_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(payload["scan_id"]),
                    str(payload["device_id"]),
                    str(payload["location"]),
                    str(payload["timestamp_utc"]),
                    payload.get("berat_total_g"),
                    int(payload.get("jumlah_deteksi", 0)),
                    payload_str,
                    nutrition.get("energi_kcal"),
                    nutrition.get("protein_g"),
                    nutrition.get("lemak_g"),
                    nutrition.get("karbohidrat_g"),
                    img_str,
                    sync_status,
                    0.0,
                    time.time(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def fetch_pending_scans(
        self,
        now: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Fetch due pending records ordered oldest-first."""
        ts = time.time() if now is None else now
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT scan_uuid, payload_json, image_path, retry_count
                FROM scans
                WHERE sync_status = 'pending' AND next_retry_at <= ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (ts, limit),
            ).fetchall()

        return [
            {
                "scan_uuid": row["scan_uuid"],
                "payload": json.loads(row["payload_json"]),
                "image_path": row["image_path"],
                "retry_count": int(row["retry_count"]),
            }
            for row in rows
        ]

    def mark_synced(self, scan_uuid: str, synced_at: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE scans
                SET sync_status = 'synced', last_error = NULL, synced_at = ?
                WHERE scan_uuid = ?
                """,
                (synced_at, scan_uuid),
            )
            conn.commit()

    def mark_retry(
        self,
        scan_uuid: str,
        error: str,
        next_retry_at: float,
        dead_letter: bool = False,
    ) -> None:
        status = "dead_letter" if dead_letter else "pending"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE scans
                SET sync_status = ?,
                    retry_count = retry_count + 1,
                    last_error = ?,
                    next_retry_at = ?
                WHERE scan_uuid = ?
                """,
                (status, error[:500], next_retry_at, scan_uuid),
            )
            conn.commit()

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_json"])
        return {
            "id": int(row["id"]),
            "scan_id": row["scan_uuid"],
            "device_id": row["device_id"],
            "location": row["location"],
            "timestamp_utc": row["timestamp_utc"],
            "berat_total_g": row["weight_total_g"],
            "jumlah_deteksi": int(row["detection_count"]),
            "komponen": payload.get("komponen", {}),
            "total_gizi": payload.get("total_gizi"),
            "image_path": row["image_path"],
            "sync_status": row["sync_status"],
            "retry_count": int(row["retry_count"]),
            "synced_at": row["synced_at"],
        }

    def get_recent_scans(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scans ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def get_all_scans(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM scans ORDER BY id DESC").fetchall()
        return [self._row_to_summary(r) for r in rows]

    def get_queue_stats(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT sync_status, COUNT(*) AS cnt FROM scans GROUP BY sync_status"
            ).fetchall()
        stats = {"pending": 0, "synced": 0, "local_only": 0, "dead_letter": 0}
        for row in rows:
            stats[str(row["sync_status"])] = int(row["cnt"])
        stats["total"] = sum(stats.values())
        return stats

    def clear_scans(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM scans")
            conn.commit()
