from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


JOB_STATUSES = {
    "pending",
    "running",
    "completed",
    "transient_failed",
    "permanent_failed",
    "not_a_game",
    "store_unavailable",
}


@dataclass(frozen=True, slots=True)
class ProfileJob:
    job_id: int
    appid: int
    job_type: str
    status: str
    priority: int
    attempt_count: int


class SteamProfileStore:
    """SQLite registry, core-profile cache, and resumable collection queue."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_registry (
                    appid INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    app_type TEXT NOT NULL DEFAULT 'game',
                    profile_status TEXT NOT NULL DEFAULT 'not_collected',
                    last_checked_at TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS core_profiles (
                    appid INTEGER PRIMARY KEY REFERENCES app_registry(appid),
                    profile_json TEXT NOT NULL,
                    profile_path TEXT,
                    completeness REAL NOT NULL DEFAULT 0,
                    collected_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profile_jobs (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    appid INTEGER NOT NULL REFERENCES app_registry(appid),
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_retry_at TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(appid, job_type)
                );

                CREATE INDEX IF NOT EXISTS idx_profile_jobs_claim
                ON profile_jobs(status, priority DESC, created_at ASC);

                CREATE INDEX IF NOT EXISTS idx_registry_profile_status
                ON app_registry(profile_status);
                """
            )

    def sync_catalog_file(self, catalog_path: Path) -> int:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        apps = payload.get("apps", payload)
        if not isinstance(apps, list):
            raise ValueError(f"Invalid Steam catalog: {catalog_path}")
        return self.sync_registry(item for item in apps if isinstance(item, dict))

    def sync_registry(self, apps: Iterable[dict[str, Any]]) -> int:
        now = _utc_now()
        rows: list[tuple[int, str, str, str]] = []
        for app in apps:
            try:
                appid = int(app.get("appid") or app.get("id"))
            except (TypeError, ValueError):
                continue
            name = str(app.get("name") or "").strip()
            if appid <= 0 or not name:
                continue
            rows.append((appid, name, str(app.get("type") or "game"), now))
        with self._connection() as connection:
            connection.executemany(
                """
                INSERT INTO app_registry(appid, name, app_type, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(appid) DO UPDATE SET
                    name=excluded.name,
                    app_type=excluded.app_type,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def import_profile_directory(self, profiles_dir: Path) -> int:
        imported = 0
        for path in sorted(profiles_dir.glob("*.json")):
            try:
                profile = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(profile, dict) or not profile.get("appid"):
                continue
            self.upsert_core_profile(profile, profile_path=path)
            imported += 1
        return imported

    def upsert_core_profile(
        self,
        profile: dict[str, Any],
        *,
        profile_path: Path | None = None,
    ) -> None:
        appid = int(profile["appid"])
        name = str(profile.get("name") or f"Steam App {appid}")
        collected_at = str(profile.get("profile_updated_at") or _utc_now())
        expires_at = str(
            profile.get("profile_expires_at")
            or (_parse_datetime(collected_at) + timedelta(days=30)).isoformat()
        )
        completeness = float(profile.get("profile_completeness") or 0.0)
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO app_registry(appid, name, profile_status, last_checked_at, updated_at)
                VALUES (?, ?, 'completed', ?, ?)
                ON CONFLICT(appid) DO UPDATE SET
                    name=excluded.name,
                    profile_status='completed',
                    last_checked_at=excluded.last_checked_at,
                    last_error=NULL,
                    updated_at=excluded.updated_at
                """,
                (appid, name, collected_at, now),
            )
            connection.execute(
                """
                INSERT INTO core_profiles(appid, profile_json, profile_path, completeness, collected_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(appid) DO UPDATE SET
                    profile_json=excluded.profile_json,
                    profile_path=excluded.profile_path,
                    completeness=excluded.completeness,
                    collected_at=excluded.collected_at,
                    expires_at=excluded.expires_at
                """,
                (
                    appid,
                    json.dumps(profile, ensure_ascii=False),
                    str(profile_path) if profile_path else None,
                    completeness,
                    collected_at,
                    expires_at,
                ),
            )

    def load_core_profiles(self, *, include_expired: bool = True) -> list[tuple[Path, dict[str, Any]]]:
        query = "SELECT appid, profile_json, profile_path FROM core_profiles"
        params: tuple[Any, ...] = ()
        if not include_expired:
            query += " WHERE expires_at > ?"
            params = (_utc_now(),)
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        profiles: list[tuple[Path, dict[str, Any]]] = []
        for row in rows:
            try:
                profile = json.loads(row["profile_json"])
            except json.JSONDecodeError:
                continue
            path = Path(row["profile_path"] or f"db_profile_{row['appid']}.json")
            profiles.append((path, profile))
        return profiles

    def enqueue(
        self,
        appid: int,
        *,
        job_type: str = "core_profile",
        priority: int = 50,
        force: bool = False,
    ) -> None:
        now = _utc_now()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT status FROM profile_jobs WHERE appid=? AND job_type=?",
                (appid, job_type),
            ).fetchone()
            if existing and existing["status"] == "completed" and not force:
                return
            connection.execute(
                """
                INSERT INTO profile_jobs(appid, job_type, status, priority, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(appid, job_type) DO UPDATE SET
                    status='pending',
                    priority=MAX(profile_jobs.priority, excluded.priority),
                    next_retry_at=NULL,
                    last_error=NULL,
                    updated_at=excluded.updated_at
                """,
                (appid, job_type, priority, now, now),
            )
            connection.execute(
                "UPDATE app_registry SET profile_status='pending', updated_at=? WHERE appid=?",
                (now, appid),
            )

    def claim_next(self) -> ProfileJob | None:
        now = _utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM profile_jobs
                WHERE status IN ('pending', 'transient_failed')
                  AND (next_retry_at IS NULL OR next_retry_at <= ?)
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                """
                UPDATE profile_jobs
                SET status='running', attempt_count=attempt_count+1, updated_at=?
                WHERE job_id=?
                """,
                (now, row["job_id"]),
            )
            connection.execute(
                "UPDATE app_registry SET profile_status='running', updated_at=? WHERE appid=?",
                (now, row["appid"]),
            )
            connection.commit()
            return ProfileJob(
                int(row["job_id"]),
                int(row["appid"]),
                str(row["job_type"]),
                "running",
                int(row["priority"]),
                int(row["attempt_count"]) + 1,
            )
        finally:
            connection.close()

    def mark_completed(self, job_id: int) -> None:
        now = _utc_now()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT appid FROM profile_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Unknown profile job: {job_id}")
            connection.execute(
                "UPDATE profile_jobs SET status='completed', last_error=NULL, updated_at=? WHERE job_id=?",
                (now, job_id),
            )
            connection.execute(
                "UPDATE app_registry SET profile_status='completed', last_error=NULL, updated_at=? WHERE appid=?",
                (now, row["appid"]),
            )

    def mark_failed(
        self,
        job_id: int,
        error: str,
        *,
        status: str = "transient_failed",
        retry_after: timedelta = timedelta(minutes=5),
    ) -> None:
        if status not in JOB_STATUSES - {"pending", "running", "completed"}:
            raise ValueError(f"Invalid failure status: {status}")
        now = datetime.now(timezone.utc)
        next_retry = (
            (now + retry_after).isoformat() if status == "transient_failed" else None
        )
        with self._connection() as connection:
            row = connection.execute(
                "SELECT appid FROM profile_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Unknown profile job: {job_id}")
            connection.execute(
                """
                UPDATE profile_jobs
                SET status=?, next_retry_at=?, last_error=?, updated_at=?
                WHERE job_id=?
                """,
                (status, next_retry, error[:1000], now.isoformat(), job_id),
            )
            connection.execute(
                """
                UPDATE app_registry
                SET profile_status=?, last_error=?, updated_at=?
                WHERE appid=?
                """,
                (status, error[:1000], now.isoformat(), row["appid"]),
            )

    def registry_game(self, appid: int) -> tuple[int, str] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT appid, name FROM app_registry WHERE appid=?", (appid,)
            ).fetchone()
        return (int(row["appid"]), str(row["name"])) if row else None

    def summary(self) -> dict[str, Any]:
        with self._connection() as connection:
            registry_count = int(connection.execute("SELECT COUNT(*) FROM app_registry").fetchone()[0])
            profile_count = int(connection.execute("SELECT COUNT(*) FROM core_profiles").fetchone()[0])
            fresh_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM core_profiles WHERE expires_at > ?", (_utc_now(),)
                ).fetchone()[0]
            )
            job_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM profile_jobs GROUP BY status"
            ).fetchall()
            latest = connection.execute(
                "SELECT MAX(collected_at) FROM core_profiles"
            ).fetchone()[0]
        return {
            "registry_count": registry_count,
            "core_profile_count": profile_count,
            "fresh_core_profile_count": fresh_count,
            "job_statuses": {str(row["status"]): int(row["count"]) for row in job_rows},
            "last_profile_collected_at": latest,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
