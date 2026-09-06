"""Separated storage for 탐색 공간 and 게임별 플레이 공간.

기획안 §4.4 / §11의 구현 규칙을 저장 키로 강제한다.

* 탐색 대화는 ``user_id + discovery_session_id``
* 공략 대화는 ``user_id + game_id + thread_id``
* 같은 게임을 다시 시작하면 ``playthrough``로 회차를 구분해 이전 진행도를 덮어쓰지 않는다.

두 공간은 서로 다른 테이블에 저장되고, 조회 함수도 분리돼 있다. 공간 분리를
프롬프트 지시가 아니라 저장 키와 조회 조건으로 보장하기 위해서다. 공략
컨텍스트를 만드는 :meth:`WorkspaceStore.play_context` 는 탐색 대화를 읽을 수
있는 경로 자체를 갖지 않는다.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


#: §8 스포일러 정책. 허용 범위가 넓어지는 순서로 정의한다.
SPOILER_LEVELS = ("no_spoiler", "progress", "all")
DEFAULT_SPOILER_LEVEL = "no_spoiler"

PREFERENCE_KINDS = ("like", "dislike", "played", "avoid")
PREFERENCE_SCOPES = ("persistent", "session")

#: 게임별 플레이 공간의 기본 주제. §4.4 "'초반 가이드', '특정 보스', '장비와 빌드'".
DEFAULT_THREAD_TOPICS = (
    ("early_guide", "초반 가이드"),
    ("boss", "특정 보스"),
    ("build", "장비와 빌드"),
    ("system", "시스템 이해"),
)


@dataclass(frozen=True, slots=True)
class DiscoverySession:
    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    conditions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "conditions": self.conditions,
        }


@dataclass(frozen=True, slots=True)
class PlayThread:
    thread_id: str
    user_id: str
    appid: int
    playthrough: int
    topic: str
    title: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "appid": self.appid,
            "playthrough": self.playthrough,
            "topic": self.topic,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class GameState:
    """§11 '게임별 상태'. 사용자와 게임과 회차를 함께 기준으로 저장한다."""

    user_id: str
    appid: int
    playthrough: int
    progress: str = ""
    character_build: str = ""
    equipment: list[str] = field(default_factory=list)
    difficulties: list[str] = field(default_factory=list)
    spoiler_level: str = DEFAULT_SPOILER_LEVEL
    platform: str = ""
    game_version: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "appid": self.appid,
            "playthrough": self.playthrough,
            "progress": self.progress,
            "character_build": self.character_build,
            "equipment": list(self.equipment),
            "difficulties": list(self.difficulties),
            "spoiler_level": self.spoiler_level,
            "platform": self.platform,
            "game_version": self.game_version,
            "updated_at": self.updated_at,
        }

    @property
    def is_empty(self) -> bool:
        return not (self.progress or self.character_build or self.equipment)


@dataclass(frozen=True, slots=True)
class Attempt:
    attempt_id: int
    appid: int
    playthrough: int
    thread_id: str
    action: str
    outcome: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "appid": self.appid,
            "playthrough": self.playthrough,
            "thread_id": self.thread_id,
            "action": self.action,
            "outcome": self.outcome,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class Preference:
    """§11 '지속 취향' 및 '현재 탐색 조건'. 근거와 함께 저장하고 수정·삭제할 수 있다."""

    preference_id: int
    user_id: str
    kind: str
    value: str
    label: str
    evidence: str
    scope: str
    session_id: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "preference_id": self.preference_id,
            "user_id": self.user_id,
            "kind": self.kind,
            "value": self.value,
            "label": self.label,
            "evidence": self.evidence,
            "scope": self.scope,
            "session_id": self.session_id,
            "created_at": self.created_at,
        }


class WorkspaceStore:
    """SQLite storage for discovery sessions, play threads, library, and memory."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # ------------------------------------------------------------------
    # connection helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
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
                CREATE TABLE IF NOT EXISTS discovery_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    conditions_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS discovery_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS library (
                    user_id TEXT NOT NULL,
                    appid INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    header_image TEXT NOT NULL DEFAULT '',
                    platform TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'manual',
                    note TEXT NOT NULL DEFAULT '',
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, appid)
                );

                CREATE TABLE IF NOT EXISTS play_threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    appid INTEGER NOT NULL,
                    playthrough INTEGER NOT NULL DEFAULT 1,
                    topic TEXT NOT NULL DEFAULT 'general',
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS play_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    appid INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS game_states (
                    user_id TEXT NOT NULL,
                    appid INTEGER NOT NULL,
                    playthrough INTEGER NOT NULL DEFAULT 1,
                    progress TEXT NOT NULL DEFAULT '',
                    character_build TEXT NOT NULL DEFAULT '',
                    equipment_json TEXT NOT NULL DEFAULT '[]',
                    difficulties_json TEXT NOT NULL DEFAULT '[]',
                    spoiler_level TEXT NOT NULL DEFAULT 'no_spoiler',
                    platform TEXT NOT NULL DEFAULT '',
                    game_version TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, appid, playthrough)
                );

                CREATE TABLE IF NOT EXISTS game_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    appid INTEGER NOT NULL,
                    playthrough INTEGER NOT NULL DEFAULT 1,
                    thread_id TEXT NOT NULL DEFAULT '',
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS preferences (
                    preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    value TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '',
                    scope TEXT NOT NULL DEFAULT 'persistent',
                    session_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE (user_id, kind, value, scope, session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_discovery_messages_session
                ON discovery_messages(session_id, message_id);

                CREATE INDEX IF NOT EXISTS idx_play_messages_thread
                ON play_messages(thread_id, message_id);

                CREATE INDEX IF NOT EXISTS idx_play_threads_game
                ON play_threads(user_id, appid, playthrough);
                """
            )

    # ------------------------------------------------------------------
    # 탐색 공간
    # ------------------------------------------------------------------
    def create_discovery_session(self, user_id: str, *, title: str = "") -> DiscoverySession:
        now = _utc_now()
        session_id = f"disc_{uuid.uuid4().hex[:16]}"
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO discovery_sessions "
                "(session_id, user_id, title, conditions_json, created_at, updated_at) "
                "VALUES (?, ?, ?, '{}', ?, ?)",
                (session_id, user_id, title, now, now),
            )
        return DiscoverySession(session_id, user_id, title, now, now, {})

    def get_discovery_session(self, user_id: str, session_id: str) -> DiscoverySession | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM discovery_sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            ).fetchone()
        return _discovery_session(row) if row else None

    def list_discovery_sessions(self, user_id: str, *, limit: int = 30) -> list[DiscoverySession]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM discovery_sessions WHERE user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, max(1, limit)),
            ).fetchall()
        return [_discovery_session(row) for row in rows]

    def update_discovery_conditions(
        self, user_id: str, session_id: str, conditions: dict[str, Any]
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE discovery_sessions SET conditions_json = ?, updated_at = ? "
                "WHERE user_id = ? AND session_id = ?",
                (json.dumps(conditions, ensure_ascii=False), _utc_now(), user_id, session_id),
            )

    def append_discovery_message(
        self,
        user_id: str,
        session_id: str,
        *,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO discovery_messages "
                "(session_id, user_id, role, content, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    user_id,
                    role,
                    content,
                    json.dumps(payload or {}, ensure_ascii=False),
                    now,
                ),
            )
            connection.execute(
                "UPDATE discovery_sessions SET updated_at = ?, title = "
                "CASE WHEN title = '' AND ? = 'user' THEN substr(?, 1, 60) ELSE title END "
                "WHERE session_id = ?",
                (now, role, content, session_id),
            )

    def recent_discovery_messages(
        self, user_id: str, session_id: str, *, limit: int = 12
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT role, content, payload_json, created_at FROM discovery_messages "
                "WHERE user_id = ? AND session_id = ? ORDER BY message_id DESC LIMIT ?",
                (user_id, session_id, max(1, limit)),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "payload": _json_dict(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]

    # ------------------------------------------------------------------
    # 내 게임
    # ------------------------------------------------------------------
    def add_library_game(
        self,
        user_id: str,
        *,
        appid: int,
        name: str,
        header_image: str = "",
        platform: str = "steam",
        source: str = "manual",
        note: str = "",
    ) -> dict[str, Any]:
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO library "
                "(user_id, appid, name, header_image, platform, source, note, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, appid) DO UPDATE SET "
                "name = excluded.name, header_image = excluded.header_image, "
                "platform = excluded.platform, note = excluded.note",
                (user_id, int(appid), name, header_image, platform, source, note, now),
            )
        return {
            "appid": int(appid),
            "name": name,
            "header_image": header_image,
            "platform": platform,
            "source": source,
            "note": note,
            "added_at": now,
        }

    def list_library(self, user_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM library WHERE user_id = ? ORDER BY added_at DESC",
                (user_id,),
            ).fetchall()
            states = connection.execute(
                "SELECT appid, playthrough, progress, spoiler_level, updated_at "
                "FROM game_states WHERE user_id = ? ORDER BY playthrough DESC",
                (user_id,),
            ).fetchall()
            threads = connection.execute(
                "SELECT appid, COUNT(*) AS thread_count FROM play_threads "
                "WHERE user_id = ? GROUP BY appid",
                (user_id,),
            ).fetchall()
        state_by_appid = {int(row["appid"]): row for row in states}
        thread_counts = {int(row["appid"]): int(row["thread_count"]) for row in threads}
        library: list[dict[str, Any]] = []
        for row in rows:
            appid = int(row["appid"])
            state = state_by_appid.get(appid)
            library.append(
                {
                    "appid": appid,
                    "name": row["name"],
                    "header_image": row["header_image"],
                    "platform": row["platform"],
                    "source": row["source"],
                    "note": row["note"],
                    "added_at": row["added_at"],
                    "progress": state["progress"] if state else "",
                    "spoiler_level": state["spoiler_level"] if state else DEFAULT_SPOILER_LEVEL,
                    "thread_count": thread_counts.get(appid, 0),
                }
            )
        return library

    def remove_library_game(self, user_id: str, appid: int) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM library WHERE user_id = ? AND appid = ?", (user_id, int(appid))
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # 게임별 플레이 공간
    # ------------------------------------------------------------------
    def open_play_thread(
        self,
        user_id: str,
        *,
        appid: int,
        topic: str = "general",
        title: str = "",
        playthrough: int = 1,
    ) -> PlayThread:
        now = _utc_now()
        thread_id = f"play_{uuid.uuid4().hex[:16]}"
        resolved_title = title or dict(DEFAULT_THREAD_TOPICS).get(topic, "새 공략 대화")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO play_threads "
                "(thread_id, user_id, appid, playthrough, topic, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    user_id,
                    int(appid),
                    max(1, int(playthrough)),
                    topic,
                    resolved_title,
                    now,
                    now,
                ),
            )
        return PlayThread(
            thread_id, user_id, int(appid), max(1, int(playthrough)), topic, resolved_title, now, now
        )

    def get_play_thread(self, user_id: str, thread_id: str) -> PlayThread | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM play_threads WHERE user_id = ? AND thread_id = ?",
                (user_id, thread_id),
            ).fetchone()
        return _play_thread(row) if row else None

    def list_play_threads(
        self, user_id: str, appid: int, *, playthrough: int | None = None
    ) -> list[PlayThread]:
        query = "SELECT * FROM play_threads WHERE user_id = ? AND appid = ?"
        params: list[Any] = [user_id, int(appid)]
        if playthrough is not None:
            query += " AND playthrough = ?"
            params.append(max(1, int(playthrough)))
        query += " ORDER BY updated_at DESC"
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_play_thread(row) for row in rows]

    def append_play_message(
        self,
        user_id: str,
        thread_id: str,
        *,
        appid: int,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO play_messages "
                "(thread_id, user_id, appid, role, content, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    user_id,
                    int(appid),
                    role,
                    content,
                    json.dumps(payload or {}, ensure_ascii=False),
                    now,
                ),
            )
            connection.execute(
                "UPDATE play_threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id)
            )

    def recent_play_messages(
        self, user_id: str, thread_id: str, *, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return only this thread's history.

        같은 게임의 다른 공략 대화도 자동으로 붙이지 않는다(§11).
        """

        with self._connection() as connection:
            rows = connection.execute(
                "SELECT role, content, payload_json, created_at FROM play_messages "
                "WHERE user_id = ? AND thread_id = ? ORDER BY message_id DESC LIMIT ?",
                (user_id, thread_id, max(1, limit)),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "payload": _json_dict(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]

    # ------------------------------------------------------------------
    # 게임 상태와 시도 기록
    # ------------------------------------------------------------------
    def get_game_state(self, user_id: str, appid: int, *, playthrough: int = 1) -> GameState:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM game_states WHERE user_id = ? AND appid = ? AND playthrough = ?",
                (user_id, int(appid), max(1, int(playthrough))),
            ).fetchone()
        if row is None:
            return GameState(user_id, int(appid), max(1, int(playthrough)))
        return GameState(
            user_id=row["user_id"],
            appid=int(row["appid"]),
            playthrough=int(row["playthrough"]),
            progress=row["progress"],
            character_build=row["character_build"],
            equipment=_json_list(row["equipment_json"]),
            difficulties=_json_list(row["difficulties_json"]),
            spoiler_level=row["spoiler_level"],
            platform=row["platform"],
            game_version=row["game_version"],
            updated_at=row["updated_at"],
        )

    def update_game_state(
        self,
        user_id: str,
        appid: int,
        *,
        playthrough: int = 1,
        progress: str | None = None,
        character_build: str | None = None,
        equipment: Sequence[str] | None = None,
        difficulties: Sequence[str] | None = None,
        spoiler_level: str | None = None,
        platform: str | None = None,
        game_version: str | None = None,
    ) -> GameState:
        """Merge a partial update into the stored state.

        이미지나 추정으로 얻은 값은 호출 전에 사용자 확인을 거친다(§11).
        """

        current = self.get_game_state(user_id, appid, playthrough=playthrough)
        level = spoiler_level if spoiler_level in SPOILER_LEVELS else current.spoiler_level
        merged = GameState(
            user_id=user_id,
            appid=int(appid),
            playthrough=max(1, int(playthrough)),
            progress=current.progress if progress is None else progress.strip(),
            character_build=(
                current.character_build if character_build is None else character_build.strip()
            ),
            equipment=list(current.equipment if equipment is None else equipment),
            difficulties=list(current.difficulties if difficulties is None else difficulties),
            spoiler_level=level,
            platform=current.platform if platform is None else platform.strip(),
            game_version=current.game_version if game_version is None else game_version.strip(),
            updated_at=_utc_now(),
        )
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO game_states (user_id, appid, playthrough, progress, character_build, "
                "equipment_json, difficulties_json, spoiler_level, platform, game_version, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, appid, playthrough) DO UPDATE SET "
                "progress = excluded.progress, character_build = excluded.character_build, "
                "equipment_json = excluded.equipment_json, "
                "difficulties_json = excluded.difficulties_json, "
                "spoiler_level = excluded.spoiler_level, platform = excluded.platform, "
                "game_version = excluded.game_version, updated_at = excluded.updated_at",
                (
                    merged.user_id,
                    merged.appid,
                    merged.playthrough,
                    merged.progress,
                    merged.character_build,
                    json.dumps(merged.equipment, ensure_ascii=False),
                    json.dumps(merged.difficulties, ensure_ascii=False),
                    merged.spoiler_level,
                    merged.platform,
                    merged.game_version,
                    merged.updated_at,
                ),
            )
        return merged

    def next_playthrough(self, user_id: str, appid: int) -> int:
        """Start a new run without overwriting the previous progress (§11)."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(playthrough), 0) AS latest FROM game_states "
                "WHERE user_id = ? AND appid = ?",
                (user_id, int(appid)),
            ).fetchone()
        return int(row["latest"] or 0) + 1

    def record_attempt(
        self,
        user_id: str,
        appid: int,
        *,
        action: str,
        outcome: str = "",
        playthrough: int = 1,
        thread_id: str = "",
    ) -> None:
        """§8-5 '결과를 반영한다'. 시도한 방법과 결과를 게임 상태에 남긴다."""

        cleaned = action.strip()
        if not cleaned:
            return
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO game_attempts "
                "(user_id, appid, playthrough, thread_id, action, outcome, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    int(appid),
                    max(1, int(playthrough)),
                    thread_id,
                    cleaned[:400],
                    outcome.strip()[:400],
                    _utc_now(),
                ),
            )

    def list_attempts(
        self, user_id: str, appid: int, *, playthrough: int = 1, limit: int = 8
    ) -> list[Attempt]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM game_attempts WHERE user_id = ? AND appid = ? AND playthrough = ? "
                "ORDER BY attempt_id DESC LIMIT ?",
                (user_id, int(appid), max(1, int(playthrough)), max(1, limit)),
            ).fetchall()
        return [
            Attempt(
                attempt_id=int(row["attempt_id"]),
                appid=int(row["appid"]),
                playthrough=int(row["playthrough"]),
                thread_id=row["thread_id"],
                action=row["action"],
                outcome=row["outcome"],
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]

    # ------------------------------------------------------------------
    # 내 취향
    # ------------------------------------------------------------------
    def set_preference(
        self,
        user_id: str,
        *,
        kind: str,
        value: str,
        label: str = "",
        evidence: str = "",
        scope: str = "persistent",
        session_id: str = "",
    ) -> Preference | None:
        """Store an explicitly stated preference together with its evidence (§11)."""

        if kind not in PREFERENCE_KINDS or not value.strip():
            return None
        resolved_scope = scope if scope in PREFERENCE_SCOPES else "persistent"
        if resolved_scope == "session" and not session_id:
            return None
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO preferences "
                "(user_id, kind, value, label, evidence, scope, session_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, kind, value, scope, session_id) DO UPDATE SET "
                "label = excluded.label, evidence = excluded.evidence",
                (
                    user_id,
                    kind,
                    value.strip(),
                    label.strip(),
                    evidence.strip(),
                    resolved_scope,
                    session_id,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM preferences WHERE user_id = ? AND kind = ? AND value = ? "
                "AND scope = ? AND session_id = ?",
                (user_id, kind, value.strip(), resolved_scope, session_id),
            ).fetchone()
        return _preference(row) if row else None

    def list_preferences(
        self, user_id: str, *, scope: str | None = None, session_id: str = ""
    ) -> list[Preference]:
        query = "SELECT * FROM preferences WHERE user_id = ?"
        params: list[Any] = [user_id]
        if scope == "persistent":
            query += " AND scope = 'persistent'"
        elif scope == "session":
            query += " AND scope = 'session' AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at DESC"
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_preference(row) for row in rows]

    def delete_preference(self, user_id: str, preference_id: int) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM preferences WHERE user_id = ? AND preference_id = ?",
                (user_id, int(preference_id)),
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # 컨텍스트 조립
    # ------------------------------------------------------------------
    def play_context(
        self,
        user_id: str,
        *,
        appid: int,
        thread_id: str,
        playthrough: int = 1,
        history_limit: int = 8,
    ) -> dict[str, Any]:
        """Assemble the context for one 공략 요청.

        §11: "각 공략 요청에는 해당 주제의 최근 대화, 확인된 게임 상태, 질문에
        필요한 자료만 전달한다." 탐색 대화와 다른 게임의 상태는 이 함수가
        접근하지 않으므로 프롬프트 실수로도 섞일 수 없다.
        """

        state = self.get_game_state(user_id, appid, playthrough=playthrough)
        return {
            "appid": int(appid),
            "playthrough": max(1, int(playthrough)),
            "thread_id": thread_id,
            "game_state": state.to_dict(),
            "messages": self.recent_play_messages(user_id, thread_id, limit=history_limit),
            "attempts": [item.to_dict() for item in self.list_attempts(user_id, appid, playthrough=playthrough)],
            "persistent_preferences": [
                item.to_dict() for item in self.list_preferences(user_id, scope="persistent")
            ],
        }

    def handoff_to_play_space(
        self,
        user_id: str,
        *,
        appid: int,
        name: str,
        header_image: str = "",
        platform: str = "steam",
    ) -> dict[str, Any]:
        """Move from 탐색 to 플레이 공간 carrying only the game and platform (§4.4).

        탐색 대화 전체, 다른 후보의 정보, 이번 구매 예산은 넘기지 않는다.
        """

        self.add_library_game(
            user_id,
            appid=appid,
            name=name,
            header_image=header_image,
            platform=platform,
            source="discovery_handoff",
        )
        threads = self.list_play_threads(user_id, appid)
        if not threads:
            threads = [
                self.open_play_thread(user_id, appid=appid, topic=topic, title=title)
                for topic, title in DEFAULT_THREAD_TOPICS[:1]
            ]
        return {
            "appid": int(appid),
            "name": name,
            "platform": platform,
            "threads": [thread.to_dict() for thread in threads],
            "game_state": self.get_game_state(user_id, appid).to_dict(),
        }


def _discovery_session(row: sqlite3.Row) -> DiscoverySession:
    return DiscoverySession(
        session_id=row["session_id"],
        user_id=row["user_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        conditions=_json_dict(row["conditions_json"]),
    )


def _play_thread(row: sqlite3.Row) -> PlayThread:
    return PlayThread(
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        appid=int(row["appid"]),
        playthrough=int(row["playthrough"]),
        topic=row["topic"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _preference(row: sqlite3.Row) -> Preference:
    return Preference(
        preference_id=int(row["preference_id"]),
        user_id=row["user_id"],
        kind=row["kind"],
        value=row["value"],
        label=row["label"],
        evidence=row["evidence"],
        scope=row["scope"],
        session_id=row["session_id"],
        created_at=row["created_at"],
    )


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def preference_terms(preferences: Iterable[Preference], kind: str) -> list[str]:
    """Return the stored values of one preference kind, newest first."""

    return [item.value for item in preferences if item.kind == kind]
