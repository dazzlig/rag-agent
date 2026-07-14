from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from steam_rag.common.interfaces import Embedder
from steam_rag.common.telemetry import current_telemetry
from steam_rag.rag_search.vector_store import VectorIndex, build_index, upsert_game_documents
from steam_rag.steam_collection.markdown_documents import chunk_documents, load_documents, parse_markdown, parse_metadata
from steam_rag.steam_collection.steam_client import CollectionResult, SteamAPIClient, SteamGame, collect_game, save_catalog


def _normalize_name(value: str) -> str:
    separated = re.sub(
        r"(?<=[a-z0-9])(?=[가-힣])|(?<=[가-힣])(?=[a-z0-9])",
        " ",
        value.casefold(),
    )
    return re.sub(r"[^\w]+", " ", separated, flags=re.UNICODE).strip()


_KOREAN_PARTICLES = (
    "으로는", "에서는", "에게는", "이라는", "이라고", "으로", "에서", "에게",
    "부터", "까지", "처럼", "보다", "과는", "와는", "이랑", "랑", "은", "는",
    "이", "가", "을", "를", "의", "에", "도", "만",
)

_GENERIC_NON_TITLE_TERMS = {
    "test", "testing", "ui", "ux", "테스트", "삭제", "화면", "질문", "메뉴",
}


def _strip_korean_particle(token: str) -> str:
    for particle in _KOREAN_PARTICLES:
        if token.endswith(particle) and len(token) > len(particle) + 1:
            return token[: -len(particle)]
    return token


def _store_search_terms(question: str) -> list[str]:
    """Extract conservative title candidates for a bounded Steam Store lookup."""

    terms: list[str] = []
    terms.extend(
        match.group(1).strip()
        for match in re.finditer(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]", question)
    )
    terms.extend(
        match.group(0).strip()
        for match in re.finditer(
            r"[A-Za-z0-9][A-Za-z0-9:'’&+.!_-]*(?:[ \t]+[A-Za-z0-9][A-Za-z0-9:'’&+.!_-]*){0,5}",
            question,
        )
    )
    for token in re.findall(r"[가-힣]{2,30}", question):
        stripped = _strip_korean_particle(token)
        if stripped != token:
            terms.append(stripped)

    ignored = {
        "steam", "appid", "top", "rag", "ragas", "게임", "추천", "최근", "평가",
        "친구", "친구들", "알려줘", "기준", "질문", "협동", "생존",
        *_GENERIC_NON_TITLE_TERMS,
    }
    unique: list[str] = []
    for term in terms:
        cleaned = term.strip(" \t\r\n.,!?()[]{}")
        normalized = _normalize_name(cleaned)
        if len(normalized) < 2 or normalized in ignored or normalized.isdigit():
            continue
        if cleaned not in unique:
            unique.append(cleaned)
    return unique


def _resolve_via_store_search(client: SteamAPIClient, question: str) -> SteamGame | None:
    """Resolve only an exact Store title match so generic queries cannot collect a wrong app."""

    normalized_question = _normalize_name(question)
    for term in _store_search_terms(question):
        normalized_term = _normalize_name(term)
        for candidate in client.search_store(term, count=10):
            name = str(candidate.get("name") or "").strip()
            try:
                appid = int(candidate.get("appid") or candidate.get("id"))
            except (TypeError, ValueError):
                continue
            normalized_name = _normalize_name(name)
            exact_term = normalized_name == normalized_term
            named_in_question = bool(
                normalized_name
                and re.search(rf"(?:^| ){re.escape(normalized_name)}(?: |$)", normalized_question)
            )
            if appid > 0 and name and (exact_term or named_in_question):
                return SteamGame(appid, name)
    return None


def explicit_appid_from_question(question: str) -> int | None:
    match = re.search(
        r"(?:store\.steampowered\.com/app/|\bappid\s*[:=#]?\s*)(\d{2,10})",
        question,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


@dataclass(frozen=True, slots=True)
class CorpusUpdate:
    game: SteamGame
    markdown_path: Path
    collected: bool
    indexed: bool
    reason: str


class SteamCatalog:
    def __init__(self, apps: list[dict]) -> None:
        self.apps = apps
        self._by_name: dict[str, list[SteamGame]] = {}
        self._by_appid: dict[int, SteamGame] = {}
        for app in apps:
            try:
                appid = int(app.get("appid") or app.get("id"))
            except (TypeError, ValueError):
                continue
            name = str(app.get("name") or "").strip()
            if not name:
                continue
            game = SteamGame(appid, name)
            self._by_appid[appid] = game
            self._by_name.setdefault(_normalize_name(name), []).append(game)

    @classmethod
    def load(cls, path: Path) -> "SteamCatalog":
        if not path.exists():
            raise FileNotFoundError(
                f"Steam catalog not found: {path}. Run `steam-rag sync-catalog` first."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        apps = payload.get("apps", payload)
        if not isinstance(apps, list):
            raise ValueError(f"Invalid Steam catalog: {path}")
        return cls([item for item in apps if isinstance(item, dict)])

    def resolve(self, question: str) -> SteamGame | None:
        explicit = explicit_appid_from_question(question)
        if explicit is not None:
            return self._by_appid.get(explicit) or SteamGame(explicit, f"Steam App {explicit}")

        normalized = _normalize_name(question)
        tokens = [
            token
            for item in normalized.split()
            for token in dict.fromkeys((item, _strip_korean_particle(item)))
        ]
        max_words = min(12, len(tokens))
        for size in range(max_words, 0, -1):
            for start in range(0, len(tokens) - size + 1):
                candidate = " ".join(tokens[start : start + size])
                games = self._by_name.get(candidate, [])
                if (
                    candidate in _GENERIC_NON_TITLE_TERMS
                    and normalized != candidate
                    and not re.search(rf"[\"'“”‘’]{re.escape(candidate)}[\"'“”‘’]", question.casefold())
                ):
                    continue
                if len(games) == 1:
                    return games[0]
                if games:
                    exact_case = [
                        game
                        for game in games
                        if re.search(
                            rf"(?<![A-Za-z0-9]){re.escape(game.name)}(?![A-Za-z0-9])",
                            question,
                        )
                    ]
                    if len(exact_case) == 1:
                        return exact_case[0]
                    return None
        return None


class OnDemandCorpusManager:
    """Resolve a queried game, collect missing/stale Markdown, and upsert its chunks."""

    def __init__(
        self,
        *,
        client: SteamAPIClient,
        catalog_path: Path,
        docs_dir: Path,
        raw_dir: Path,
        index_path: Path,
        profiles_dir: Path | None = None,
        max_age: timedelta = timedelta(hours=24),
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_reviews: int = 50,
        news_count: int = 20,
    ) -> None:
        self.client = client
        self.catalog_path = catalog_path
        self.docs_dir = docs_dir
        self.raw_dir = raw_dir
        self.profiles_dir = profiles_dir or docs_dir.parent / "game_profiles"
        self.index_path = index_path
        self.max_age = max_age
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_reviews = max_reviews
        self.news_count = news_count
        self._lock = threading.Lock()

    def ensure_question(self, question: str, embedder: Embedder, *, force: bool = False) -> CorpusUpdate:
        explicit = explicit_appid_from_question(question)
        if self.catalog_path.exists():
            game = SteamCatalog.load(self.catalog_path).resolve(question)
        elif explicit is None and os.getenv("STEAM_WEB_API_KEY"):
            apps = self.client.fetch_catalog(os.environ["STEAM_WEB_API_KEY"])
            save_catalog(self.catalog_path, apps)
            game = SteamCatalog(apps).resolve(question)
        else:
            game = SteamGame(explicit, f"Steam App {explicit}") if explicit is not None else None
        if game is None:
            game = _resolve_via_store_search(self.client, question)
        if game is None:
            raise LookupError(
                "질문에서 Steam 게임을 식별하지 못했습니다. `steam-rag sync-catalog`를 먼저 실행하거나 Steam URL/appid를 포함해 주세요."
            )
        return self.ensure_game(game, embedder, force=force)

    def ensure_questions(
        self,
        questions: list[str],
        embedder: Embedder,
        *,
        force: bool = False,
        strict: bool = False,
        max_games: int | None = None,
    ) -> list[CorpusUpdate]:
        """Collect/index every distinct game resolved from expanded queries.

        This is the comparison and alias-aware counterpart to
        :meth:`ensure_question`.  Query variants are discovery hints only;
        successful results are deduplicated by the canonical Steam AppID.
        """

        updates: list[CorpusUpdate] = []
        seen_appids: set[int] = set()
        errors: list[Exception] = []
        for question in questions:
            if not str(question).strip():
                continue
            try:
                update = self.ensure_question(str(question), embedder, force=force)
            except (LookupError, FileNotFoundError, RuntimeError, OSError) as exc:
                errors.append(exc)
                continue
            if update.game.appid in seen_appids:
                continue
            seen_appids.add(update.game.appid)
            updates.append(update)
            if max_games is not None and len(updates) >= max(1, max_games):
                break
        if strict and not updates and errors:
            raise errors[-1]
        return updates

    def ensure_game(self, game: SteamGame, embedder: Embedder, *, force: bool = False) -> CorpusUpdate:
        with self._lock:
            existing = self._find_markdown(game.appid)
            if existing:
                metadata = parse_metadata("\n".join(existing.read_text(encoding="utf-8").splitlines()[:40]))
                game = SteamGame(
                    game.appid,
                    str(metadata.get("name") or game.name),
                    str(metadata.get("game_key") or existing.stem),
                )
            fresh = existing is not None and self._is_fresh(existing)
            collected = False
            reason = "fresh"
            markdown_path = existing

            if force or not fresh:
                result: CollectionResult = collect_game(
                    self.client,
                    game,
                    docs_dir=self.docs_dir,
                    raw_dir=self.raw_dir,
                    profiles_dir=self.profiles_dir,
                    max_reviews=self.max_reviews,
                    news_count=self.news_count,
                )
                markdown_path = result.markdown_path
                game = result.game
                collected = True
                reason = "forced" if force else "missing" if existing is None else "stale"

            if markdown_path is None:
                raise RuntimeError(f"Markdown collection did not produce a file for appid={game.appid}")

            indexed = self._ensure_index(markdown_path, game, embedder, force=collected)
            update = CorpusUpdate(game, markdown_path, collected, indexed, reason)
            collector = current_telemetry()
            if collector is not None:
                collector.record_corpus_update(
                    appid=game.appid,
                    name=game.name,
                    collected=collected,
                    indexed=indexed,
                    reason=reason,
                )
            return update

    def _find_markdown(self, appid: int) -> Path | None:
        for path in self.docs_dir.glob("*.md"):
            head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:40])
            metadata = parse_metadata(head)
            if str(metadata.get("appid")) == str(appid):
                return path
        return None

    def _is_fresh(self, path: Path) -> bool:
        metadata = parse_metadata("\n".join(path.read_text(encoding="utf-8").splitlines()[:40]))
        collected_at = metadata.get("collected_at")
        if collected_at:
            try:
                timestamp = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                return datetime.now(timezone.utc) - timestamp <= self.max_age
            except ValueError:
                pass
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        return datetime.now(timezone.utc) - modified <= self.max_age

    def _ensure_index(
        self,
        markdown_path: Path,
        game: SteamGame,
        embedder: Embedder,
        *,
        force: bool,
    ) -> bool:
        if not self.index_path.exists():
            documents = load_documents(
                self.docs_dir,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            index = build_index(documents, embedder)
            index.save(self.index_path)
            return True

        index = VectorIndex.load(self.index_path)
        already_indexed = any(
            str(document.metadata.get("appid")) == str(game.appid)
            for document in index.documents
        )
        if already_indexed and not force:
            return False

        base = parse_markdown(markdown_path)
        documents = chunk_documents(
            base,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        updated = upsert_game_documents(index, documents, embedder, appid=game.appid)
        updated.save(self.index_path)
        return True
