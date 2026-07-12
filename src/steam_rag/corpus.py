from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .index import VectorIndex, build_index, upsert_game_documents
from .markdown_corpus import chunk_documents, load_documents, parse_markdown, parse_metadata
from .interfaces import Embedder
from .steam import CollectionResult, SteamAPIClient, SteamGame, collect_game, save_catalog


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


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
        self._by_name: dict[str, SteamGame] = {}
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
            self._by_name.setdefault(_normalize_name(name), game)

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
        tokens = normalized.split()
        max_words = min(12, len(tokens))
        for size in range(max_words, 0, -1):
            for start in range(0, len(tokens) - size + 1):
                candidate = " ".join(tokens[start : start + size])
                game = self._by_name.get(candidate)
                if game:
                    return game
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
            raise LookupError(
                "질문에서 Steam 게임을 식별하지 못했습니다. `steam-rag sync-catalog`를 먼저 실행하거나 Steam URL/appid를 포함해 주세요."
            )
        return self.ensure_game(game, embedder, force=force)

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
            return CorpusUpdate(game, markdown_path, collected, indexed, reason)

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
