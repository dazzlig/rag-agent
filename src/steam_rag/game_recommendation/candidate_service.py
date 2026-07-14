from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

from steam_rag.game_analysis.time_aware import run_time_analysis_and_index
from steam_rag.game_recommendation.profile_builder import collect_recommendation_profile
from steam_rag.game_recommendation.profile_store import SteamProfileStore
from steam_rag.game_recommendation.query_parser import (
    RecommendationProfileIndex,
    RecommendationQuery,
    RecommendationSelection,
)
from steam_rag.steam_collection.corpus_manager import OnDemandCorpusManager
from steam_rag.steam_collection.steam_client import DEFAULT_COUNTRY, DEFAULT_LANGUAGE, SteamAPIClient, SteamGame


DISPLAY_TERMS = {
    "action_rpg": "Action RPG",
    "turn_based": "Turn-Based Combat",
    "real_time": "Real-Time Combat",
    "third_person": "Third Person",
    "first_person": "First Person",
    "top_down": "Top-Down",
    "isometric": "Isometric",
    "2d": "2D",
    "2_5d": "2.5D",
    "3d": "3D",
    "open_world": "Open World",
    "souls_like": "Souls-like",
    "roguelike": "Roguelike",
    "metroidvania": "Metroidvania",
    "survival": "Survival",
    "co_op": "Co-op",
    "story_rich": "Story Rich",
    "character_progression": "RPG",
}


@dataclass(slots=True)
class DynamicRecommendationRun:
    selection: RecommendationSelection
    initial_candidate_count: int
    discovery_terms: list[str] = field(default_factory=list)
    discovered_app_count: int = 0
    new_core_profiles: list[int] = field(default_factory=list)
    core_profile_failures: list[dict[str, Any]] = field(default_factory=list)
    detail_collected: list[int] = field(default_factory=list)
    temporal_analyzed: list[int] = field(default_factory=list)
    detail_failures: list[dict[str, Any]] = field(default_factory=list)
    store_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection": self.selection.to_dict(),
            "initial_candidate_count": self.initial_candidate_count,
            "discovery_terms": self.discovery_terms,
            "discovered_app_count": self.discovered_app_count,
            "new_core_profiles": self.new_core_profiles,
            "core_profile_failures": self.core_profile_failures,
            "detail_collected": self.detail_collected,
            "temporal_analyzed": self.temporal_analyzed,
            "detail_failures": self.detail_failures,
            "profile_store": self.store_summary,
        }


class DynamicRecommendationService:
    """Question-driven core-profile expansion followed by Top-N detail enrichment."""

    def __init__(
        self,
        *,
        client: SteamAPIClient,
        store: SteamProfileStore,
        profiles_dir: Path,
        language: str = DEFAULT_LANGUAGE,
        country: str = DEFAULT_COUNTRY,
    ) -> None:
        self.client = client
        self.store = store
        self.profiles_dir = profiles_dir
        self.language = language
        self.country = country

    def recommend(
        self,
        question: str,
        query: RecommendationQuery,
        *,
        min_candidates: int = 20,
        candidate_limit: int = 20,
        detail_limit: int = 5,
        expand_profiles: bool = True,
        max_new_profiles: int = 20,
        discovery_per_term: int = 30,
        enrich_details: bool = False,
        embedder: Any | None = None,
        catalog_path: Path | None = None,
        docs_dir: Path | None = None,
        raw_dir: Path | None = None,
        index_path: Path | None = None,
        time_analysis_dir: Path | None = None,
        allowed_appids: set[int] | None = None,
    ) -> DynamicRecommendationRun:
        self.store.import_profile_directory(self.profiles_dir)
        selection = self._search(
            question,
            query,
            candidate_limit=candidate_limit,
            detail_limit=detail_limit,
            allowed_appids=allowed_appids,
        )
        run = DynamicRecommendationRun(
            selection=selection,
            initial_candidate_count=selection.hard_filter_matches,
        )
        if expand_profiles and selection.hard_filter_matches < min_candidates:
            self._expand_core_profiles(
                query,
                run,
                max_new_profiles=max_new_profiles,
                discovery_per_term=discovery_per_term,
            )
            selection = self._search(
                question,
                query,
                candidate_limit=candidate_limit,
                detail_limit=detail_limit,
                allowed_appids=allowed_appids,
            )
            run.selection = selection

        if enrich_details:
            if embedder is None or not all(
                (catalog_path, docs_dir, raw_dir, index_path, time_analysis_dir)
            ):
                raise ValueError("detail enrichment requires embedder and all corpus paths")
            self._enrich_details(
                run,
                query,
                embedder=embedder,
                catalog_path=catalog_path,
                docs_dir=docs_dir,
                raw_dir=raw_dir,
                index_path=index_path,
                time_analysis_dir=time_analysis_dir,
            )
            run.selection = self._search(
                question,
                query,
                candidate_limit=candidate_limit,
                detail_limit=detail_limit,
                allowed_appids=allowed_appids,
            )
        run.store_summary = self.store.summary()
        return run

    def _search(
        self,
        question: str,
        query: RecommendationQuery,
        *,
        candidate_limit: int,
        detail_limit: int,
        allowed_appids: set[int] | None = None,
    ) -> RecommendationSelection:
        profiles = self.store.load_core_profiles(include_expired=True)
        return RecommendationProfileIndex(profiles).search(
            question,
            query,
            candidate_limit=candidate_limit,
            detail_limit=detail_limit,
            allowed_appids=allowed_appids,
        )

    def _expand_core_profiles(
        self,
        query: RecommendationQuery,
        run: DynamicRecommendationRun,
        *,
        max_new_profiles: int,
        discovery_per_term: int,
    ) -> None:
        terms = discovery_terms(query)
        run.discovery_terms = terms
        existing_appids = {
            int(profile.get("appid"))
            for _, profile in self.store.load_core_profiles(include_expired=True)
        }
        discovered_rows: dict[int, dict[str, Any]] = {}
        individual_terms = terms[1:] if len(terms) > 1 else terms
        combined_tag_rows: list[dict[str, Any]] = []
        if hasattr(self.client, "search_store_by_tags") and individual_terms:
            try:
                combined_tag_rows = self.client.search_store_by_tags(
                    individual_terms,
                    count=max(discovery_per_term, max_new_profiles),
                    language=self.language,
                    country=self.country,
                )
            except Exception:
                combined_tag_rows = []
            for item in combined_tag_rows:
                _merge_discovery_row(discovered_rows, item, existing_appids)

        # A combined tag search is the highest-quality discovery source. Tag
        # landing pages contain unrelated side modules, so use them only when
        # Steam cannot resolve or return the requested combination.
        if not combined_tag_rows and hasattr(self.client, "fetch_tag_candidates"):
            for term in individual_terms:
                try:
                    rows = self.client.fetch_tag_candidates(
                        term,
                        max_apps=discovery_per_term,
                        language=self.language,
                        country=self.country,
                    )
                except Exception:
                    rows = []
                for item in rows:
                    _merge_discovery_row(discovered_rows, item, existing_appids)
        if not discovered_rows:
            for term in terms[:2]:
                try:
                    rows = self.client.search_store(
                        term,
                        count=discovery_per_term,
                        language=self.language,
                        country=self.country,
                    )
                except Exception:
                    rows = []
                for item in rows:
                    _merge_discovery_row(discovered_rows, item, existing_appids)
        ordered_rows = sorted(
            discovered_rows.values(),
            key=lambda row: (-int(row["condition_hits"]), int(row["best_rank"]), int(row["appid"])),
        )[:max_new_profiles]
        discovered = {
            int(row["appid"]): SteamGame(int(row["appid"]), str(row["name"]))
            for row in ordered_rows
        }
        run.discovered_app_count = len(discovered)
        self.store.sync_registry(
            {"appid": game.appid, "name": game.name, "type": "game"}
            for game in discovered.values()
        )
        for game in discovered.values():
            self.store.enqueue(game.appid, priority=100)

        processed = 0
        while processed < max_new_profiles:
            job = self.store.claim_next()
            if job is None:
                break
            registry_game = self.store.registry_game(job.appid)
            if registry_game is None:
                self.store.mark_failed(job.job_id, "registry entry missing", status="permanent_failed")
                continue
            game = SteamGame(*registry_game)
            try:
                path = collect_recommendation_profile(
                    self.client,
                    game,
                    profiles_dir=self.profiles_dir,
                    language=self.language,
                    country=self.country,
                )
                profile = json.loads(path.read_text(encoding="utf-8"))
                self.store.upsert_core_profile(profile, profile_path=path)
                self.store.sync_registry(
                    [
                        {
                            "appid": game.appid,
                            "name": str(profile.get("name") or game.name),
                            "type": "game",
                        }
                    ]
                )
                self.store.mark_completed(job.job_id)
                run.new_core_profiles.append(game.appid)
            except LookupError as exc:
                self.store.mark_failed(
                    job.job_id,
                    str(exc),
                    status="store_unavailable",
                )
                run.core_profile_failures.append(
                    {"appid": game.appid, "status": "store_unavailable", "error": str(exc)}
                )
            except Exception as exc:
                self.store.mark_failed(
                    job.job_id,
                    f"{type(exc).__name__}: {exc}",
                    status="transient_failed",
                )
                run.core_profile_failures.append(
                    {"appid": game.appid, "status": "transient_failed", "error": str(exc)}
                )
            processed += 1

    def _enrich_details(
        self,
        run: DynamicRecommendationRun,
        query: RecommendationQuery,
        *,
        embedder: Any,
        catalog_path: Path,
        docs_dir: Path,
        raw_dir: Path,
        index_path: Path,
        time_analysis_dir: Path,
    ) -> None:
        manager = OnDemandCorpusManager(
            client=self.client,
            catalog_path=catalog_path,
            docs_dir=docs_dir,
            raw_dir=raw_dir,
            profiles_dir=self.profiles_dir,
            index_path=index_path,
            max_age=timedelta(hours=24),
        )
        for candidate in run.selection.detail_targets:
            game = SteamGame(candidate.appid, candidate.name)
            try:
                update = manager.ensure_game(game, embedder)
                run.detail_collected.append(candidate.appid)
                profile_path = self.profiles_dir / f"{update.game.resolved_key()}.json"
                if profile_path.exists():
                    self.store.upsert_core_profile(
                        json.loads(profile_path.read_text(encoding="utf-8")),
                        profile_path=profile_path,
                    )
                if query.after_update_required:
                    run_time_analysis_and_index(
                        client=self.client,
                        embedder=embedder,
                        game=update.game,
                        catalog_path=catalog_path,
                        docs_dir=docs_dir,
                        raw_dir=raw_dir,
                        profiles_dir=self.profiles_dir,
                        index_path=index_path,
                        output_dir=time_analysis_dir,
                        language=self.language,
                    )
                    run.temporal_analyzed.append(candidate.appid)
            except Exception as exc:
                run.detail_failures.append(
                    {"appid": candidate.appid, "error": f"{type(exc).__name__}: {exc}"}
                )


def discovery_terms(query: RecommendationQuery) -> list[str]:
    normalized = query.normalized()
    raw_terms = [
        *normalized.required_tags,
        *normalized.perspective,
        *normalized.dimension,
        *normalized.combat,
        *normalized.playstyle,
        *normalized.genres,
    ]
    display = [DISPLAY_TERMS.get(term, term.replace("_", " ").title()) for term in raw_terms]
    deduped: list[str] = []
    if display:
        combination = " ".join(display[:3])
        deduped.append(combination)
    for term in display:
        if term not in deduped:
            deduped.append(term)
    return deduped[:5] or ["Steam game"]


def _merge_discovery_row(
    target: dict[int, dict[str, Any]],
    item: dict[str, Any],
    existing_appids: set[int],
) -> None:
    try:
        appid = int(item["appid"])
    except (KeyError, TypeError, ValueError):
        return
    if appid in existing_appids:
        return
    name = str(item.get("name") or "").strip()
    if not name:
        return
    try:
        rank = int(item.get("tag_rank") or 100)
    except (TypeError, ValueError):
        rank = 100
    row = target.setdefault(
        appid,
        {
            "appid": appid,
            "name": name,
            "condition_hits": int(item.get("condition_hits") or 0),
            "best_rank": rank,
            "terms": set(),
        },
    )
    term = str(item.get("discovery_term") or item.get("source") or "unknown")
    if term not in row["terms"]:
        row["terms"].add(term)
        row["condition_hits"] = max(
            int(row["condition_hits"]), int(item.get("condition_hits") or 0)
        ) + (0 if item.get("condition_hits") else 1)
    row["best_rank"] = min(int(row["best_rank"]), rank)
