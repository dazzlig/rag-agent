from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from steam_rag.agents.multi_agent_workflow import QueryExpansionAgent
from steam_rag.application.rag_pipeline import RAGPipeline
from steam_rag.common.telemetry import telemetry_session
from steam_rag.external_apis.openai_client import OpenAIAnswerGenerator, OpenAIEmbedder, load_env_file
from steam_rag.game_recommendation.candidate_service import DynamicRecommendationService
from steam_rag.game_recommendation.profile_builder import collect_recommendation_profile
from steam_rag.game_recommendation.profile_store import SteamProfileStore
from steam_rag.game_recommendation.query_parser import OpenAIRecommendationQueryStructurer, parse_recommendation_query
from steam_rag.game_recommendation.similarity_ranker import (
    ReferenceGame,
    SimilarityScore,
    SimilaritySpec,
    adapt_similarity_spec_to_question,
    build_similarity_spec,
    describe_similarity_spec,
    rank_similar_profiles,
    resolve_reference_game,
)
from steam_rag.rag_search.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    Reranker,
)
from steam_rag.rag_search.vector_store import VectorIndex
from steam_rag.steam_collection.corpus_manager import OnDemandCorpusManager, CorpusUpdate, explicit_appid_from_question
from steam_rag.steam_collection.steam_client import SteamAPIClient, SteamGame


@dataclass(frozen=True, slots=True)
class ServicePaths:
    docs_dir: Path = Path("data/docs_timeaware_playstyle")
    index_path: Path = Path("data/chroma/steam_rag_timeaware_playstyle")
    raw_dir: Path = Path("data/raw/on_demand")
    catalog_path: Path = Path("data/steam_catalog.json")
    profiles_dir: Path = Path("data/game_profiles")
    service_db: Path = Path("data/steam_service.db")
    time_analysis_dir: Path = Path("data/time_analysis")


class SteamServiceRuntime:
    """Production-shaped prototype runtime shared by the consumer website."""

    def __init__(
        self,
        *,
        paths: ServicePaths | None = None,
        embedding_model: str = "text-embedding-3-small",
        answer_model: str = "gpt-5-mini",
        enable_reranker: bool | None = None,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.paths = paths or ServicePaths()
        self.embedding_model = embedding_model
        self.answer_model = answer_model
        self.enable_reranker = enable_reranker
        self.reranker_model = reranker_model
        self._reranker: Reranker | None = None
        self._reranker_lock = threading.Lock()

    def _get_reranker(self) -> Reranker | None:
        """Return one lazy cross-encoder per service runtime.

        Constructing ``CrossEncoderReranker`` does not load or download the
        model.  The expensive model is loaded only on the first real rerank.
        Tests and low-resource prototype runs can opt out with the constructor
        flag or ``STEAM_RAG_ENABLE_RERANKER=0``.
        """

        enabled = self.enable_reranker
        if enabled is None:
            enabled = _env_flag("STEAM_RAG_ENABLE_RERANKER", default=True)
        if not enabled:
            return None
        if self._reranker is None:
            with self._reranker_lock:
                if self._reranker is None:
                    model_name = os.getenv(
                        "STEAM_RAG_RERANKER_MODEL", self.reranker_model
                    ).strip()
                    self._reranker = CrossEncoderReranker(model_name or DEFAULT_RERANKER_MODEL)
        return self._reranker

    def ask(
        self,
        question: str,
        *,
        top_k: int = 6,
        history: list[dict[str, str]] | None = None,
        context_games: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with telemetry_session() as telemetry:
            payload = self._ask_impl(
                question,
                top_k=top_k,
                history=history,
                context_games=context_games,
            )
            payload["telemetry"] = telemetry.snapshot()
            return payload

    def _ask_impl(
        self,
        question: str,
        *,
        top_k: int = 6,
        history: list[dict[str, str]] | None = None,
        context_games: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        question = str(question or "").strip()
        if not question:
            raise ValueError("질문을 입력해 주세요.")
        load_env_file()
        resolved_question = question
        context_used = False
        followup_relation = _followup_relation(question, history or [], context_games or [])
        rewrite_context_games = [] if followup_relation == "correction" else (context_games or [])
        if history or context_games:
            try:
                resolved_question = OpenAIAnswerGenerator(self.answer_model).rewrite_followup_question(
                    question,
                    history or [],
                    context_games=rewrite_context_games,
                )
                context_used = resolved_question.strip() != question
            except Exception:
                if rewrite_context_games and len(rewrite_context_games) == 1:
                    game = rewrite_context_games[0]
                    resolved_question = (
                        f"{question} (대상 게임: {game.get('name')}, appid: {game.get('appid')})"
                    )
                    context_used = True
                else:
                    resolved_question = question
        excluded_appids = (
            _context_appids(context_games or []) if followup_relation == "correction" else set()
        )
        if _is_broad_recommendation(resolved_question):
            payload = self._recommend(resolved_question, excluded_appids=excluded_appids)
        else:
            payload = self._research(
                resolved_question,
                top_k=max(1, min(int(top_k), 10)),
            )
        payload["resolved_question"] = resolved_question
        payload["conversation_context_used"] = context_used
        payload["followup_relation"] = followup_relation
        payload["excluded_appids"] = sorted(excluded_appids)
        return payload

    def health(self) -> dict[str, Any]:
        index_exists = self.paths.index_path.exists()
        chunks = 0
        if index_exists:
            try:
                chunks = len(VectorIndex.load(self.paths.index_path).documents)
            except Exception:
                chunks = 0
        return {
            "status": "ready" if index_exists else "needs_index",
            "index_exists": index_exists,
            "chunks": chunks,
            "documents": len(list(self.paths.docs_dir.glob("*.md"))) if self.paths.docs_dir.exists() else 0,
            "workflow": "LangGraph Multi-Agent",
        }

    def _research(self, question: str, *, top_k: int) -> dict[str, Any]:
        embedder = OpenAIEmbedder(self.embedding_model)
        generator = OpenAIAnswerGenerator(self.answer_model)
        variants = QueryExpansionAgent(generator, max_variants=4).expand(question)
        manager = OnDemandCorpusManager(
            client=SteamAPIClient(),
            catalog_path=self.paths.catalog_path,
            docs_dir=self.paths.docs_dir,
            raw_dir=self.paths.raw_dir,
            profiles_dir=self.paths.profiles_dir,
            index_path=self.paths.index_path,
            max_age=timedelta(hours=24),
        )
        expected_game_count = _expected_game_count(question)
        updates = manager.ensure_questions(
            variants,
            embedder,
            strict=False,
            max_games=expected_game_count,
        )
        if not self.paths.index_path.exists():
            raise FileNotFoundError(
                "검색할 벡터 인덱스가 없습니다. 게임 이름을 더 정확히 입력하거나 먼저 문서를 수집해 주세요."
            )
        pipeline = RAGPipeline.from_path(
            self.paths.index_path,
            embedder,
            generator,
            reranker=self._get_reranker(),
            rerank_candidates=24,
        )
        target_appids = list(dict.fromkeys(update.game.appid for update in updates))
        explicit_appid = explicit_appid_from_question(question)
        if explicit_appid is not None and explicit_appid not in target_appids:
            target_appids.insert(0, explicit_appid)
        if len(target_appids) < expected_game_count:
            indexed_targets = _index_appids_for_variants(
                pipeline,
                variants,
                limit=expected_game_count,
            )
            target_appids.extend(
                appid for appid in indexed_targets if appid not in target_appids
            )
        indexed_appids = {
            int(document.metadata["appid"])
            for document in pipeline.index.documents
            if str(document.metadata.get("appid") or "").isdigit()
        }
        target_appids = [appid for appid in target_appids if appid in indexed_appids]
        if expected_game_count > 1 and len(target_appids) < expected_game_count:
            resolved_names = ", ".join(
                update.game.name for update in updates if update.game.appid in target_appids
            ) or "없음"
            raise LookupError(
                "비교 질문은 모든 게임의 문서와 인덱스가 준비되어야 합니다. "
                f"필요한 게임 {expected_game_count}개 중 {len(target_appids)}개만 확인했습니다"
                f"(확인된 게임: {resolved_names}). 누락된 비교 대상의 정식명, 영문명, "
                "Steam URL 또는 appid를 포함해 다시 질문해 주세요. 부분 비교는 실행하지 않았습니다."
            )
        if not target_appids:
            raise LookupError(
                "질문에서 대상 Steam 게임을 확정하지 못해 다른 게임 문서를 섞지 않고 검색을 중단했습니다. "
                "게임의 정식명, 영문명, Steam URL 또는 appid를 함께 입력해 주세요."
            )
        result = pipeline.ask_multi_agent(
            question,
            k=top_k,
            max_steps=2,
            use_hyde=_should_use_hyde(question),
            query_variants=variants,
            allowed_appids=target_appids,
        )
        trace = [
            {
                "agent": "Entity Resolution Agent",
                "status": "completed" if updates else "no_new_document",
                "detail": _updates_detail(updates),
            },
            *list(result.metadata.get("agent_trace", [])),
        ]
        games = [_game_from_update(update) for update in updates]
        known_appids = {int(game["appid"]) for game in games if game.get("appid")}
        for source in result.sources:
            metadata = source.document.metadata
            try:
                appid = int(metadata.get("appid"))
            except (TypeError, ValueError):
                continue
            if appid in known_appids:
                continue
            if updates:
                continue
            known_appids.add(appid)
            games.append(
                {
                    "appid": appid,
                    "name": metadata.get("game_name") or metadata.get("game_key") or f"Steam App {appid}",
                    "image": _steam_header_image(appid),
                    "url": f"https://store.steampowered.com/app/{appid}/?l=koreana&cc=kr",
                    "status": "기존 인덱스 근거 사용",
                }
            )
        return {
            "mode": "research",
            "answer": result.answer,
            "query_variants": result.metadata.get("query_variants", variants),
            "agents": trace,
            "games": games,
            "sources": [_source_payload(source) for source in result.sources],
            "evidence_coverage": result.metadata.get("evidence_coverage", {}),
            "corpus_updates": [_update_payload(update) for update in updates],
        }

    def _recommend(
        self,
        question: str,
        *,
        excluded_appids: set[int] | None = None,
    ) -> dict[str, Any]:
        generator = OpenAIAnswerGenerator(self.answer_model)
        client = SteamAPIClient()
        query = OpenAIRecommendationQueryStructurer(self.answer_model).structure(question)
        excluded_appids = set(excluded_appids or ())
        concept_recommendation = _is_concept_recommendation(question)
        store = SteamProfileStore(self.paths.service_db)
        store.import_profile_directory(self.paths.profiles_dir)
        if store.summary()["registry_count"] == 0 and self.paths.catalog_path.exists():
            store.sync_catalog_file(self.paths.catalog_path)

        reference: ReferenceGame | None = None
        similarity_spec: SimilaritySpec | None = None
        reference_payload: dict[str, Any] = {}
        similarity_scores: list[SimilarityScore] = []
        if concept_recommendation:
            reference, seed_profile, reference_payload = self._resolve_similarity_reference(
                question,
                generator=generator,
                client=client,
                store=store,
            )
            if reference is not None and seed_profile is not None:
                similarity_spec = adapt_similarity_spec_to_question(
                    build_similarity_spec(seed_profile, reference=reference),
                    question,
                )
                similarity_scores = [
                    item
                    for item in rank_similar_profiles(
                        store.load_core_profiles(include_expired=True),
                        similarity_spec,
                        limit=20,
                    )
                    if item.appid not in excluded_appids
                ]

        use_web_discovery = _should_use_web_discovery(question, query)
        discovery: dict[str, Any] = {}
        discovery_error = ""
        # Official local profiles are both cheaper and more reliable than a
        # broad web search. Tavily is a bounded recall fallback only when the
        # local pool is too small or has fewer than three strong matches.
        strong_local_matches = sum(item.score >= 65.0 for item in similarity_scores)
        web_discovery_needed = use_web_discovery and (
            similarity_spec is None
            or len(similarity_scores) < 5
            or strong_local_matches < 3
        )
        if web_discovery_needed:
            try:
                discovery = generator.discover_game_candidates(
                    question,
                    limit=10,
                    reference_game=reference_payload,
                    similarity_spec=(
                        similarity_spec.to_dict()
                        if similarity_spec is not None
                        else {"search_terms": reference_payload.get("similarity_terms", [])}
                    ),
                )
            except Exception as exc:
                discovery_error = f"{type(exc).__name__}: {exc}"

        verified = self._verify_discovered_candidates(client, store, discovery)
        similarity_info: dict[int, dict[str, Any]] = {}
        allowed_appids: set[int] | None = None
        if similarity_spec is not None:
            similarity_scores = [
                item
                for item in rank_similar_profiles(
                    store.load_core_profiles(include_expired=True),
                    similarity_spec,
                    limit=20,
                )
                if item.appid not in excluded_appids
            ]
            allowed_appids = {item.appid for item in similarity_scores}
            similarity_info = {
                item.appid: {
                    "candidate_name": item.name,
                    "reason": (
                        f"{similarity_spec.seed.name}와 공통 요소: "
                        f"{', '.join(item.matched_aspects[:4])}."
                    ),
                    "similarity_score": item.score,
                    "matched_aspects": list(item.matched_aspects),
                }
                for item in similarity_scores
            }
            discovery["concept_summary"] = describe_similarity_spec(similarity_spec)
        scoped_discovery = bool(
            similarity_spec is not None
            or _requires_verified_discovery_scope(question, query)
        )
        if concept_recommendation:
            # 웹 조사 Agent가 의미 기반 후보를 이미 좁혔다. 여기서 LLM이 만든
            # 비공식 태그를 hard filter로 다시 쓰면 적절한 후보까지 0건이 될 수 있다.
            query = parse_recommendation_query(question)
        if allowed_appids is None and scoped_discovery:
            allowed_appids = set(verified) - excluded_appids
        needs_detail = bool(query.recent_rating_required or query.after_update_required)
        run = DynamicRecommendationService(
            client=client,
            store=store,
            profiles_dir=self.paths.profiles_dir,
        ).recommend(
            question,
            query,
            min_candidates=12,
            candidate_limit=12,
            detail_limit=5,
            expand_profiles=not scoped_discovery,
            max_new_profiles=10,
            enrich_details=needs_detail,
            embedder=OpenAIEmbedder(self.embedding_model) if needs_detail else None,
            catalog_path=self.paths.catalog_path,
            docs_dir=self.paths.docs_dir,
            raw_dir=self.paths.raw_dir,
            index_path=self.paths.index_path,
            time_analysis_dir=self.paths.time_analysis_dir,
            allowed_appids=allowed_appids if scoped_discovery else None,
        )
        ranked_candidates = list(run.selection.candidates)
        if similarity_spec is not None:
            score_order = {item.appid: item.score for item in similarity_scores}
            ranked_candidates.sort(key=lambda item: -score_order.get(item.appid, -1.0))
        elif concept_recommendation:
            discovery_order = {appid: rank for rank, appid in enumerate(verified)}
            ranked_candidates.sort(key=lambda item: discovery_order.get(item.appid, len(discovery_order)))
        candidates = [
            _candidate_payload(
                item,
                discovery_info=(similarity_info.get(item.appid) or verified.get(item.appid)),
            )
            for item in ranked_candidates[:5]
        ]
        answer = _recommendation_markdown(question, candidates, discovery, query)
        if not candidates and web_discovery_needed and "TAVILY_API_KEY" in discovery_error:
            answer = (
                "이 질문은 Steam 태그만으로 후보를 확정하기 어려워 Tavily 웹 후보 검색이 필요합니다. "
                "`.env`에 `TAVILY_API_KEY`를 설정한 뒤 다시 질문해 주세요."
            )
        agents = [
            {
                "agent": "Intent Router Agent",
                "status": "completed",
                "detail": "broad_recommendation",
            },
            {
                "agent": "Reference Game Grounding Agent",
                "status": "completed" if reference_payload else ("not_required" if not concept_recommendation else "unresolved"),
                "detail": (
                    f"seed={reference.name}, appid={reference.appid}, source={reference.source}"
                    if reference is not None
                    else str(reference_payload.get("canonical_name") or "no_seed")
                ),
            },
            {
                "agent": "Similarity Spec Agent",
                "status": "completed" if similarity_spec is not None else ("not_required" if not concept_recommendation else "fallback"),
                "detail": (
                    f"must={list(similarity_spec.must_have)}, candidates={len(similarity_scores)}"
                    if similarity_spec is not None
                    else "일반 추천 조건 사용"
                ),
            },
            {
                "agent": "Candidate Discovery Agent",
                "status": "completed" if (similarity_scores or discovery) else ("fallback" if use_web_discovery else "skipped"),
                "detail": (
                    f"web_candidates={len(discovery.get('candidates', []))}, "
                    f"steam_verified={len(verified)}, local_similar={len(similarity_scores)}, "
                    f"scoped={scoped_discovery}, provider={'tavily' if web_discovery_needed else 'steam_profiles'}, "
                    f"credits={discovery.get('search_credits', 0)}, cache={discovery.get('search_cache_hit', False)}"
                    if use_web_discovery
                    else "Steam 공식 프로필과 인기 태그만 사용"
                ),
            },
            {
                "agent": "Recommendation Research Agent",
                "status": "completed",
                "detail": f"ranked={len(run.selection.candidates)}, profiles={run.selection.scanned_profiles}",
            },
            {"agent": "Answer Agent", "status": "completed", "detail": f"games={len(candidates)}"},
        ]
        return {
            "mode": "recommendation",
            "answer": answer,
            "query_variants": [],
            "agents": agents,
            "games": candidates,
            "sources": [
                {"title": "후보 발굴 참고 자료", "url": url, "section": "web_discovery"}
                for url in discovery.get("source_urls", [])[:5]
                if str(url).startswith("http")
            ],
            "evidence_coverage": {},
            "discovery_error": discovery_error,
            "recommendation": {
                **run.to_dict(),
                "reference_game": reference_payload,
                "similarity_spec": similarity_spec.to_dict() if similarity_spec else {},
                "excluded_appids": sorted(excluded_appids),
            },
        }

    def _resolve_similarity_reference(
        self,
        question: str,
        *,
        generator: OpenAIAnswerGenerator,
        client: SteamAPIClient,
        store: SteamProfileStore,
    ) -> tuple[ReferenceGame | None, dict[str, Any] | None, dict[str, Any]]:
        profiles = store.load_core_profiles(include_expired=True)
        reference = resolve_reference_game(question, profiles)
        if reference is not None:
            seed = _profile_for_appid(profiles, reference.appid)
            payload = {**reference.to_dict(), "canonical_name": reference.name}
            return reference, seed, payload

        try:
            grounded = generator.ground_reference_game(question)
        except Exception:
            grounded = {}
        if not grounded.get("is_similarity_request"):
            return None, None, grounded
        canonical_name = str(grounded.get("canonical_name") or "").strip()
        if not canonical_name:
            return None, None, grounded

        # The LLM resolves language/alias only. Steam still verifies identity.
        local_reference = resolve_reference_game(f"{canonical_name} 같은 게임", profiles)
        game = (
            SteamGame(local_reference.appid, local_reference.name)
            if local_reference is not None
            else _resolve_candidate(client, canonical_name)
        )
        if game is None:
            return None, None, grounded
        seed = _profile_for_appid(profiles, game.appid)
        if seed is None:
            try:
                path = collect_recommendation_profile(
                    client,
                    game,
                    profiles_dir=self.paths.profiles_dir,
                )
                seed = json.loads(path.read_text(encoding="utf-8"))
                store.upsert_core_profile(seed, profile_path=path)
                store.sync_registry([{"appid": game.appid, "name": game.name, "type": "game"}])
            except Exception:
                return None, None, grounded
        reference = ReferenceGame(
            game.appid,
            str(seed.get("name") or game.name),
            str(grounded.get("reference_phrase") or canonical_name),
            "llm_alias_steam_verified",
            0.9,
        )
        return reference, seed, {**grounded, **reference.to_dict(), "canonical_name": reference.name}

    def _verify_discovered_candidates(
        self,
        client: SteamAPIClient,
        store: SteamProfileStore,
        discovery: dict[str, Any],
    ) -> dict[int, dict[str, Any]]:
        verified: dict[int, dict[str, Any]] = {}
        for item in discovery.get("candidates", [])[:8]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            game = _resolve_candidate(client, name)
            if game is None:
                continue
            try:
                path = collect_recommendation_profile(
                    client,
                    game,
                    profiles_dir=self.paths.profiles_dir,
                )
                profile = json.loads(path.read_text(encoding="utf-8"))
                store.upsert_core_profile(profile, profile_path=path)
                store.sync_registry([{"appid": game.appid, "name": game.name, "type": "game"}])
            except Exception:
                continue
            verified[game.appid] = {
                "candidate_name": name,
                "reason": str(item.get("reason") or "").strip(),
            }
        return verified


def _context_appids(context_games: list[dict[str, Any]]) -> set[int]:
    appids: set[int] = set()
    for game in context_games:
        try:
            appid = int(game.get("appid"))
        except (TypeError, ValueError):
            continue
        if appid > 0:
            appids.add(appid)
    return appids


def _followup_relation(
    question: str,
    history: list[dict[str, str]],
    context_games: list[dict[str, Any]],
) -> str:
    """Classify feedback before rewriting so rejected proposals never become constraints."""

    if not history and not context_games:
        return "standalone"
    if re.search(
        r"(?:^|\s)(?:아니|아니야|아니고)|그런\s*(?:게임|추천).*말고|"
        r"(?:내가\s*)?(?:원한|원하는|원했던|원해)|추천(?:한|된)?\s*게임.*(?:다르|이상)",
        question.casefold(),
    ):
        return "correction"
    return "continuation"


def _is_broad_recommendation(question: str) -> bool:
    lowered = question.casefold()
    return bool(
        re.search(
            r"게임\s*추천|추천\s*게임|추천해\s*줘|추천해줘|추천해\s*주세요|"
            r"게임(?:들)?(?:을|를)?\s*원해|같은\s*(?:게임|작품)|"
            r"현재\s*(?:세일|할인)|세일\s*중|할인\s*중|앞으로\s*나올|출시\s*예정|"
            r"신작.*(?:요약|추천)|기대작",
            lowered,
        )
    )


def _expected_game_count(question: str) -> int:
    comparison = re.search(
        r"비교|\bvs\.?\b|(?:와|과|랑|이랑).*(?:중|차이|어느|뭐가)",
        question,
        flags=re.IGNORECASE,
    )
    return 2 if comparison else 1


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _requires_verified_discovery_scope(question: str, query: Any) -> bool:
    return bool(
        getattr(query, "upcoming_required", False)
        or re.search(r"같은\s*(?:게임|작품)|서브컬처|기대작|앞으로\s*나올|출시\s*예정|신작", question.casefold())
    )


def _is_concept_recommendation(question: str) -> bool:
    return bool(re.search(r"같은\s*(?:게임|작품)|서브컬처", question.casefold()))


def _should_use_web_discovery(question: str, query: Any) -> bool:
    """Spend a web-search credit only when Steam taxonomy cannot discover candidates."""

    return bool(
        getattr(query, "upcoming_required", False)
        or re.search(r"같은\s*(?:게임|작품)|서브컬처|기대작|앞으로\s*나올|출시\s*예정|신작", question.casefold())
    )


def _should_use_hyde(question: str) -> bool:
    """Reserve the extra LLM call for genuinely complex investigations."""

    return len(question) >= 90 or bool(
        re.search(r"비교|원인|왜|패치\s*(?:전후|이후)|업데이트\s*(?:전후|이후)|변화|근거를\s*종합", question)
    )


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", value.casefold()).strip()


def _resolve_candidate(client: SteamAPIClient, name: str) -> SteamGame | None:
    if not name:
        return None
    target = _normalize(name)
    if target.replace(" ", "") in {"game", "games", "steam", "게임"}:
        return None
    rows = client.search_store(name, count=8)
    scored: list[tuple[float, SteamGame]] = []
    for row in rows:
        try:
            appid = int(row.get("appid") or row.get("id"))
        except (TypeError, ValueError):
            continue
        candidate_name = str(row.get("name") or "").strip()
        if not candidate_name:
            continue
        score = SequenceMatcher(None, target, _normalize(candidate_name)).ratio()
        exact = _normalize(candidate_name) == target
        scored.append((1.0 if exact else score, SteamGame(appid, candidate_name)))
    if not scored:
        return None
    score, game = max(scored, key=lambda item: item[0])
    if score < 0.9:
        return None
    try:
        app = client.fetch_app_details(game.appid, language="koreana", country="KR")
    except Exception:
        return None
    if str(app.get("type") or "").casefold() != "game":
        return None
    canonical_name = str(app.get("name") or game.name).strip()
    if SequenceMatcher(None, target, _normalize(canonical_name)).ratio() < 0.88:
        return None
    return SteamGame(game.appid, canonical_name)


def _index_appids_for_variants(
    pipeline: RAGPipeline,
    variants: list[str],
    *,
    limit: int,
) -> list[int]:
    """Resolve aliases against existing Chroma metadata without a global search fallback."""

    game_keys: list[str] = []
    for variant in variants:
        game_keys.extend(pipeline.retriever._detect_game_keys(variant))  # noqa: SLF001
    selected_keys = list(dict.fromkeys(game_keys))[: max(1, limit)]
    appids: list[int] = []
    for key in selected_keys:
        for document in pipeline.index.documents:
            if str(document.metadata.get("game_key") or "") != key:
                continue
            try:
                appid = int(document.metadata.get("appid"))
            except (TypeError, ValueError):
                continue
            if appid not in appids:
                appids.append(appid)
            break
    if appids:
        return appids

    # LLM title expansion often returns a short alias such as "33 원정대" or
    # "클레르 옵스퀴르" while the indexed title is the full localized name.
    # Accept only a clean alias contained in that title; never use semantic
    # similarity here because a false positive would reintroduce cross-game evidence.
    candidates: list[tuple[int, int]] = []
    seen_games: set[tuple[int, str]] = set()
    for document in pipeline.index.documents:
        metadata = document.metadata
        try:
            appid = int(metadata.get("appid"))
        except (TypeError, ValueError):
            continue
        name = _normalize(str(metadata.get("game_name") or ""))
        if not name or (appid, name) in seen_games:
            continue
        seen_games.add((appid, name))
        for variant in variants:
            alias = _normalize(variant)
            alias_tokens = _title_tokens(alias)
            name_tokens = _title_tokens(name)
            shared = set(alias_tokens) & set(name_tokens)
            clean_subtitle = 4 <= len(alias) and len(alias.split()) <= 6 and alias in name
            token_match = len(shared) >= 2 and (
                any(token.isdigit() for token in shared)
                or len(shared) / max(len(set(name_tokens)), 1) >= 0.6
            )
            if clean_subtitle or token_match:
                candidates.append((len(alias) if clean_subtitle else 10 * len(shared), appid))
                break
    for _, appid in sorted(candidates, reverse=True):
        if appid not in appids:
            appids.append(appid)
        if len(appids) >= max(1, limit):
            break
    return appids


def _title_tokens(value: str) -> list[str]:
    particles = ("으로는", "에서는", "이라는", "이라고", "으로", "에서", "처럼", "보다", "은", "는", "이", "가", "을", "를", "의", "에", "도", "만")
    tokens: list[str] = []
    for token in _normalize(value).split():
        cleaned = token
        for particle in particles:
            if cleaned.endswith(particle) and len(cleaned) > len(particle) + 1:
                cleaned = cleaned[: -len(particle)]
                break
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _source_payload(source: Any) -> dict[str, Any]:
    metadata = source.document.metadata
    content = re.sub(r"\s+", " ", source.document.page_content).strip()
    return {
        "rank": source.rank,
        "game": metadata.get("game_name") or metadata.get("game_key") or "Steam 게임",
        "appid": metadata.get("appid"),
        "section": metadata.get("section") or "document",
        "source_type": metadata.get("source_type") or "steam_corpus",
        "publisher": metadata.get("publisher") or "",
        "title": metadata.get("item_title") or metadata.get("section") or "근거",
        "date": metadata.get("source_date") or metadata.get("collected_at") or "",
        "url": metadata.get("url") or "",
        "score": round(float(source.score), 4),
        "snippet": content[:220] + ("…" if len(content) > 220 else ""),
    }


def _profile_for_appid(
    profiles: list[tuple[Path, dict[str, Any]]],
    appid: int,
) -> dict[str, Any] | None:
    for _, profile in profiles:
        try:
            if int(profile.get("appid")) == int(appid):
                return profile
        except (TypeError, ValueError):
            continue
    return None


def _updates_detail(updates: list[CorpusUpdate]) -> str:
    if not updates:
        return "새로 확인된 게임 문서 없음"
    labels = [
        f"{item.game.name}(appid={item.game.appid}, collected={item.collected}, indexed={item.indexed})"
        for item in updates
    ]
    return ", ".join(labels)


def _game_from_update(update: CorpusUpdate) -> dict[str, Any]:
    return {
        "appid": update.game.appid,
        "name": update.game.name,
        "image": _steam_header_image(update.game.appid),
        "url": f"https://store.steampowered.com/app/{update.game.appid}/",
        "status": "새 문서 생성" if update.collected else "기존 문서 사용",
    }


def _update_payload(update: CorpusUpdate) -> dict[str, Any]:
    return {
        "appid": update.game.appid,
        "name": update.game.name,
        "collected": update.collected,
        "indexed": update.indexed,
        "reason": update.reason,
        "markdown_path": str(update.markdown_path),
    }


def _candidate_payload(candidate: Any, *, discovery_info: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = candidate.profile if isinstance(candidate.profile, dict) else {}
    review = profile.get("recent_review_summary") or {}
    price = profile.get("price") or {}
    final_price = price.get("final")
    return {
        "appid": candidate.appid,
        "name": candidate.name,
        "score": round(float(candidate.score), 2),
        "matched_tags": list(candidate.matched_tags)[:5],
        "matched_facets": list(candidate.matched_facets)[:5],
        "positive_ratio": review.get("positive_ratio"),
        "sample_size": review.get("sample_size"),
        "is_free": price.get("is_free") is True,
        "discount_percent": price.get("discount_percent", 0),
        "price": int(final_price / 100) if isinstance(final_price, (int, float)) else None,
        "release_date": profile.get("release_date") or "",
        "release_coming_soon": bool(profile.get("release_coming_soon")),
        "discovery_reason": str((discovery_info or {}).get("reason") or ""),
        "similarity_score": (discovery_info or {}).get("similarity_score"),
        "matched_aspects": list((discovery_info or {}).get("matched_aspects") or [])[:6],
        "store_summary": str(profile.get("store_summary") or ""),
        "genres": list(profile.get("genres") or [])[:3],
        "popular_tags": [
            str(tag.get("name") or "")
            for tag in profile.get("popular_user_tags") or []
            if isinstance(tag, dict) and str(tag.get("name") or "").strip()
        ][:4],
        "image": profile.get("header_image") or _steam_header_image(candidate.appid),
        "url": f"https://store.steampowered.com/app/{candidate.appid}/?l=koreana&cc=kr",
    }


def _recommendation_markdown(
    question: str,
    candidates: list[dict[str, Any]],
    discovery: dict[str, Any],
    query: Any,
) -> str:
    if not candidates:
        return "조건을 확인했지만 Steam에서 검증된 추천 후보를 충분히 찾지 못했습니다. 조건을 조금 넓혀 다시 질문해 주세요."
    summary = str(discovery.get("concept_summary") or "").strip()
    lines = ["## 추천 결과"]
    if summary:
        lines.append(summary)
    elif getattr(query, "sale_required", False):
        lines.append("현재 할인 여부와 최근 사용자 평가를 다시 확인해, 가격과 반응을 함께 볼 만한 게임을 골랐습니다.")
    elif getattr(query, "upcoming_required", False):
        lines.append("아직 출시되지 않은 Steam 게임만 확인해 장르와 인기 태그가 요청 조건에 가까운 순서로 정리했습니다.")
    else:
        lines.append("공식 장르·카테고리와 인기 사용자 태그를 확인해 요청한 플레이 경험에 가까운 게임을 골랐습니다.")
    requested_count = _requested_recommendation_count(question)
    if requested_count and len(candidates) < requested_count:
        lines.append(
            f"Steam에서 모든 조건을 다시 확인한 후보는 요청한 {requested_count}개 중 "
            f"{len(candidates)}개입니다. 맞지 않는 게임으로 수를 채우지 않았습니다."
        )
    for rank, item in enumerate(candidates, start=1):
        reason = _recommendation_reason(item)
        ratio = item.get("positive_ratio")
        rating = f" · 최근 표본 긍정 {ratio * 100:.0f}%" if isinstance(ratio, (int, float)) else ""
        discount = item.get("discount_percent") or 0
        sale = (
            " · 무료 플레이"
            if item.get("is_free")
            else (f" · {discount}% 할인" if discount else "")
        )
        release = f" · 출시일 {item['release_date']}" if getattr(query, "upcoming_required", False) and item.get("release_date") else ""
        lines.append(f"{rank}. **{item['name']}** — {reason}{rating}{sale}{release}")
    lines.append("")
    if discovery.get("source_urls"):
        lines.append(
            "Tavily 검색은 후보를 찾는 데만 사용했고, 표시된 게임은 Steam AppID와 공식 스토어 프로필로 다시 확인했습니다."
        )
    else:
        lines.append("표시된 게임은 Steam 공식 장르·카테고리와 인기 사용자 태그를 기준으로 확인했습니다.")
    return "\n".join(line for line in lines if line is not None)


def _steam_header_image(appid: int) -> str:
    return f"https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/{int(appid)}/header.jpg"


def _requested_recommendation_count(question: str) -> int | None:
    match = re.search(r"(?:게임|작품|기대작|RPG)?\s*([1-9]|10)\s*개", question, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _recommendation_reason(item: dict[str, Any]) -> str:
    discovery_reason = re.sub(r"\s+", " ", str(item.get("discovery_reason") or "")).strip()
    if discovery_reason:
        return discovery_reason[:220]
    summary = re.sub(r"\s+", " ", str(item.get("store_summary") or "")).strip()
    if summary:
        sentence = re.split(r"(?<=[.!?。])\s+", summary, maxsplit=1)[0]
        if len(sentence) >= 25:
            return sentence[:220] + ("…" if len(sentence) > 220 else "")
    genres = [str(value) for value in item.get("genres", []) if str(value).strip()]
    tags = [str(value) for value in item.get("popular_tags", []) if str(value).strip()]
    if genres and tags:
        return f"공식 장르는 {', '.join(genres[:2])}이며, 인기 태그에서 {', '.join(tags[:3])} 성향이 확인됩니다."
    if tags:
        return f"Steam 인기 태그에서 {', '.join(tags[:3])} 성향이 확인됩니다."
    return "Steam 공식 상품 페이지와 요청 조건이 일치해 후보로 확인했습니다."
