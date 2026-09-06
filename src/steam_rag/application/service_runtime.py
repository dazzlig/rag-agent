from __future__ import annotations

import hashlib
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
from steam_rag.game_expert.expert import GameExpertAgent
from steam_rag.game_expert.support_scope import (
    GameExpertProfile,
    GameExpertRegistry,
    SupportScope,
    classify_topic,
)
from steam_rag.game_recommendation.candidate_service import DynamicRecommendationService
from steam_rag.game_recommendation.comparison import comparison_markdown
from steam_rag.game_recommendation.constraints import UNVERIFIED
from steam_rag.tools.game_tools import (
    ToolBudget,
    compare_candidates,
    get_game_facts,
    get_user_game_state,
)
from steam_rag.user_workspace.store import DEFAULT_THREAD_TOPICS, WorkspaceStore
from steam_rag.game_recommendation.profile_builder import collect_recommendation_profile
from steam_rag.game_recommendation.profile_store import SteamProfileStore
from steam_rag.game_recommendation.query_parser import (
    OpenAIRecommendationQueryStructurer,
    RecommendationQuery,
    parse_recommendation_query,
)
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
    #: §11 개인화 정보는 게임 사실 데이터와 다른 저장소에 분리한다.
    workspace_db: Path = Path("data/steam_workspace.db")
    #: §6.2 게임별 전문가 설정과 지원 범위.
    expert_dir: Path = Path("data/game_experts")


DISCOVERY_WORKSPACE = "discovery"
PLAY_WORKSPACE = "play"

#: §7 "추가 조사"로 확인할 수 있는 조건. 게임 설명·리뷰 문서에서 답이 나오는 항목만
#: 넣는다. 가격·할인·출시 상태는 문서 검색으로 확인하지 않는다.
RESEARCHABLE_CONDITION_GROUPS = frozenset(
    {
        "combat",
        "perspective",
        "dimension",
        "playstyle",
        "genres",
        "categories",
        "required_tags",
        "excluded_conditions",
    }
)

#: §4.3 후보 거절 이유가 담긴 표현. 있으면 검색 계획 자체를 바꾼다.
CANDIDATE_FEEDBACK_PATTERN = re.compile(
    r"(그림체|아트|비주얼|분위기|전투|스토리|난도|난이도|반복|협동|멀티|가격|예산)"
    r"[^\n]{0,20}(좋은데|괜찮은데|별로|싫|아쉬|말고|보다|더|이미\s*해)"
)


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
        self._workspace: WorkspaceStore | None = None
        self._experts: GameExpertRegistry | None = None
        self._workspace_lock = threading.Lock()

    @property
    def workspace(self) -> WorkspaceStore:
        """Lazily open the user-scoped store shared by both spaces (§11)."""

        if self._workspace is None:
            with self._workspace_lock:
                if self._workspace is None:
                    self._workspace = WorkspaceStore(self.paths.workspace_db)
        return self._workspace

    @property
    def experts(self) -> GameExpertRegistry:
        """Load the per-game expert configuration once (§6.2)."""

        if self._experts is None:
            with self._workspace_lock:
                if self._experts is None:
                    self._experts = GameExpertRegistry.load(self.paths.expert_dir)
        return self._experts

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
        conversation_state: dict[str, Any] | None = None,
        workspace: str = DISCOVERY_WORKSPACE,
        user_id: str = "local",
        session_id: str = "",
        game_id: int | None = None,
        thread_id: str = "",
        playthrough: int = 1,
    ) -> dict[str, Any]:
        """Answer one request inside exactly one workspace.

        §4.4 / §11: 탐색 공간과 게임별 플레이 공간은 대화 저장과 컨텍스트 선택을
        모두 분리한다. 공략 요청은 탐색 대화를 읽지 않고, 탐색 요청은 다른
        게임의 진행도나 스포일러 설정을 읽지 않는다.
        """

        with telemetry_session() as telemetry:
            budget = ToolBudget()
            if workspace == PLAY_WORKSPACE and game_id:
                payload = self._play(
                    question,
                    user_id=user_id,
                    appid=int(game_id),
                    thread_id=thread_id,
                    playthrough=max(1, int(playthrough)),
                    top_k=max(1, min(int(top_k), 10)),
                    budget=budget,
                )
            else:
                payload = self._ask_impl(
                    question,
                    top_k=top_k,
                    history=history,
                    context_games=context_games,
                    conversation_state=conversation_state,
                    user_id=user_id,
                    session_id=session_id,
                    budget=budget,
                )
            payload["workspace"] = workspace if workspace == PLAY_WORKSPACE and game_id else DISCOVERY_WORKSPACE
            payload["budget"] = budget.to_dict()
            payload["telemetry"] = telemetry.snapshot()
            return payload

    def _ask_impl(
        self,
        question: str,
        *,
        top_k: int = 6,
        history: list[dict[str, str]] | None = None,
        context_games: list[dict[str, Any]] | None = None,
        conversation_state: dict[str, Any] | None = None,
        user_id: str = "local",
        session_id: str = "",
        budget: ToolBudget | None = None,
    ) -> dict[str, Any]:
        question = str(question or "").strip()
        if not question:
            raise ValueError("질문을 입력해 주세요.")
        load_env_file()
        budget = budget or ToolBudget()
        prior_state = _normalize_conversation_state(conversation_state)
        active_games = _merge_context_games(
            prior_state.get("active_games", []),
            context_games or [],
        )
        resolved_question = question
        context_used = False
        followup_relation = _followup_relation(question, history or [], active_games)
        rewrite_context_games = [] if followup_relation == "correction" else active_games
        if history or active_games:
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
        target_games = _select_context_targets(
            question,
            resolved_question,
            active_games,
            followup_relation=followup_relation,
        )
        if target_games:
            bound_question = _bind_appids_to_question(resolved_question, target_games)
            context_used = context_used or bound_question != resolved_question
            resolved_question = bound_question
        excluded_appids = _context_appids(active_games) if followup_relation == "correction" else set()
        route = _route_intent(
            question,
            resolved_question,
            followup_relation=followup_relation,
            prior_state=prior_state,
            target_games=target_games,
        )
        feedback = self._candidate_feedback(
            question, active_games, user_id=user_id, session_id=session_id
        )
        excluded_appids |= {int(value) for value in feedback.get("excluded_appids", [])}
        if route == "recommendation":
            recommendation_kwargs: dict[str, Any] = {
                "excluded_appids": excluded_appids,
                "budget": budget,
                "feedback": feedback,
            }
            if prior_state.get("recommendation_query"):
                recommendation_kwargs["prior_query"] = prior_state["recommendation_query"]
            if prior_state.get("similarity_spec"):
                recommendation_kwargs["prior_similarity_spec"] = prior_state["similarity_spec"]
            payload = self._recommend(resolved_question, **recommendation_kwargs)
        else:
            research_kwargs: dict[str, Any] = {
                "top_k": max(1, min(int(top_k), 10)),
            }
            if target_games:
                research_kwargs["target_games"] = target_games
            payload = self._research(resolved_question, **research_kwargs)
        payload["resolved_question"] = resolved_question
        payload["conversation_context_used"] = context_used
        payload["followup_relation"] = followup_relation
        payload["excluded_appids"] = sorted(excluded_appids)
        payload["intent_route"] = route
        payload["candidate_feedback"] = feedback
        payload["conversation_state"] = _next_conversation_state(
            payload,
            resolved_question=resolved_question,
            prior_state=prior_state,
        )
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
            "supported_experts": self.experts.summary(),
        }

    # ------------------------------------------------------------------
    # 게임별 플레이 공간 (§4.4, §8)
    # ------------------------------------------------------------------
    def open_play_space(
        self,
        user_id: str,
        *,
        appid: int,
        name: str,
        header_image: str = "",
        platform: str = "steam",
    ) -> dict[str, Any]:
        """Hand off from 탐색 to 플레이 공간 with the game and platform only (§4.4)."""

        handoff = self.workspace.handoff_to_play_space(
            user_id,
            appid=appid,
            name=name,
            header_image=header_image or _steam_header_image(appid),
            platform=platform,
        )
        expert = self.experts.get(appid)
        handoff["support"] = expert.support.to_dict() if expert else {}
        handoff["expert_verified"] = expert is not None
        handoff["available_topics"] = [
            {"topic": topic, "title": title} for topic, title in DEFAULT_THREAD_TOPICS
        ]
        return handoff

    # ------------------------------------------------------------------
    # 내 게임 · 내 취향 · 주제별 대화 (§4.5, §11)
    # ------------------------------------------------------------------
    def list_library(self, user_id: str) -> list[dict[str, Any]]:
        return self.workspace.list_library(user_id)

    def add_library_game(
        self,
        user_id: str,
        *,
        appid: int,
        name: str,
        header_image: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        return self.workspace.add_library_game(
            user_id,
            appid=appid,
            name=name,
            header_image=header_image or _steam_header_image(appid),
            note=note,
        )

    def remove_library_game(self, user_id: str, appid: int) -> bool:
        return self.workspace.remove_library_game(user_id, appid)

    def list_preferences(self, user_id: str) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.workspace.list_preferences(user_id)]

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
    ) -> dict[str, Any] | None:
        preference = self.workspace.set_preference(
            user_id,
            kind=kind,
            value=value,
            label=label,
            evidence=evidence,
            scope=scope,
            session_id=session_id,
        )
        return preference.to_dict() if preference else None

    def delete_preference(self, user_id: str, preference_id: int) -> bool:
        return self.workspace.delete_preference(user_id, preference_id)

    def list_play_threads(
        self, user_id: str, appid: int, *, playthrough: int | None = None
    ) -> list[dict[str, Any]]:
        return [
            thread.to_dict()
            for thread in self.workspace.list_play_threads(user_id, appid, playthrough=playthrough)
        ]

    def open_play_thread(
        self,
        user_id: str,
        *,
        appid: int,
        topic: str = "general",
        title: str = "",
        playthrough: int = 1,
    ) -> dict[str, Any]:
        thread = self.workspace.open_play_thread(
            user_id, appid=appid, topic=topic, title=title, playthrough=playthrough
        )
        return thread.to_dict()

    def play_thread_messages(
        self, user_id: str, thread_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        return self.workspace.recent_play_messages(user_id, thread_id, limit=limit)

    def game_state(self, user_id: str, appid: int, *, playthrough: int = 1) -> dict[str, Any]:
        return get_user_game_state(self.workspace, user_id, appid, playthrough=playthrough)

    def update_game_state(self, user_id: str, appid: int, **changes: Any) -> dict[str, Any]:
        playthrough = int(changes.pop("playthrough", 1) or 1)
        state = self.workspace.update_game_state(
            user_id, appid, playthrough=playthrough, **changes
        )
        return state.to_dict()

    def start_new_playthrough(self, user_id: str, appid: int) -> dict[str, Any]:
        """Start a new run without overwriting the previous progress (§11)."""

        playthrough = self.workspace.next_playthrough(user_id, appid)
        state = self.workspace.update_game_state(user_id, appid, playthrough=playthrough, progress="")
        return state.to_dict()

    def _play(
        self,
        question: str,
        *,
        user_id: str,
        appid: int,
        thread_id: str,
        playthrough: int,
        top_k: int,
        budget: ToolBudget,
    ) -> dict[str, Any]:
        """Answer inside one game's play space.

        컨텍스트는 이 게임, 이 회차, 이 주제 대화만 사용한다. 탐색 대화와 다른
        게임 상태는 이 경로에서 읽지 않는다(§11).
        """

        load_env_file()
        store = self.workspace
        thread = store.get_play_thread(user_id, thread_id) if thread_id else None
        if thread is None or thread.appid != int(appid):
            thread = store.open_play_thread(
                user_id,
                appid=appid,
                topic=classify_topic(question),
                playthrough=playthrough,
            )
        playthrough = thread.playthrough
        context = store.play_context(
            user_id, appid=appid, thread_id=thread.thread_id, playthrough=playthrough
        )
        state = store.get_game_state(user_id, appid, playthrough=playthrough)
        profile = self.experts.get(appid) or self._adhoc_expert_profile(appid)

        embedder = OpenAIEmbedder(self.embedding_model)
        generator = OpenAIAnswerGenerator(self.answer_model)
        manager = OnDemandCorpusManager(
            client=SteamAPIClient(),
            catalog_path=self.paths.catalog_path,
            docs_dir=self.paths.docs_dir,
            raw_dir=self.paths.raw_dir,
            profiles_dir=self.paths.profiles_dir,
            index_path=self.paths.index_path,
            max_age=timedelta(hours=24),
        )
        updates: list[CorpusUpdate] = []
        try:
            updates = manager.ensure_questions(
                [f"{profile.name} appid: {appid}"], embedder, strict=False, max_games=1
            )
        except Exception:
            updates = []
        if not self.paths.index_path.exists():
            raise FileNotFoundError(
                "검색할 벡터 인덱스가 없습니다. 이 게임의 문서를 먼저 수집해 주세요."
            )
        pipeline = RAGPipeline.from_path(
            self.paths.index_path,
            embedder,
            generator,
            reranker=self._get_reranker(),
            rerank_candidates=20,
        )
        indexed = {
            int(document.metadata["appid"])
            for document in pipeline.index.documents
            if str(document.metadata.get("appid") or "").isdigit()
        }
        if int(appid) not in indexed:
            raise LookupError(
                f"'{profile.name}'의 문서를 아직 수집하지 못해 공략 답변을 만들 수 없습니다. "
                "게임의 정식명 또는 Steam AppID를 확인한 뒤 다시 시도해 주세요."
            )

        budget.take_expert_call(f"appid:{appid}")
        agent = GameExpertAgent(
            profile,
            pipeline.retriever,
            embedder,
            generator,
            reranker=self._get_reranker(),
        )
        result = agent.answer(
            question,
            state=state,
            attempts=context["attempts"],
            thread_messages=context["messages"],
            k=top_k,
        )

        store.append_play_message(
            user_id, thread.thread_id, appid=appid, role="user", content=question
        )
        store.append_play_message(
            user_id,
            thread.thread_id,
            appid=appid,
            role="assistant",
            content=result.answer,
            payload={"scope": result.scope.to_dict(), "spoiler": result.spoiler},
        )
        if result.is_retry:
            store.record_attempt(
                user_id,
                appid,
                action=question,
                outcome="사용자가 효과 없었다고 보고",
                playthrough=playthrough,
                thread_id=thread.thread_id,
            )

        sources = [_source_payload(item) for item in result.evidence]
        evidence_contexts = [_evidence_payload(item) for item in result.evidence]
        return {
            "mode": "play",
            "answer": result.answer,
            "query_variants": [],
            "agents": [
                {
                    "agent": "Play Space Router",
                    "status": "completed",
                    "detail": (
                        f"appid={appid}, thread={thread.topic}, playthrough={playthrough}, "
                        f"documents={_updates_detail(updates)}"
                    ),
                },
                *result.trace,
            ],
            "games": [
                {
                    "appid": int(appid),
                    "name": profile.name,
                    "image": _steam_header_image(appid),
                    "url": f"https://store.steampowered.com/app/{int(appid)}/?l=koreana&cc=kr",
                    "status": "게임별 플레이 공간",
                }
            ],
            "sources": sources,
            "evidence_contexts": evidence_contexts,
            "claim_citations": [],
            "evidence_coverage": {},
            "expert": result.to_dict(),
            "thread": thread.to_dict(),
            "game_state": state.to_dict(),
            "conversation_state": {
                "active_games": [{"appid": int(appid), "name": profile.name}],
                "last_mode": "play",
                "last_resolved_question": question[:1600],
                "recommendation_query": {},
                "similarity_spec": {},
            },
        }

    def _adhoc_expert_profile(self, appid: int) -> GameExpertProfile:
        """Answer outside the verified 3 games without promising the same level (§9.1)."""

        name = f"Steam App {int(appid)}"
        try:
            facts = get_game_facts(SteamProfileStore(self.paths.service_db), appid)
            if facts.get("found") and facts.get("name"):
                name = str(facts["name"])
        except Exception:
            pass
        return GameExpertProfile(
            appid=int(appid),
            game_key=f"app_{int(appid)}",
            name=name,
            platforms=("steam",),
            support=SupportScope(
                out_of_scope_note="검증된 지원 범위가 지정되지 않은 게임입니다.",
            ),
        )

    # ------------------------------------------------------------------
    # 비교 (§4.3, §4.5)
    # ------------------------------------------------------------------
    def compare(self, appids: list[int]) -> dict[str, Any]:
        """Compare selected candidates on the same experience axes."""

        store = SteamProfileStore(self.paths.service_db)
        store.import_profile_directory(self.paths.profiles_dir)
        table, missing = compare_candidates(store, appids)
        return {
            "mode": "comparison",
            "comparison": table.to_dict(),
            "answer": comparison_markdown(table),
            "missing_appids": missing,
            "agents": [
                {
                    "agent": "Comparison Service",
                    "status": "completed" if len(table.games) >= 2 else "insufficient_games",
                    "detail": (
                        f"games={len(table.games)}, differing_axes={len(table.differing_axes)}, "
                        f"missing_profiles={len(missing)}"
                    ),
                }
            ],
        }

    def _candidate_feedback(
        self,
        question: str,
        active_games: list[dict[str, Any]],
        *,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Turn a rejection reason into a new search plan (§4.3).

        필수 조건은 사용자의 말 없이 완화하지 않는다. 이 함수는 제외 후보와
        선호 가중치, 그리고 사용자가 이번 문장에서 새로 말한 조건만 만든다.
        """

        if not active_games or not CANDIDATE_FEEDBACK_PATTERN.search(question):
            return {}
        try:
            parsed = OpenAIAnswerGenerator(self.answer_model).interpret_candidate_feedback(
                question, active_games
            )
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict) or not parsed:
            return {}
        known = {int(game["appid"]) for game in active_games}
        rejected = [
            {
                "appid": int(row.get("appid")),
                "aspect": str(row.get("aspect") or ""),
                "reason": str(row.get("reason") or "")[:200],
            }
            for row in parsed.get("rejected") or []
            if isinstance(row, dict) and _int_or_none(row.get("appid")) in known
        ]
        liked = [
            {"appid": int(row.get("appid")), "aspect": str(row.get("aspect") or "")}
            for row in parsed.get("liked") or []
            if isinstance(row, dict) and _int_or_none(row.get("appid")) in known
        ]
        played = {
            int(value)
            for value in parsed.get("already_played") or []
            if _int_or_none(value) in known
        }
        feedback = {
            "excluded_appids": sorted({row["appid"] for row in rejected} | played),
            "rejected": rejected,
            "liked": liked,
            "preferred_aspects": [
                str(value) for value in parsed.get("preferred_aspects") or [] if str(value).strip()
            ][:6],
            "new_must": [str(value) for value in parsed.get("new_must") or [] if str(value).strip()][:6],
            "new_exclude": [
                str(value) for value in parsed.get("new_exclude") or [] if str(value).strip()
            ][:6],
            "needs_new_candidates": bool(parsed.get("needs_new_candidates", True)),
        }
        self._remember_feedback(user_id, session_id, feedback, question)
        return feedback

    def _remember_feedback(
        self,
        user_id: str,
        session_id: str,
        feedback: dict[str, Any],
        question: str,
    ) -> None:
        """Record this turn's stated conditions as session memory, not permanent taste (§11)."""

        if not session_id:
            return
        store = self.workspace
        evidence = question.strip()[:200]
        for value in feedback.get("new_exclude", []):
            store.set_preference(
                user_id,
                kind="dislike",
                value=value,
                label=value,
                evidence=evidence,
                scope="session",
                session_id=session_id,
            )
        for value in [*feedback.get("new_must", []), *feedback.get("preferred_aspects", [])]:
            store.set_preference(
                user_id,
                kind="like",
                value=value,
                label=value,
                evidence=evidence,
                scope="session",
                session_id=session_id,
            )

    def _research(
        self,
        question: str,
        *,
        top_k: int,
        target_games: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        embedder = OpenAIEmbedder(self.embedding_model)
        generator = OpenAIAnswerGenerator(self.answer_model)
        variants = QueryExpansionAgent(generator, max_variants=4).expand(question)
        trusted_games = _normalize_context_games(target_games or [])
        trusted_variants = [
            f"{game['name']} appid: {game['appid']}"
            for game in trusted_games
        ]
        variants = list(dict.fromkeys([*trusted_variants, *variants]))
        manager = OnDemandCorpusManager(
            client=SteamAPIClient(),
            catalog_path=self.paths.catalog_path,
            docs_dir=self.paths.docs_dir,
            raw_dir=self.paths.raw_dir,
            profiles_dir=self.paths.profiles_dir,
            index_path=self.paths.index_path,
            max_age=timedelta(hours=24),
        )
        expected_game_count = max(_expected_game_count(question), len(trusted_games))
        updates = manager.ensure_questions(
            trusted_variants or variants,
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
        target_appids = [int(game["appid"]) for game in trusted_games]
        target_appids.extend(
            update.game.appid for update in updates if update.game.appid not in target_appids
        )
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
        sources = [_source_payload(source) for source in result.sources]
        evidence_contexts = [_evidence_payload(source) for source in result.sources]
        coverage = result.metadata.get("evidence_coverage", {})
        return {
            "mode": "research",
            "answer": result.answer,
            "query_variants": result.metadata.get("query_variants", variants),
            "agents": trace,
            "games": games,
            "sources": sources,
            "evidence_contexts": evidence_contexts,
            "claim_citations": _claim_citations(coverage, evidence_contexts),
            "evidence_coverage": coverage,
            "corpus_updates": [_update_payload(update) for update in updates],
        }

    def _recommend(
        self,
        question: str,
        *,
        excluded_appids: set[int] | None = None,
        prior_query: dict[str, Any] | None = None,
        prior_similarity_spec: dict[str, Any] | None = None,
        budget: ToolBudget | None = None,
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        budget = budget or ToolBudget()
        feedback = feedback or {}
        generator = OpenAIAnswerGenerator(self.answer_model)
        client = SteamAPIClient()
        current_query = OpenAIRecommendationQueryStructurer(self.answer_model).structure(question)
        query = _merge_recommendation_query(prior_query, current_query, question)
        excluded_appids = set(excluded_appids or ())
        restored_similarity_spec = _similarity_spec_from_state(prior_similarity_spec)
        concept_recommendation = _is_concept_recommendation(question) or restored_similarity_spec is not None
        store = SteamProfileStore(self.paths.service_db)
        store.import_profile_directory(self.paths.profiles_dir)
        if store.summary()["registry_count"] == 0 and self.paths.catalog_path.exists():
            store.sync_catalog_file(self.paths.catalog_path)

        reference: ReferenceGame | None = None
        similarity_spec: SimilaritySpec | None = None
        reference_payload: dict[str, Any] = {}
        similarity_scores: list[SimilarityScore] = []
        if restored_similarity_spec is not None:
            reference = restored_similarity_spec.seed
            similarity_spec = adapt_similarity_spec_to_question(
                restored_similarity_spec,
                question,
            )
            reference_payload = {
                **reference.to_dict(),
                "canonical_name": reference.name,
                "restored_from_conversation": True,
            }
            similarity_scores = [
                item
                for item in rank_similar_profiles(
                    store.load_core_profiles(include_expired=True),
                    similarity_spec,
                    limit=20,
                )
                if item.appid not in excluded_appids
            ]
        elif concept_recommendation:
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
        if concept_recommendation and not prior_query:
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
        selected_candidates = ranked_candidates[:5]
        verification = self._investigate_unverified_conditions(
            selected_candidates, budget=budget
        )
        candidates = [
            _candidate_payload(
                item,
                discovery_info=(similarity_info.get(item.appid) or verified.get(item.appid)),
            )
            for item in selected_candidates
        ]
        evidence_contexts = [_candidate_evidence_payload(item) for item in selected_candidates]
        hard_gate = _hard_constraint_gate_payload(query, run, candidates)
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
                "detail": (
                    f"ranked={len(run.selection.candidates)}, "
                    f"verified={len(run.selection.verified_candidates)}, "
                    f"unverified={len(run.selection.unverified_candidates)}, "
                    f"profiles={run.selection.scanned_profiles}"
                ),
            },
            {
                "agent": "Condition Verification Agent",
                "status": verification["status"],
                "detail": verification["detail"],
            },
            {"agent": "Answer Agent", "status": "completed", "detail": f"games={len(candidates)}"},
        ]
        if feedback.get("rejected"):
            agents.insert(
                1,
                {
                    "agent": "Candidate Feedback Agent",
                    "status": "completed",
                    "detail": (
                        f"rejected={len(feedback['rejected'])}, "
                        f"preferred={','.join(feedback.get('preferred_aspects', [])) or '없음'}, "
                        f"new_candidates={feedback.get('needs_new_candidates', True)}"
                    ),
                },
            )
        return {
            "mode": "recommendation",
            "answer": answer,
            "query_variants": [],
            "agents": agents,
            "games": candidates,
            "sources": [
                {
                    "source_id": f"web-discovery:{index}",
                    "title": "후보 발굴 참고 자료",
                    "url": url,
                    "section": "web_discovery",
                }
                for index, url in enumerate(discovery.get("source_urls", [])[:5], start=1)
                if str(url).startswith("http")
            ],
            "evidence_contexts": evidence_contexts,
            "claim_citations": [
                {
                    "claim_id": "recommendation_fit",
                    "supported": bool(evidence_contexts),
                    "source_ids": [item["source_id"] for item in evidence_contexts],
                }
            ],
            "evidence_coverage": {},
            "discovery_error": discovery_error,
            "recommendation": {
                **run.to_dict(),
                "reference_game": reference_payload,
                "similarity_spec": similarity_spec.to_dict() if similarity_spec else {},
                "excluded_appids": sorted(excluded_appids),
                "effective_query": query.model_dump(),
                "hard_constraint_gate": hard_gate,
                "condition_verification": verification,
                "candidate_feedback": feedback,
            },
        }

    def _investigate_unverified_conditions(
        self,
        candidates: list[Any],
        *,
        budget: ToolBudget,
    ) -> dict[str, Any]:
        """§7 '추가 조사': 태그만으로 부족한 조건을 게임 문서에서 다시 확인한다.

        한 요청의 추가 검색과 전문가 호출은 :class:`ToolBudget` 상한을 지킨다.
        상한에 도달하면 확인한 결과와 남은 미확인 항목을 그대로 반환한다.
        """

        # 가격·할인·출시 상태는 문서 검색으로 확인할 수 없다. 게임 설명에서 답이
        # 나올 수 있는 조건이 있을 때만 인덱스를 연다.
        pending = [
            item
            for item in candidates
            if item.constraints is not None
            and any(
                verdict.group in RESEARCHABLE_CONDITION_GROUPS
                for verdict in item.constraints.must_unverified
            )
        ]
        if not pending:
            return {"status": "not_required", "detail": "문서로 확인할 미확인 조건 없음", "resolved": []}
        if not self.paths.index_path.exists():
            return {
                "status": "no_corpus",
                "detail": f"추가 조사 대상 {len(pending)}개, 검색 인덱스 없음",
                "resolved": [],
            }
        try:
            embedder = OpenAIEmbedder(self.embedding_model)
            pipeline = RAGPipeline.from_path(self.paths.index_path, embedder)
        except Exception as exc:
            return {
                "status": "unavailable",
                "detail": f"{type(exc).__name__}: {exc}",
                "resolved": [],
            }
        indexed = {
            int(document.metadata["appid"])
            for document in pipeline.index.documents
            if str(document.metadata.get("appid") or "").isdigit()
        }
        resolved: list[dict[str, Any]] = []
        checked = 0
        for candidate in pending:
            if candidate.appid not in indexed:
                continue
            if not budget.take_expert_call(f"verify:{candidate.appid}"):
                break
            checked += 1
            labels = [verdict.label for verdict in candidate.constraints.must_unverified]
            try:
                results = pipeline.search(
                    f"{candidate.name} {' '.join(labels)}", k=4
                )
            except Exception:
                continue
            supporting = [
                _source_payload(result)
                for result in results
                if int(result.document.metadata.get("appid") or 0) == candidate.appid
            ]
            resolved.append(
                {
                    "appid": candidate.appid,
                    "name": candidate.name,
                    "unverified": labels,
                    "evidence_found": len(supporting),
                    "sources": supporting[:2],
                }
            )
        return {
            "status": "completed" if resolved else ("budget_exhausted" if checked else "no_corpus"),
            "detail": (
                f"대상 {len(pending)}개, 추가 조사 {checked}개, "
                f"근거 확보 {sum(1 for item in resolved if item['evidence_found'])}개"
            ),
            "resolved": resolved,
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


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _normalize_context_games(context_games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: dict[int, dict[str, Any]] = {}
    for raw in context_games:
        if not isinstance(raw, dict):
            continue
        try:
            appid = int(raw.get("appid"))
        except (TypeError, ValueError):
            continue
        name = re.sub(r"\s+", " ", str(raw.get("name") or "")).strip()
        if appid <= 0 or not name or name in {":", f":{appid}"}:
            continue
        normalized[appid] = {"appid": appid, "name": name[:200]}
    return list(normalized.values())[:10]


def _merge_context_games(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for group in groups:
        for game in _normalize_context_games(group):
            merged[int(game["appid"])] = game
    return list(merged.values())[:10]


def _normalize_conversation_state(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "active_games": [],
            "last_mode": "",
            "last_resolved_question": "",
            "recommendation_query": {},
            "similarity_spec": {},
        }
    mode = str(value.get("last_mode") or "").strip()
    if mode not in {"research", "recommendation"}:
        mode = ""
    return {
        "active_games": _normalize_context_games(value.get("active_games") or []),
        "last_mode": mode,
        "last_resolved_question": str(value.get("last_resolved_question") or "")[:1600],
        "recommendation_query": (
            dict(value.get("recommendation_query") or {})
            if isinstance(value.get("recommendation_query"), dict)
            else {}
        ),
        "similarity_spec": (
            dict(value.get("similarity_spec") or {})
            if isinstance(value.get("similarity_spec"), dict)
            else {}
        ),
    }


def _select_context_targets(
    question: str,
    resolved_question: str,
    active_games: list[dict[str, Any]],
    *,
    followup_relation: str,
) -> list[dict[str, Any]]:
    """Select only AppIDs explicitly grounded by the current turn.

    Multiple recommendation cards are context, not automatic retrieval targets.
    The rewriter must name/bind one of them, or the user must ask for an ordinal
    or a comparison, before research is scoped to those AppIDs.
    """

    games = _normalize_context_games(active_games)
    if not games or followup_relation == "correction":
        return []
    explicit_appids = set(
        int(value)
        for value in re.findall(
            r"(?:store\.steampowered\.com/app/|\bappid\s*[:=#]?\s*)(\d{2,10})",
            resolved_question,
            flags=re.IGNORECASE,
        )
    )
    selected = [game for game in games if int(game["appid"]) in explicit_appids]
    if selected:
        return selected

    normalized_resolved = _normalize(resolved_question)
    selected = [
        game
        for game in games
        if len(_normalize(game["name"])) >= 3
        and _normalize(game["name"]) in normalized_resolved
    ]
    if selected:
        return selected
    if len(games) == 1:
        return games
    ordinal = re.search(r"(?:첫\s*번째|1\s*번|첫째)", question)
    if ordinal:
        return games[:1]
    ordinal = re.search(r"(?:두\s*번째|2\s*번|둘째)", question)
    if ordinal and len(games) >= 2:
        return games[1:2]
    if _expected_game_count(question) > 1 and re.search(r"그\s*둘|두\s*게임|둘\s*중|비교", question):
        return games[:2]
    return []


def _bind_appids_to_question(question: str, games: list[dict[str, Any]]) -> str:
    bound = question.strip()
    existing = set(
        int(value)
        for value in re.findall(r"\bappid\s*[:=#]?\s*(\d{2,10})", bound, flags=re.IGNORECASE)
    )
    additions = [
        f"{game['name']} (appid: {game['appid']})"
        for game in games
        if int(game["appid"]) not in existing
    ]
    if additions:
        bound = f"{bound}\n대상 Steam 게임: {', '.join(additions)}"
    return bound


def _route_intent(
    original_question: str,
    resolved_question: str,
    *,
    followup_relation: str,
    prior_state: dict[str, Any],
    target_games: list[dict[str, Any]],
) -> str:
    """Apply deterministic routing before any expensive recommendation work."""

    lowered = resolved_question.casefold()
    detail_intent = bool(
        re.search(
            r"가격|할인\s*(?:중|여부|율)|평가|리뷰|업데이트|패치|전투|장점|단점|"
            r"어떤\s*게임|자세히|분석|플레이\s*방식|지원해|가능해",
            lowered,
        )
    )
    if target_games and detail_intent and not _is_concept_recommendation(resolved_question):
        return "research"
    prior_recommendation = bool(prior_state.get("recommendation_query"))
    recommendation_refinement = bool(
        re.search(
            r"그\s*중|후보|추천|조건|만\s*(?:알려|골라)|빼고|제외|상관\s*없|"
            r"비슷|같은\s*(?:게임|작품)|다른\s*(?:게임|후보)",
            original_question.casefold(),
        )
    )
    if prior_recommendation and followup_relation != "standalone" and recommendation_refinement and not target_games:
        return "recommendation"
    return "recommendation" if _is_broad_recommendation(resolved_question) else "research"


def _next_conversation_state(
    payload: dict[str, Any],
    *,
    resolved_question: str,
    prior_state: dict[str, Any],
) -> dict[str, Any]:
    active_games = _normalize_context_games(payload.get("games") or [])
    if not active_games:
        active_games = _normalize_context_games(prior_state.get("active_games") or [])
    recommendation = payload.get("recommendation") if isinstance(payload.get("recommendation"), dict) else {}
    query = recommendation.get("effective_query")
    spec = recommendation.get("similarity_spec")
    return {
        "active_games": active_games,
        "last_mode": str(payload.get("mode") or ""),
        "last_resolved_question": resolved_question[:1600],
        "recommendation_query": (
            dict(query)
            if isinstance(query, dict)
            else dict(prior_state.get("recommendation_query") or {})
        ),
        "similarity_spec": (
            dict(spec)
            if isinstance(spec, dict) and spec
            else dict(prior_state.get("similarity_spec") or {})
        ),
    }


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
    if re.search(
        r"게임\s*추천|추천\s*게임|추천해\s*줘|추천해줘|추천해\s*주세요|"
        r"게임(?:들)?(?:을|를)?\s*원해|같은\s*(?:게임|작품)|"
        r"앞으로\s*나올|출시\s*예정|신작.*(?:요약|추천)|기대작",
        lowered,
    ):
        return True
    # A price/sale phrase alone is a single-game analysis intent. Treat it as
    # discovery only when the user explicitly asks for a list/candidates.
    return bool(
        re.search(r"현재\s*(?:세일|할인)|세일\s*중|할인\s*중", lowered)
        and re.search(r"게임들|목록|후보|뭐가|어떤\s*게임|추천|\d+\s*개", lowered)
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


def _merge_recommendation_query(
    prior: dict[str, Any] | None,
    current: RecommendationQuery,
    question: str,
) -> RecommendationQuery:
    """Preserve prior hard constraints and apply the current turn as a delta."""

    if not prior:
        return current.normalized()
    try:
        previous = RecommendationQuery.model_validate(prior).normalized()
    except Exception:
        return current.normalized()
    list_fields = (
        "genres",
        "categories",
        "required_tags",
        "combat",
        "perspective",
        "dimension",
        "playstyle",
        "excluded_conditions",
    )
    update: dict[str, Any] = {
        field: list(dict.fromkeys([*getattr(previous, field), *getattr(current, field)]))
        for field in list_fields
    }
    for field in (
        "recent_rating_required",
        "after_update_required",
        "sale_required",
        "upcoming_required",
        "currently_playable_required",
    ):
        update[field] = bool(getattr(previous, field) or getattr(current, field))
    update["price_max_krw"] = (
        current.price_max_krw
        if current.price_max_krw is not None
        else previous.price_max_krw
    )

    lowered = question.casefold()
    if re.search(r"가격\s*(?:은|는)?\s*상관\s*없|예산\s*(?:은|는)?\s*상관\s*없", lowered):
        update["price_max_krw"] = None
    if re.search(r"할인\s*(?:은|는)?\s*상관\s*없|세일\s*(?:은|는)?\s*상관\s*없", lowered):
        update["sale_required"] = False
    if current.upcoming_required:
        update["currently_playable_required"] = False
    if current.currently_playable_required:
        update["upcoming_required"] = False
    return previous.model_copy(update=update).normalized()


def _similarity_spec_from_state(value: dict[str, Any] | None) -> SimilaritySpec | None:
    if not isinstance(value, dict) or not value:
        return None
    seed = value.get("seed")
    if not isinstance(seed, dict):
        return None
    try:
        reference = ReferenceGame(
            appid=int(seed["appid"]),
            name=str(seed["name"]).strip(),
            matched_alias=str(seed.get("matched_alias") or seed["name"]),
            source=str(seed.get("source") or "conversation_state"),
            confidence=float(seed.get("confidence") or 1.0),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if reference.appid <= 0 or not reference.name:
        return None
    raw_weights = value.get("feature_weights")
    weights = (
        tuple((str(key), float(weight)) for key, weight in raw_weights.items())
        if isinstance(raw_weights, dict)
        else ()
    )
    return SimilaritySpec(
        seed=reference,
        must_have=tuple(str(item) for item in value.get("must_have") or [] if str(item)),
        should_have=tuple(str(item) for item in value.get("should_have") or [] if str(item)),
        excluded=tuple(str(item) for item in value.get("excluded") or [] if str(item)),
        feature_weights=weights,
        seed_features=tuple(str(item) for item in value.get("seed_features") or [] if str(item)),
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
    source_id = _source_id(metadata, source.document.page_content)
    return {
        "source_id": source_id,
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


def _source_id(metadata: dict[str, Any], content: str) -> str:
    chunk_id = str(metadata.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    identity = "|".join(
        [
            str(metadata.get("appid") or ""),
            str(metadata.get("section") or ""),
            str(metadata.get("source_date") or metadata.get("collected_at") or ""),
            content,
        ]
    )
    return f"chunk:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def _evidence_payload(source: Any) -> dict[str, Any]:
    metadata = dict(source.document.metadata)
    return {
        "source_id": _source_id(metadata, source.document.page_content),
        "rank": source.rank,
        "appid": metadata.get("appid"),
        "game": metadata.get("game_name") or metadata.get("game_key") or "Steam 게임",
        "section": metadata.get("section") or "document",
        "source_type": metadata.get("source_type") or "steam_corpus",
        "date": metadata.get("source_date") or metadata.get("collected_at") or "",
        "url": metadata.get("url") or "",
        "score": round(float(source.score), 4),
        "content": source.document.page_content,
        "metadata": metadata,
    }


def _claim_citations(
    coverage: dict[str, Any],
    evidence_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_rank = {
        int(item["rank"]): str(item["source_id"])
        for item in evidence_contexts
        if isinstance(item.get("rank"), int) and item.get("source_id")
    }
    citations: list[dict[str, Any]] = []
    for claim in coverage.get("claims", []) if isinstance(coverage, dict) else []:
        if not isinstance(claim, dict):
            continue
        ranks = [int(rank) for rank in claim.get("evidence_ranks") or [] if str(rank).isdigit()]
        citations.append(
            {
                "claim_id": str(claim.get("claim_id") or ""),
                "claim": str(claim.get("text") or ""),
                "supported": bool(claim.get("supported")),
                "evidence_ranks": ranks,
                "source_ids": [by_rank[rank] for rank in ranks if rank in by_rank],
            }
        )
    return citations


def _candidate_evidence_payload(candidate: Any) -> dict[str, Any]:
    profile = candidate.profile if isinstance(candidate.profile, dict) else {}
    appid = int(candidate.appid)
    evidence = {
        "appid": appid,
        "name": candidate.name,
        "collected_at": profile.get("collected_at"),
        "app_type": profile.get("app_type"),
        "genres": profile.get("genres") or [],
        "categories": profile.get("categories") or [],
        "popular_user_tags": profile.get("popular_user_tags") or [],
        "combat_facets": profile.get("combat_facets") or [],
        "perspective_facets": profile.get("perspective_facets") or [],
        "dimension_facets": profile.get("dimension_facets") or [],
        "playstyle_facets": profile.get("playstyle_facets") or [],
        "price": profile.get("price") or {},
        "release_date": profile.get("release_date"),
        "release_coming_soon": profile.get("release_coming_soon"),
        "recent_review_summary": profile.get("recent_review_summary") or {},
        "store_summary": profile.get("store_summary") or "",
    }
    collected_at = str(profile.get("collected_at") or "unknown")
    return {
        "source_id": f"steam-profile:{appid}:{collected_at}",
        "rank": 0,
        "appid": appid,
        "game": candidate.name,
        "section": "recommendation_profile",
        "source_type": "steam_official_profile_and_popular_tags",
        "date": collected_at,
        "url": f"https://store.steampowered.com/app/{appid}/?l=koreana&cc=kr",
        "content": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        "metadata": {"appid": appid, "collected_at": collected_at},
    }


def _hard_constraint_gate_payload(
    query: RecommendationQuery,
    run: Any,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized = query.normalized().model_dump()
    constraints = {
        key: value
        for key, value in normalized.items()
        if value not in (None, False, [], "")
    }
    return {
        "status": "passed" if candidates else "no_verified_match",
        "constraints": constraints,
        "scanned_profiles": int(run.selection.scanned_profiles),
        "hard_filter_matches": int(run.selection.hard_filter_matches),
        "candidate_appids": [int(item["appid"]) for item in candidates],
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
    constraints = getattr(candidate, "constraints", None)
    return {
        "appid": candidate.appid,
        "name": candidate.name,
        "score": round(float(candidate.score), 2),
        "matched_tags": list(candidate.matched_tags)[:5],
        "matched_facets": list(candidate.matched_facets)[:5],
        # §4.2 후보 카드: 잘 맞는 점 / 선택 전 확인 / 정보 상태
        "condition_status": constraints.status if constraints is not None else "satisfied",
        "fit_reasons": constraints.fit_reasons[:5] if constraints is not None else [],
        "checks_before_choosing": (
            constraints.checks_before_choosing[:5] if constraints is not None else []
        ),
        "information_status": constraints.information_status if constraints is not None else {},
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
    unverified_names = [
        item["name"] for item in candidates if item.get("condition_status") == UNVERIFIED
    ]
    if unverified_names and len(unverified_names) == len(candidates):
        lines.append(
            "모든 필수 조건을 확인한 게임은 아직 없습니다. 아래 후보는 일부 조건이 미확인 상태이며, "
            "미확인 항목을 추가로 조사하거나 조건을 조정해 다시 찾을 수 있습니다."
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
        checks = item.get("checks_before_choosing") or []
        if checks:
            lines.append(f"   - 선택 전 확인: {', '.join(checks[:3])}")
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
