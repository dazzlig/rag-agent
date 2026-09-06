"""Narrow tools shared by the orchestrating runtime and the HTTP API.

기획안 §12는 초기 도구를 ``search_games``, ``get_game_facts``,
``compare_candidates``, ``search_game_knowledge``, ``get_user_game_state``
처럼 범위를 좁혀 설계하라고 정한다.

이 모듈은 그중 **기존 구성 요소로 대체할 수 없는 세 가지**만 구현한다.
나머지 두 가지는 이미 같은 계약을 가진 코드가 있어 얇은 wrapper를 새로 만들지
않는다(저장소의 과설계 방지 규칙).

======================== ==================================================
기획안 도구 이름          이 저장소의 구현
======================== ==================================================
``search_games``         :class:`~steam_rag.game_recommendation.candidate_service.DynamicRecommendationService`
``get_game_facts``       :func:`get_game_facts` (이 모듈)
``compare_candidates``   :func:`compare_candidates` (이 모듈)
``search_game_knowledge``:meth:`~steam_rag.rag_search.hybrid_retriever.HybridTimeAwareRetriever.retrieve`
                         (``allowed_appids``로 게임을 고정하고 스포일러 정책으로 필터)
``get_user_game_state``  :func:`get_user_game_state` (이 모듈)
======================== ==================================================

:class:`ToolBudget` 는 §7의 초기 운영값(한 요청의 추가 검색 2회, 전문가 호출
최대 3개)과 §15의 비용 기록을 함께 담당한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from steam_rag.game_recommendation.comparison import ComparisonTable, compare_profiles
from steam_rag.game_recommendation.constraints import evaluate_candidate_conditions
from steam_rag.game_recommendation.profile_store import SteamProfileStore
from steam_rag.game_recommendation.query_parser import RecommendationQuery
from steam_rag.user_workspace.store import WorkspaceStore


@dataclass(slots=True)
class ToolBudget:
    """Per-request call limits (§7) and cost attribution (§15).

    상한에 도달하면 예외를 던지지 않고 ``False``를 돌려준다. 호출부는 확인한
    결과와 부족한 정보를 그대로 반환한다.
    """

    max_extra_searches: int = 2
    max_expert_calls: int = 3
    extra_searches: int = 0
    expert_calls: int = 0
    denied: list[str] = field(default_factory=list)

    def take_extra_search(self, reason: str = "") -> bool:
        if self.extra_searches >= self.max_extra_searches:
            self.denied.append(f"extra_search:{reason or 'limit'}")
            return False
        self.extra_searches += 1
        return True

    def take_expert_call(self, reason: str = "") -> bool:
        if self.expert_calls >= self.max_expert_calls:
            self.denied.append(f"expert_call:{reason or 'limit'}")
            return False
        self.expert_calls += 1
        return True

    @property
    def exhausted(self) -> bool:
        return (
            self.extra_searches >= self.max_extra_searches
            and self.expert_calls >= self.max_expert_calls
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "extra_searches": self.extra_searches,
            "max_extra_searches": self.max_extra_searches,
            "expert_calls": self.expert_calls,
            "max_expert_calls": self.max_expert_calls,
            "denied": list(self.denied),
        }


def get_game_facts(
    store: SteamProfileStore,
    appid: int,
    *,
    query: RecommendationQuery | None = None,
) -> dict[str, Any]:
    """Return one game's confirmed attributes with sources and check times.

    §10.2 '사실과 근거: 항목별 값, 출처, 확인 시점, 적용 범위, 검증 상태'.
    조건이 함께 주어지면 충족·위반·미확인 판정도 반환한다.
    """

    profile = _profile(store, appid)
    if profile is None:
        return {"appid": int(appid), "found": False}

    price = profile.get("price") if isinstance(profile.get("price"), dict) else {}
    review = profile.get("recent_review_summary")
    review = review if isinstance(review, dict) else {}
    facts: dict[str, Any] = {
        "appid": int(appid),
        "found": True,
        "name": profile.get("name"),
        "app_type": profile.get("app_type"),
        "platforms": ["steam"],
        "genres": profile.get("steam_genres_normalized") or [],
        "categories": profile.get("steam_categories_normalized") or [],
        "popular_user_tags": [
            {"name": item.get("name"), "normalized": item.get("normalized"), "rank": item.get("rank")}
            for item in profile.get("popular_user_tags") or []
            if isinstance(item, dict)
        ][:12],
        "combat_facets": profile.get("combat_facets") or [],
        "perspective_facets": profile.get("perspective_facets") or [],
        "dimension_facets": profile.get("dimension_facets") or [],
        "playstyle_facets": profile.get("playstyle_facets") or [],
        "facet_evidence": profile.get("facet_evidence") or [],
        "release_date": profile.get("release_date"),
        "release_coming_soon": profile.get("release_coming_soon"),
        "price": price,
        "recent_review_summary": review,
        "store_summary": profile.get("store_summary") or "",
        "header_image": profile.get("header_image") or "",
        "collected_at": profile.get("collected_at"),
    }
    if query is not None:
        report = evaluate_candidate_conditions(profile, query)
        facts["conditions"] = report.to_dict()
    return facts


def compare_candidates(
    store: SteamProfileStore,
    appids: Sequence[int],
) -> tuple[ComparisonTable, list[int]]:
    """Compare 2~3 games on the fixed experience axes (§4.3).

    Returns the table plus the AppIDs that had no local profile, so the caller
    can say what it could not compare instead of silently dropping a game.
    """

    profiles: list[dict[str, Any]] = []
    missing: list[int] = []
    for appid in list(dict.fromkeys(int(value) for value in appids))[:3]:
        profile = _profile(store, appid)
        if profile is None:
            missing.append(appid)
            continue
        profiles.append(profile)
    return compare_profiles(profiles), missing


def get_user_game_state(
    workspace: WorkspaceStore,
    user_id: str,
    appid: int,
    *,
    playthrough: int = 1,
) -> dict[str, Any]:
    """Read the confirmed state for one user, one game, and one playthrough (§11)."""

    state = workspace.get_game_state(user_id, appid, playthrough=playthrough)
    threads = workspace.list_play_threads(user_id, appid, playthrough=playthrough)
    return {
        **state.to_dict(),
        "threads": [thread.to_dict() for thread in threads],
        "attempts": [item.to_dict() for item in workspace.list_attempts(user_id, appid, playthrough=playthrough)],
        "state_known": not state.is_empty,
    }


def _profile(store: SteamProfileStore, appid: int) -> dict[str, Any] | None:
    for _, profile in store.load_core_profiles(include_expired=True):
        try:
            if int(profile.get("appid")) == int(appid):
                return profile
        except (TypeError, ValueError):
            continue
    return None
