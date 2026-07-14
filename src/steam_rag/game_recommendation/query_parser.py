from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, Field

from steam_rag.game_metadata.playstyle import TAG_ALIASES, extract_query_facets, normalize_steam_tags


FACET_SELECTOR_TAGS = {
    "2d",
    "2_5d",
    "3d",
    "third_person",
    "first_person",
    "top_down",
    "isometric",
    "side_scroller",
    "turn_based_combat",
    "turn_based_tactics",
    "real_time_combat",
}

CombatFacet = Literal[
    "turn_based",
    "real_time",
    "melee",
    "ranged",
    "tactical",
    "party_based",
    "boss_focused",
    "direct_control",
    "command_based",
    "auto_combat",
    "shooter",
    "stealth",
]
PerspectiveFacet = Literal[
    "first_person",
    "third_person",
    "side_view",
    "top_down",
    "isometric",
]
DimensionFacet = Literal["2d", "2_5d", "3d", "vr"]
PlaystyleFacet = Literal[
    "exploration",
    "open_world",
    "story_rich",
    "choices_matter",
    "co_op",
    "multiplayer",
    "online_multiplayer",
    "pvp",
    "platforming",
    "survival",
    "crafting",
    "hunting",
    "metroidvania",
    "character_progression",
    "souls_like",
    "roguelike",
]


class RecommendationQuery(BaseModel):
    """Structured conditions extracted from a natural-language recommendation query."""

    genres: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    required_tags: list[str] = Field(default_factory=list)
    combat: list[CombatFacet] = Field(default_factory=list)
    perspective: list[PerspectiveFacet] = Field(default_factory=list)
    dimension: list[DimensionFacet] = Field(default_factory=list)
    playstyle: list[PlaystyleFacet] = Field(default_factory=list)
    recent_rating_required: bool = False
    after_update_required: bool = False
    sale_required: bool = False
    upcoming_required: bool = False
    price_max_krw: int | None = None
    excluded_conditions: list[str] = Field(default_factory=list)

    def normalized(self) -> "RecommendationQuery":
        normalized_required_tags = normalize_steam_tags(self.required_tags)
        return self.model_copy(
            update={
                "genres": normalize_steam_tags(self.genres),
                "categories": normalize_steam_tags(self.categories),
                "required_tags": [
                    tag for tag in normalized_required_tags if tag not in FACET_SELECTOR_TAGS
                ],
                "combat": _normalized_values(self.combat),
                "perspective": _normalized_values(self.perspective),
                "dimension": _normalized_values(self.dimension),
                "playstyle": _normalized_values(self.playstyle),
                "excluded_conditions": normalize_steam_tags(self.excluded_conditions),
            }
        )


def _normalized_values(values: Iterable[str]) -> list[str]:
    return sorted(
        {
            re.sub(r"[^a-z0-9가-힣]+", "_", str(value).casefold()).strip("_")
            for value in values
            if str(value).strip()
        }
    )


def parse_recommendation_query(question: str) -> RecommendationQuery:
    """Deterministic Korean/English fallback parser used when an LLM is unavailable."""

    lowered = question.casefold()
    label = re.sub(r"[^a-z0-9가-힣]+", " ", lowered).strip()
    matched_tags: set[str] = set()
    for alias, canonical in TAG_ALIASES.items():
        if re.search(rf"(?:^| ){re.escape(alias)}(?: |$)", label):
            matched_tags.add(canonical)

    facets = extract_query_facets(question)
    genre_terms = {
        "action": ("action", "액션"),
        "rpg": ("rpg", "롤플레잉"),
        "adventure": ("adventure", "어드벤처", "모험"),
        "strategy": ("strategy", "전략"),
        "simulation": ("simulation", "시뮬레이션"),
        "indie": ("indie", "인디"),
        "sports": ("sports", "스포츠"),
        "racing": ("racing", "레이싱"),
    }
    genres = [
        canonical
        for canonical, terms in genre_terms.items()
        if any(re.search(rf"(?:^|\s){re.escape(term)}(?:\s|$)", label) for term in terms)
    ]
    categories: list[str] = []
    for canonical, terms in {
        "singleplayer": ("single player", "singleplayer", "싱글 플레이어", "싱글플레이어"),
        "multiplayer": ("multiplayer", "멀티플레이어"),
        "co_op": ("co op", "협동"),
    }.items():
        if any(term in label for term in terms):
            categories.append(canonical)

    price_max = None
    won_match = re.search(r"([0-9][0-9,]*)\s*원\s*(?:이하|미만|안쪽|까지)", question)
    manwon_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*만\s*원?\s*(?:이하|미만|안쪽|까지)", question)
    if manwon_match:
        price_max = int(float(manwon_match.group(1)) * 10_000)
    elif won_match:
        price_max = int(won_match.group(1).replace(",", ""))

    excluded: list[str] = []
    for pattern in (r"([^,]+?)\s*(?:제외|빼고)", r"([^,]+?)은\s*싫"):
        for match in re.finditer(pattern, question):
            excluded.extend(normalize_steam_tags([match.group(1).strip()]))

    facet_backed_tags = {
        canonical
        for canonical in matched_tags
        if canonical
        not in set(genres)
        | {"singleplayer", "multiplayer", "co_op"}
        | FACET_SELECTOR_TAGS
    }
    return RecommendationQuery(
        genres=genres,
        categories=categories,
        required_tags=sorted(facet_backed_tags),
        combat=facets.get("combat_facets", []),
        perspective=facets.get("perspective_facets", []),
        dimension=facets.get("dimension_facets", []),
        playstyle=facets.get("playstyle_facets", []),
        recent_rating_required=bool(re.search(r"최근.*(?:평가|리뷰)|평가.*(?:좋|개선|상승)", question)),
        after_update_required=bool(re.search(r"(?:업데이트|패치).*(?:이후|후)|(?:이후|후).*(?:평가|리뷰)", question)),
        sale_required=bool(re.search(r"현재.*(?:세일|할인)|(?:세일|할인)\s*중", question)),
        upcoming_required=bool(re.search(r"앞으로\s*나올|출시\s*예정|미출시|신작.*(?:예정|기대)|기대작", question)),
        price_max_krw=price_max,
        excluded_conditions=excluded,
    ).normalized()


class OpenAIRecommendationQueryStructurer:
    """Use OpenAI Structured Outputs, with deterministic parsing as a safe fallback."""

    def __init__(self, model: str = "gpt-5-mini", client: Any | None = None) -> None:
        self.model = model
        self._client = client or self._build_client()

    @staticmethod
    def _build_client() -> Any:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
        from openai import OpenAI

        return OpenAI()

    def structure(self, question: str) -> RecommendationQuery:
        try:
            completion = self._client.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Steam 게임 추천 질문을 검색 조건으로 변환한다. "
                            "genres/categories는 공식 Steam 필터, required_tags는 인기 사용자 태그, "
                            "combat/perspective/dimension/playstyle은 정규화된 snake_case facet이다. "
                            "질문에 없는 조건을 추측하지 않는다. 가격은 한국 원화 정수로 변환한다. "
                            "sale_required는 사용자가 현재 할인 중인 게임을 요구할 때만 true다. "
                            "upcoming_required는 아직 출시되지 않은 출시 예정작을 요구할 때만 true다."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                response_format=RecommendationQuery,
            )
            parsed = completion.choices[0].message.parsed
            if isinstance(parsed, RecommendationQuery):
                parsed = parsed.normalized()
                fallback = parse_recommendation_query(question)
                return parsed.model_copy(
                    update={
                        "recent_rating_required": (
                            parsed.recent_rating_required or fallback.recent_rating_required
                        ),
                        "after_update_required": (
                            parsed.after_update_required or fallback.after_update_required
                        ),
                        "sale_required": parsed.sale_required or fallback.sale_required,
                        "upcoming_required": parsed.upcoming_required or fallback.upcoming_required,
                        "price_max_krw": parsed.price_max_krw or fallback.price_max_krw,
                    }
                )
        except Exception:
            pass
        return parse_recommendation_query(question)


@dataclass(slots=True)
class CandidateScore:
    appid: int
    name: str
    score: float
    profile_path: str
    official_match_score: float = 0.0
    tag_combination_score: float = 0.0
    tag_rank_rarity_score: float = 0.0
    facet_score: float = 0.0
    soft_inference_score: float = 0.0
    rating_score: float = 0.0
    matched_tags: list[str] = field(default_factory=list)
    matched_facets: list[str] = field(default_factory=list)
    deferred_checks: list[str] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self, *, include_profile: bool = False) -> dict[str, Any]:
        value = {
            "appid": self.appid,
            "name": self.name,
            "score": round(self.score, 6),
            "profile_path": self.profile_path,
            "score_breakdown": {
                "official_match": round(self.official_match_score, 6),
                "tag_combination": round(self.tag_combination_score, 6),
                "tag_rank_rarity": round(self.tag_rank_rarity_score, 6),
                "facet": round(self.facet_score, 6),
                "soft_inference": round(self.soft_inference_score, 6),
                "recent_rating": round(self.rating_score, 6),
            },
            "matched_tags": self.matched_tags,
            "matched_facets": self.matched_facets,
            "deferred_checks": self.deferred_checks,
        }
        if include_profile:
            value["profile"] = self.profile
        return value


@dataclass(slots=True)
class RecommendationSelection:
    question: str
    query: RecommendationQuery
    scanned_profiles: int
    hard_filter_matches: int
    candidates: list[CandidateScore]
    detail_targets: list[CandidateScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "structured_query": self.query.model_dump(),
            "scanned_profiles": self.scanned_profiles,
            "hard_filter_matches": self.hard_filter_matches,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "detail_targets": [candidate.to_dict() for candidate in self.detail_targets],
        }


class RecommendationProfileIndex:
    def __init__(self, profiles: Sequence[tuple[Path, dict[str, Any]]]) -> None:
        self.profiles = list(profiles)
        self.tag_document_frequency: Counter[str] = Counter()
        for _, profile in self.profiles:
            self.tag_document_frequency.update(set(_profile_tags(profile)))

    @classmethod
    def load(cls, profiles_dir: Path) -> "RecommendationProfileIndex":
        profiles: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(profiles_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("appid"):
                profiles.append((path, payload))
        return cls(profiles)

    def search(
        self,
        question: str,
        query: RecommendationQuery,
        *,
        candidate_limit: int = 20,
        detail_limit: int = 5,
        allowed_appids: set[int] | None = None,
    ) -> RecommendationSelection:
        normalized = query.normalized()
        ranked: list[CandidateScore] = []
        for path, profile in self.profiles:
            try:
                profile_appid = int(profile.get("appid"))
            except (TypeError, ValueError):
                continue
            if allowed_appids is not None and profile_appid not in allowed_appids:
                continue
            if not self._passes_hard_filters(profile, normalized):
                continue
            ranked.append(self._score(path, profile, normalized))
        ranked.sort(key=lambda item: (-item.score, item.name.casefold(), item.appid))
        candidates = ranked[: max(0, candidate_limit)]
        detail_targets = candidates[: max(0, detail_limit)]
        return RecommendationSelection(
            question=question,
            query=normalized,
            scanned_profiles=len(self.profiles),
            hard_filter_matches=len(ranked),
            candidates=candidates,
            detail_targets=detail_targets,
        )

    def _passes_hard_filters(self, profile: dict[str, Any], query: RecommendationQuery) -> bool:
        if str(profile.get("app_type") or "game").casefold() != "game":
            return False
        if re.sub(r"[^a-z0-9가-힣]+", "", str(profile.get("name") or "").casefold()) in {
            "game", "games", "steam", "게임",
        }:
            return False
        requirements = {
            "genres": set(query.genres),
            "categories": set(query.categories),
            "tags": set(query.required_tags),
            "combat_facets": set(query.combat),
            "perspective_facets": set(query.perspective),
            "dimension_facets": set(query.dimension),
            "playstyle_facets": set(query.playstyle),
        }
        available = {
            "genres": set(_strings(profile.get("steam_genres_normalized"))),
            "categories": set(_strings(profile.get("steam_categories_normalized"))),
            "tags": set(_profile_tags(profile)),
            "combat_facets": set(_strings(profile.get("combat_facets"))),
            "perspective_facets": set(_strings(profile.get("perspective_facets"))),
            "dimension_facets": set(_strings(profile.get("dimension_facets"))),
            "playstyle_facets": set(_strings(profile.get("playstyle_facets"))),
        }
        available["genres"].update(available["tags"])
        available["categories"].update(available["tags"])
        if any(wanted and not wanted.issubset(available[field]) for field, wanted in requirements.items()):
            return False
        searchable = set(_strings(profile.get("searchable_terms"))) | set().union(*available.values())
        if set(query.excluded_conditions) & searchable:
            return False
        if query.price_max_krw is not None:
            price = profile.get("price") or {}
            final_price = price.get("final") if isinstance(price, dict) else None
            currency = price.get("currency") if isinstance(price, dict) else None
            if currency != "KRW" or not isinstance(final_price, (int, float)):
                return False
            if final_price > query.price_max_krw * 100:
                return False
        if query.sale_required:
            price = profile.get("price") or {}
            if not isinstance(price, dict) or int(price.get("discount_percent") or 0) <= 0:
                return False
        if query.upcoming_required and profile.get("release_coming_soon") is not True:
            return False
        return True

    def _score(self, path: Path, profile: dict[str, Any], query: RecommendationQuery) -> CandidateScore:
        profile_tags = set(_profile_tags(profile))
        requested_tags = set(query.required_tags)
        matched_tags = sorted(requested_tags & profile_tags)
        official_requested = set(query.genres) | set(query.categories)
        official_available = set(_strings(profile.get("steam_genres_normalized"))) | set(
            _strings(profile.get("steam_categories_normalized"))
        )
        official_score = _ratio(official_requested & official_available, official_requested)
        tag_combo_score = _ratio(set(matched_tags), requested_tags)

        rank_map = {
            str(item.get("normalized")): int(item.get("rank") or 20)
            for item in profile.get("popular_user_tags") or []
            if isinstance(item, dict) and item.get("normalized")
        }
        rarity_terms = requested_tags or profile_tags
        weighted_terms: list[float] = []
        for tag in rarity_terms:
            rank = rank_map.get(tag, 20)
            idf = math.log((len(self.profiles) + 1) / (self.tag_document_frequency[tag] + 1)) + 1.0
            weighted_terms.append(idf / math.log2(rank + 1))
        tag_rank_rarity = sum(weighted_terms) / len(weighted_terms) if weighted_terms else 0.0

        facet_pairs = {
            "combat_facets": set(query.combat),
            "perspective_facets": set(query.perspective),
            "dimension_facets": set(query.dimension),
            "playstyle_facets": set(query.playstyle),
        }
        matched_facets: list[str] = []
        requested_facet_count = 0
        for field, wanted in facet_pairs.items():
            requested_facet_count += len(wanted)
            for facet in sorted(wanted & set(_strings(profile.get(field)))):
                matched_facets.append(f"{field}:{facet}")
        facet_score = len(matched_facets) / requested_facet_count if requested_facet_count else 0.0

        inferred = profile.get("inferred_facets") or {}
        inferred_matches = 0
        for field, wanted in facet_pairs.items():
            if isinstance(inferred, dict):
                inferred_matches += len(wanted & set(_strings(inferred.get(field))))
        soft_score = inferred_matches / requested_facet_count if requested_facet_count else 0.0

        review = profile.get("recent_review_summary") or {}
        ratio = review.get("positive_ratio") if isinstance(review, dict) else None
        rating_score = float(ratio) if isinstance(ratio, (int, float)) else 0.0
        deferred: list[str] = []
        if query.recent_rating_required and ratio is None:
            deferred.append("recent_reviews")
        if query.after_update_required:
            deferred.append("after_update_analysis")

        score = (
            2.0 * official_score
            + 2.5 * tag_combo_score
            + 1.5 * tag_rank_rarity
            + 2.0 * facet_score
            + 0.4 * soft_score
            + (1.5 * rating_score if query.recent_rating_required else 0.0)
        )
        return CandidateScore(
            appid=int(profile.get("appid")),
            name=str(profile.get("name") or profile.get("game_key") or profile.get("appid")),
            score=score,
            profile_path=str(path),
            official_match_score=official_score,
            tag_combination_score=tag_combo_score,
            tag_rank_rarity_score=tag_rank_rarity,
            facet_score=facet_score,
            soft_inference_score=soft_score,
            rating_score=rating_score,
            matched_tags=matched_tags,
            matched_facets=matched_facets,
            deferred_checks=deferred,
            profile=profile,
        )


def _strings(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _profile_tags(profile: dict[str, Any]) -> list[str]:
    ranked = [
        str(item.get("normalized"))
        for item in profile.get("popular_user_tags") or []
        if isinstance(item, dict) and item.get("normalized")
    ]
    return ranked or _strings(profile.get("steam_tags_normalized"))


def _ratio(matched: set[str], requested: set[str]) -> float:
    return len(matched) / len(requested) if requested else 0.0
