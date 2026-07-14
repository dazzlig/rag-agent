from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from steam_rag.common.models import SearchResult
from steam_rag.game_metadata.playstyle import extract_query_facets


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    """One answerable unit and the evidence contract required to support it."""

    claim_id: str
    text: str
    required_sections: tuple[str, ...]
    keywords: tuple[str, ...] = ()
    required_game_keys: tuple[str, ...] = ()
    require_date: bool = False
    min_evidence: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "required_sections": list(self.required_sections),
            "keywords": list(self.keywords),
            "required_game_keys": list(self.required_game_keys),
            "require_date": self.require_date,
            "min_evidence": self.min_evidence,
        }


@dataclass(frozen=True, slots=True)
class SearchSpec:
    """Executable retrieval contract shared by retrieval, agents, and evaluation."""

    question: str
    intent: str
    game_keys: tuple[str, ...]
    primary_sections: tuple[str, ...]
    secondary_sections: tuple[str, ...]
    query_variants: tuple[str, ...]
    claims: tuple[EvidenceClaim, ...]
    facets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    time_requirement: str = "none"
    recommendation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "game_keys": list(self.game_keys),
            "primary_sections": list(self.primary_sections),
            "secondary_sections": list(self.secondary_sections),
            "query_variants": list(self.query_variants),
            "claims": [claim.to_dict() for claim in self.claims],
            "facets": {key: list(values) for key, values in self.facets.items()},
            "time_requirement": self.time_requirement,
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True, slots=True)
class ClaimCoverage:
    claim_id: str
    text: str
    supported: bool
    score: float
    evidence_ranks: tuple[int, ...]
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "supported": self.supported,
            "score": self.score,
            "evidence_ranks": list(self.evidence_ranks),
            "missing": list(self.missing),
        }


@dataclass(frozen=True, slots=True)
class EvidenceCoverageReport:
    claim_count: int
    supported_claims: int
    coverage_ratio: float
    claims: tuple[ClaimCoverage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_count": self.claim_count,
            "supported_claims": self.supported_claims,
            "coverage_ratio": self.coverage_ratio,
            "claims": [claim.to_dict() for claim in self.claims],
        }


def build_search_spec(
    question: str,
    *,
    intent: str,
    primary_sections: Sequence[str],
    secondary_sections: Sequence[str],
    game_key: str | None = None,
    game_keys: Sequence[str] = (),
) -> SearchSpec:
    if not question.strip():
        raise ValueError("question must not be empty")
    resolved_game_keys = tuple(dict.fromkeys([*game_keys, *([game_key] if game_key else [])]))
    facets = {
        key: tuple(values)
        for key, values in extract_query_facets(question).items()
        if values
    }
    lowered = question.casefold()
    recent = bool(re.search(r"최근|최신|이후|recent|latest|after|since", lowered))
    recommendation = bool(re.search(r"추천|recommend", lowered))
    needs_update = intent in {"news", "after_update"} or bool(
        re.search(r"업데이트|패치|핫픽스|update|patch|hotfix", lowered)
    )
    needs_review = intent in {"review", "after_update"} or bool(
        re.search(r"리뷰|평가|반응|review|sentiment|reaction", lowered)
    )
    claims: list[EvidenceClaim] = []

    if needs_update:
        claims.append(
            EvidenceClaim(
                "latest_update",
                "최신 패치 또는 업데이트 사건",
                ("analysis", "news") if intent == "after_update" else ("news",),
                ("patch", "update", "hotfix", "패치", "업데이트"),
                resolved_game_keys,
                require_date=True,
            )
        )
    if needs_review:
        claims.append(
            EvidenceClaim(
                "player_sentiment",
                "사용자 평가와 그 근거",
                ("analysis", "review") if intent == "after_update" else ("review",),
                ("positive", "negative", "review", "평가", "긍정", "부정"),
                resolved_game_keys,
                require_date=recent,
            )
        )
    if intent == "after_update":
        claims.append(
            EvidenceClaim(
                "review_change_after_update",
                "업데이트 전후 사용자 평가 변화",
                ("analysis",),
                ("positive_ratio_delta_pp", "change_direction"),
                resolved_game_keys,
                require_date=True,
            )
        )
    if intent == "price":
        claims.append(
            EvidenceClaim(
                "current_stored_price",
                "수집 시점의 가격과 할인",
                ("metadata", "store_summary"),
                ("price", "discount", "가격", "할인"),
                resolved_game_keys,
                require_date=True,
            )
        )
    if intent in {"gameplay", "general"} or recommendation:
        claims.append(
            EvidenceClaim(
                "game_profile",
                "공식 게임 특성과 플레이 방식",
                ("metadata", "store_summary", "about"),
                (),
                resolved_game_keys,
            )
        )
    for facet_type, values in facets.items():
        for value in values:
            claims.append(
                EvidenceClaim(
                    f"facet_{facet_type}_{value}",
                    f"요청 조건 {facet_type}={value}",
                    ("metadata", "store_summary", "about"),
                    tuple(_facet_keywords(value)),
                    resolved_game_keys,
                )
            )
    if recommendation:
        claims.append(
            EvidenceClaim(
                "recommendation_fit",
                "추천 조건을 만족하는 게임과 주의점",
                ("metadata", "store_summary", "about", "analysis", "review"),
                (),
                (),
                min_evidence=1,
            )
        )
    if not claims:
        claims.append(EvidenceClaim("overview", "질문에 대한 기본 근거", tuple(primary_sections), (), resolved_game_keys))

    resolved_primary = list(primary_sections)
    resolved_secondary = list(secondary_sections)
    if needs_update:
        resolved_primary.extend(["analysis", "news"] if intent == "after_update" else ["news"])
    if needs_review:
        resolved_primary.append("review")
    if facets:
        resolved_primary.extend(["metadata", "store_summary", "about"])
    variants = tuple(_query_variants(question, intent, claims))
    return SearchSpec(
        question=question.strip(),
        intent=intent,
        game_keys=resolved_game_keys,
        primary_sections=tuple(dict.fromkeys(resolved_primary)),
        secondary_sections=tuple(
            section for section in dict.fromkeys(resolved_secondary) if section not in resolved_primary
        ),
        query_variants=variants,
        claims=tuple(_dedupe_claims(claims)),
        facets=facets,
        time_requirement="after_update" if intent == "after_update" else ("recent" if recent else "none"),
        recommendation=recommendation,
    )


def evaluate_evidence_coverage(
    spec: SearchSpec,
    results: Sequence[SearchResult],
) -> EvidenceCoverageReport:
    rows: list[ClaimCoverage] = []
    for claim in spec.claims:
        evidence: list[int] = []
        best_score = 0.0
        best_missing: tuple[str, ...] = ("section",)
        for position, result in enumerate(results, start=1):
            score, missing = _claim_result_score(claim, result)
            if score > best_score:
                best_score, best_missing = score, missing
            if score >= 0.999:
                evidence.append(result.rank or position)
        supported = len(evidence) >= claim.min_evidence
        rows.append(
            ClaimCoverage(
                claim.claim_id,
                claim.text,
                supported,
                round(1.0 if supported else best_score, 4),
                tuple(evidence),
                () if supported else best_missing,
            )
        )
    supported_count = sum(row.supported for row in rows)
    ratio = supported_count / len(rows) if rows else 1.0
    return EvidenceCoverageReport(len(rows), supported_count, round(ratio, 4), tuple(rows))


def _claim_result_score(claim: EvidenceClaim, result: SearchResult) -> tuple[float, tuple[str, ...]]:
    metadata = result.document.metadata
    section_ok = str(metadata.get("section") or "") in claim.required_sections
    game_ok = not claim.required_game_keys or str(metadata.get("game_key") or "") in claim.required_game_keys
    date_ok = not claim.require_date or bool(
        metadata.get("source_date") or metadata.get("patch_date") or metadata.get("price_collected_at")
    )
    haystack = _searchable_result_text(result)
    keyword_ok = not claim.keywords or any(keyword.casefold() in haystack for keyword in claim.keywords)
    checks = (("section", section_ok), ("game", game_ok), ("date", date_ok), ("keyword", keyword_ok))
    missing = tuple(name for name, passed in checks if not passed)
    return sum(passed for _, passed in checks) / len(checks), missing


def _searchable_result_text(result: SearchResult) -> str:
    metadata = result.document.metadata
    values = [result.document.page_content]
    for key in (
        "item_title", "steam_tags_normalized", "steam_genres_normalized", "combat_facets",
        "perspective_facets", "dimension_facets", "playstyle_facets", "change_direction",
        "sentiment", "price_currency", "price_discount_percent", "price_final",
        "positive_ratio_delta_pp", "before_positive_ratio", "after_positive_ratio",
    ):
        values.append(f"{key} {metadata.get(key) or ''}")
    return " ".join(values).casefold()


def _facet_keywords(value: str) -> list[str]:
    aliases = {
        "turn_based": ["turn_based", "turn-based", "turn based", "턴제"],
        "real_time": ["real_time", "real-time", "real time", "실시간"],
        "third_person": ["third_person", "third person", "3인칭"],
        "first_person": ["first_person", "first person", "1인칭"],
        "top_down": ["top_down", "top-down", "탑다운"],
        "isometric": ["isometric", "아이소메트릭", "쿼터뷰"],
        "2_5d": ["2_5d", "2.5d"],
    }
    return aliases.get(value, [value, value.replace("_", " ")])


def _query_variants(question: str, intent: str, claims: Sequence[EvidenceClaim]) -> list[str]:
    variants = [question.strip()]
    for claim in claims:
        suffix = " ".join(claim.keywords[:4]) or " ".join(claim.required_sections)
        candidate = f"{question.strip()} {claim.text} {suffix}".strip()
        if candidate not in variants:
            variants.append(candidate)
    if intent == "after_update":
        variants.append(f"{question.strip()} patch date before after reviews sentiment")
    return variants


def _dedupe_claims(claims: Sequence[EvidenceClaim]) -> list[EvidenceClaim]:
    seen: set[str] = set()
    output: list[EvidenceClaim] = []
    for claim in claims:
        if claim.claim_id not in seen:
            seen.add(claim.claim_id)
            output.append(claim)
    return output
