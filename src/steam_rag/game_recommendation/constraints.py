"""Three-state judgement of recommendation conditions.

기획안 §4.1 / §7 요구사항을 코드로 옮긴 모듈이다.

* 필수 조건의 판정은 **충족 / 위반 / 미확인**으로 나눈다.
* 전투 방식이 미확인인 게임을 조건 충족으로 취급하지 않는다.
* 확인된 사실, 자료 해석, 아직 확인하지 못한 항목을 카드에서 구분해 보여준다.

판정 규칙은 프로필 데이터의 존재 여부만으로 결정한다.

* 요청 값이 프로필의 해당 항목에 있으면 ``satisfied``
* 항목에 값이 있는데 요청 값이 없으면 ``violated``
* 항목 자체가 비어 있으면 ``unverified``

``unverified``는 결코 충족으로 승격되지 않으며, 후보에서 자동으로 제거되지도
않는다. 대신 카드의 "선택 전 확인"과 "정보 상태"에 그대로 드러난다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from steam_rag.game_metadata.playstyle import coerce_list

if TYPE_CHECKING:  # pragma: no cover - 순환 import를 피하려고 타입 검사에서만 읽는다.
    from steam_rag.game_recommendation.query_parser import RecommendationQuery


SATISFIED = "satisfied"
VIOLATED = "violated"
UNVERIFIED = "unverified"

CONFIRMED = "confirmed"
INTERPRETED = "interpreted"
UNKNOWN = "unknown"

MUST = "must"
EXCLUDE = "exclude"
PREFER = "prefer"

#: 공식 Steam 데이터에서 나온 근거. 이 출처만 "확인된 사실"로 취급한다.
OFFICIAL_EVIDENCE_SOURCES = frozenset(
    {"steam_genre", "steam_category", "steam_popular_user_tag"}
)

FACET_FIELD_BY_CONDITION = {
    "combat": "combat_facets",
    "perspective": "perspective_facets",
    "dimension": "dimension_facets",
    "playstyle": "playstyle_facets",
}

CONDITION_GROUP_LABELS = {
    "genres": "공식 장르",
    "categories": "공식 카테고리",
    "required_tags": "인기 태그",
    "combat": "전투 방식",
    "perspective": "화면 시점",
    "dimension": "표현 차원",
    "playstyle": "플레이 성향",
}

VALUE_LABELS = {
    "turn_based": "턴제 전투",
    "real_time": "실시간 전투",
    "direct_control": "직접 조작",
    "command_based": "명령식 전투",
    "auto_combat": "자동 전투",
    "party_based": "파티 기반 전투",
    "boss_focused": "보스 중심 전투",
    "tactical": "전술 전투",
    "melee": "근접 전투",
    "ranged": "원거리 전투",
    "shooter": "슈팅",
    "stealth": "잠입",
    "first_person": "1인칭",
    "third_person": "3인칭",
    "side_view": "횡스크롤 시점",
    "top_down": "탑다운 시점",
    "isometric": "아이소메트릭 시점",
    "2d": "2D",
    "2_5d": "2.5D",
    "3d": "3D",
    "vr": "VR",
    "story_rich": "스토리 비중이 큼",
    "choices_matter": "선택이 결과를 바꿈",
    "open_world": "오픈 월드",
    "exploration": "탐험",
    "co_op": "협동 플레이",
    "multiplayer": "멀티플레이",
    "online_multiplayer": "온라인 멀티플레이",
    "pvp": "PvP",
    "platforming": "플랫포밍",
    "survival": "생존",
    "crafting": "제작",
    "hunting": "사냥",
    "metroidvania": "메트로배니아",
    "character_progression": "캐릭터 성장",
    "souls_like": "소울라이크",
    "roguelike": "로그라이크",
    "roguelite": "로그라이트",
    # Steam 인기 태그의 정규화 값도 한국어로 보여준다.
    "turn_based_combat": "턴제 전투",
    "turn_based_tactics": "턴제 전술",
    "real_time_combat": "실시간 전투",
    "action_rpg": "액션 RPG",
    "action_adventure": "액션 어드벤처",
    "action_roguelike": "액션 로그라이크",
    "party_based_rpg": "파티 기반 RPG",
    "hack_and_slash": "핵 앤 슬래시",
    "online_co_op": "온라인 협동",
    "online_pvp": "온라인 PvP",
    "platformer": "플랫포머",
    "side_scroller": "횡스크롤",
    "difficult": "높은 난도",
    "action": "액션",
    "rpg": "RPG",
    "adventure": "어드벤처",
    "strategy": "전략",
    "simulation": "시뮬레이션",
    "indie": "인디",
    "singleplayer": "싱글 플레이",
}


def condition_label(group: str, value: str) -> str:
    """Return the Korean phrase shown on a candidate card."""

    readable = VALUE_LABELS.get(value, value.replace("_", " "))
    group_label = CONDITION_GROUP_LABELS.get(group)
    if group_label is None:
        return readable
    return f"{group_label}: {readable}"


@dataclass(frozen=True, slots=True)
class ConditionVerdict:
    """One condition judged against one candidate profile."""

    condition_id: str
    kind: str
    group: str
    value: str
    label: str
    verdict: str
    confidence: str = UNKNOWN
    evidence: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "kind": self.kind,
            "group": self.group,
            "value": self.value,
            "label": self.label,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "note": self.note,
        }


@dataclass(slots=True)
class CandidateConstraintReport:
    """Per-candidate 충족·위반·미확인 판정과 카드 표시용 요약."""

    appid: int
    name: str
    verdicts: list[ConditionVerdict] = field(default_factory=list)
    information_status: dict[str, Any] = field(default_factory=dict)

    def _by(self, kind: str, verdict: str) -> list[ConditionVerdict]:
        return [item for item in self.verdicts if item.kind == kind and item.verdict == verdict]

    @property
    def must_violated(self) -> list[ConditionVerdict]:
        return self._by(MUST, VIOLATED) + self._by(EXCLUDE, VIOLATED)

    @property
    def must_unverified(self) -> list[ConditionVerdict]:
        return self._by(MUST, UNVERIFIED) + self._by(EXCLUDE, UNVERIFIED)

    @property
    def must_satisfied(self) -> list[ConditionVerdict]:
        return self._by(MUST, SATISFIED) + self._by(EXCLUDE, SATISFIED)

    @property
    def passes(self) -> bool:
        """True when no required condition is known to be violated."""

        return not self.must_violated

    @property
    def fully_verified(self) -> bool:
        """True only when every required condition is confirmed satisfied."""

        return self.passes and not self.must_unverified

    @property
    def status(self) -> str:
        if self.must_violated:
            return VIOLATED
        if self.must_unverified:
            return UNVERIFIED
        return SATISFIED

    @property
    def fit_reasons(self) -> list[str]:
        """§4.2 '잘 맞는 점'."""

        return [item.label for item in self.must_satisfied if item.kind == MUST]

    @property
    def checks_before_choosing(self) -> list[str]:
        """§4.2 '선택 전 확인'."""

        checks = [f"{item.label} 미확인" for item in self.must_unverified]
        checks.extend(
            f"{item.label} 근거가 스토어 설명 해석뿐"
            for item in self.must_satisfied
            if item.confidence == INTERPRETED
        )
        return checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "appid": self.appid,
            "name": self.name,
            "status": self.status,
            "passes": self.passes,
            "fully_verified": self.fully_verified,
            "satisfied": [item.to_dict() for item in self.must_satisfied],
            "violated": [item.to_dict() for item in self.must_violated],
            "unverified": [item.to_dict() for item in self.must_unverified],
            "preferences": [item.to_dict() for item in self.verdicts if item.kind == PREFER],
            "fit_reasons": self.fit_reasons,
            "checks_before_choosing": self.checks_before_choosing,
            "information_status": self.information_status,
        }


def evaluate_candidate_conditions(
    profile: dict[str, Any],
    query: "RecommendationQuery",
) -> CandidateConstraintReport:
    """Judge every condition in ``query`` against one Steam profile."""

    normalized = query.normalized()
    report = CandidateConstraintReport(
        appid=_int(profile.get("appid")),
        name=str(profile.get("name") or profile.get("game_key") or ""),
    )
    evidence_index = _facet_evidence_index(profile)
    available = _available_values(profile)

    for group in ("genres", "categories", "required_tags"):
        for value in getattr(normalized, group):
            report.verdicts.append(_taxonomy_verdict(group, value, available))
    for group, facet_field in FACET_FIELD_BY_CONDITION.items():
        for value in getattr(normalized, group):
            report.verdicts.append(
                _facet_verdict(group, value, facet_field, profile, evidence_index)
            )
    for value in normalized.excluded_conditions:
        report.verdicts.append(_exclusion_verdict(value, profile, available))

    report.verdicts.extend(_state_verdicts(profile, normalized))
    report.information_status = _information_status(profile, report)
    return report


def _taxonomy_verdict(
    group: str,
    value: str,
    available: dict[str, set[str]],
) -> ConditionVerdict:
    pool = available[group]
    label = condition_label(group, value)
    if not pool:
        return ConditionVerdict(
            condition_id=f"{group}:{value}",
            kind=MUST,
            group=group,
            value=value,
            label=label,
            verdict=UNVERIFIED,
            note="Steam 프로필에 해당 분류 정보가 없어 확인하지 못했습니다.",
        )
    if value in pool:
        return ConditionVerdict(
            condition_id=f"{group}:{value}",
            kind=MUST,
            group=group,
            value=value,
            label=label,
            verdict=SATISFIED,
            confidence=CONFIRMED,
            evidence=("steam_official_taxonomy",),
        )
    return ConditionVerdict(
        condition_id=f"{group}:{value}",
        kind=MUST,
        group=group,
        value=value,
        label=label,
        verdict=VIOLATED,
        confidence=CONFIRMED,
        evidence=("steam_official_taxonomy",),
        note="Steam 분류에 해당 값이 없습니다.",
    )


def _facet_verdict(
    group: str,
    value: str,
    facet_field: str,
    profile: dict[str, Any],
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> ConditionVerdict:
    label = condition_label(group, value)
    confirmed = set(coerce_list(profile.get(facet_field)))
    inferred = profile.get("inferred_facets")
    soft = set(coerce_list(inferred.get(facet_field))) if isinstance(inferred, dict) else set()
    condition_id = f"{group}:{value}"

    if value in confirmed:
        sources = _evidence_sources(evidence_index, facet_field, value)
        official = sorted(sources & OFFICIAL_EVIDENCE_SOURCES)
        if official:
            return ConditionVerdict(
                condition_id=condition_id,
                kind=MUST,
                group=group,
                value=value,
                label=label,
                verdict=SATISFIED,
                confidence=CONFIRMED,
                evidence=tuple(official),
            )
        if sources:
            return ConditionVerdict(
                condition_id=condition_id,
                kind=MUST,
                group=group,
                value=value,
                label=label,
                verdict=SATISFIED,
                confidence=INTERPRETED,
                evidence=tuple(sorted(sources)),
                note="스토어 설명이나 리뷰 문장에서 해석한 값입니다.",
            )
        return ConditionVerdict(
            condition_id=condition_id,
            kind=MUST,
            group=group,
            value=value,
            label=label,
            verdict=SATISFIED,
            confidence=INTERPRETED,
            evidence=("playstyle_profile",),
            note="출처별 근거 기록이 없는 이전 버전 프로필입니다.",
        )

    if confirmed:
        confirmed_labels = ", ".join(VALUE_LABELS.get(item, item) for item in sorted(confirmed))
        return ConditionVerdict(
            condition_id=condition_id,
            kind=MUST,
            group=group,
            value=value,
            label=label,
            verdict=VIOLATED,
            confidence=CONFIRMED,
            evidence=tuple(f"{facet_field}:{item}" for item in sorted(confirmed)),
            note=f"확인된 값은 {confirmed_labels}입니다.",
        )

    note = "확인된 근거가 없습니다."
    if value in soft:
        note = "스토어 설명에서 추정만 되고 공식 근거로 확인되지 않았습니다."
    return ConditionVerdict(
        condition_id=condition_id,
        kind=MUST,
        group=group,
        value=value,
        label=label,
        verdict=UNVERIFIED,
        note=note,
    )


def _exclusion_verdict(
    value: str,
    profile: dict[str, Any],
    available: dict[str, set[str]],
) -> ConditionVerdict:
    readable = VALUE_LABELS.get(value, value.replace("_", " "))
    label = f"제외 조건: {readable}"
    searchable = set(coerce_list(profile.get("searchable_terms")))
    for pool in available.values():
        searchable |= pool
    for facet_field in FACET_FIELD_BY_CONDITION.values():
        searchable |= set(coerce_list(profile.get(facet_field)))
    if value in searchable:
        return ConditionVerdict(
            condition_id=f"exclude:{value}",
            kind=EXCLUDE,
            group="excluded_conditions",
            value=value,
            label=label,
            verdict=VIOLATED,
            confidence=CONFIRMED,
            evidence=("steam_official_taxonomy",),
            note="사용자가 제외한 요소가 확인됐습니다.",
        )
    if not searchable:
        return ConditionVerdict(
            condition_id=f"exclude:{value}",
            kind=EXCLUDE,
            group="excluded_conditions",
            value=value,
            label=label,
            verdict=UNVERIFIED,
            note="분류 정보가 없어 제외 조건을 확인하지 못했습니다.",
        )
    return ConditionVerdict(
        condition_id=f"exclude:{value}",
        kind=EXCLUDE,
        group="excluded_conditions",
        value=value,
        label=label,
        verdict=SATISFIED,
        confidence=CONFIRMED,
        evidence=("steam_official_taxonomy",),
    )


def _state_verdicts(
    profile: dict[str, Any],
    query: "RecommendationQuery",
) -> list[ConditionVerdict]:
    """Price, sale, release, and rating conditions."""

    verdicts: list[ConditionVerdict] = []
    price = profile.get("price") if isinstance(profile.get("price"), dict) else {}
    final_price = price.get("final")
    currency = price.get("currency")

    if query.price_max_krw is not None:
        label = f"가격: {query.price_max_krw:,}원 이하"
        if currency != "KRW" or not isinstance(final_price, (int, float)):
            verdicts.append(
                _state_verdict(
                    "price_max_krw", label, UNVERIFIED, note="원화 가격을 확인하지 못했습니다."
                )
            )
        elif final_price <= query.price_max_krw * 100:
            verdicts.append(
                _state_verdict("price_max_krw", label, SATISFIED, evidence=("steam_price",))
            )
        else:
            verdicts.append(
                _state_verdict(
                    "price_max_krw",
                    label,
                    VIOLATED,
                    evidence=("steam_price",),
                    note=f"수집 시점 가격은 {int(final_price / 100):,}원입니다.",
                )
            )

    if query.sale_required:
        label = "현재 할인 중"
        if not price:
            verdicts.append(
                _state_verdict(
                    "sale_required", label, UNVERIFIED, note="가격 정보를 확인하지 못했습니다."
                )
            )
        elif price.get("is_free") is True:
            verdicts.append(
                _state_verdict(
                    "sale_required",
                    label,
                    VIOLATED,
                    evidence=("steam_price",),
                    note="무료 게임입니다.",
                )
            )
        elif int(price.get("discount_percent") or 0) > 0:
            verdicts.append(
                _state_verdict("sale_required", label, SATISFIED, evidence=("steam_price",))
            )
        else:
            verdicts.append(
                _state_verdict(
                    "sale_required",
                    label,
                    VIOLATED,
                    evidence=("steam_price",),
                    note="할인 중이 아닙니다.",
                )
            )

    coming_soon = profile.get("release_coming_soon")
    if query.upcoming_required:
        label = "출시 예정작"
        if coming_soon is True:
            verdicts.append(
                _state_verdict("upcoming_required", label, SATISFIED, evidence=("steam_release",))
            )
        elif coming_soon is False:
            verdicts.append(
                _state_verdict(
                    "upcoming_required",
                    label,
                    VIOLATED,
                    evidence=("steam_release",),
                    note="이미 출시된 게임입니다.",
                )
            )
        else:
            verdicts.append(
                _state_verdict(
                    "upcoming_required", label, UNVERIFIED, note="출시 상태를 확인하지 못했습니다."
                )
            )

    if query.currently_playable_required and not query.upcoming_required:
        label = "지금 플레이 가능"
        if coming_soon is True:
            verdicts.append(
                _state_verdict(
                    "currently_playable_required",
                    label,
                    VIOLATED,
                    evidence=("steam_release",),
                    note="아직 출시되지 않았습니다.",
                )
            )
        elif coming_soon is False:
            verdicts.append(
                _state_verdict(
                    "currently_playable_required",
                    label,
                    SATISFIED,
                    evidence=("steam_release",),
                )
            )
        else:
            verdicts.append(
                _state_verdict(
                    "currently_playable_required",
                    label,
                    UNVERIFIED,
                    note="출시 상태를 확인하지 못했습니다.",
                )
            )

    review = profile.get("recent_review_summary")
    ratio = review.get("positive_ratio") if isinstance(review, dict) else None
    if query.recent_rating_required:
        label = "최근 사용자 평가 확인"
        if isinstance(ratio, (int, float)):
            verdicts.append(
                ConditionVerdict(
                    condition_id="prefer:recent_rating",
                    kind=PREFER,
                    group="recent_rating_required",
                    value="recent_rating",
                    label=label,
                    verdict=SATISFIED,
                    confidence=CONFIRMED,
                    evidence=("steam_recent_reviews",),
                    note=f"최근 표본 긍정률 {ratio * 100:.0f}%",
                )
            )
        else:
            verdicts.append(
                ConditionVerdict(
                    condition_id="prefer:recent_rating",
                    kind=PREFER,
                    group="recent_rating_required",
                    value="recent_rating",
                    label=label,
                    verdict=UNVERIFIED,
                    note="최근 리뷰 표본을 아직 수집하지 못했습니다.",
                )
            )
    if query.after_update_required:
        verdicts.append(
            ConditionVerdict(
                condition_id="prefer:after_update",
                kind=PREFER,
                group="after_update_required",
                value="after_update",
                label="업데이트 이후 반응 확인",
                verdict=UNVERIFIED,
                note="패치 전후 표본 비교는 상세 분석 단계에서 확인합니다.",
            )
        )
    return verdicts


def _state_verdict(
    condition_id: str,
    label: str,
    verdict: str,
    *,
    evidence: Sequence[str] = (),
    note: str = "",
) -> ConditionVerdict:
    return ConditionVerdict(
        condition_id=f"state:{condition_id}",
        kind=MUST,
        group=condition_id,
        value=condition_id,
        label=label,
        verdict=verdict,
        confidence=CONFIRMED if evidence else UNKNOWN,
        evidence=tuple(evidence),
        note=note,
    )


def _available_values(profile: dict[str, Any]) -> dict[str, set[str]]:
    tags = _profile_tags(profile)
    genres = set(coerce_list(profile.get("steam_genres_normalized")))
    categories = set(coerce_list(profile.get("steam_categories_normalized")))
    return {
        "genres": genres | tags if (genres or tags) else set(),
        "categories": categories | tags if (categories or tags) else set(),
        "required_tags": tags,
    }


def _profile_tags(profile: dict[str, Any]) -> set[str]:
    ranked = {
        str(item.get("normalized"))
        for item in profile.get("popular_user_tags") or []
        if isinstance(item, dict) and item.get("normalized")
    }
    return ranked or set(coerce_list(profile.get("steam_tags_normalized")))


def _facet_evidence_index(
    profile: dict[str, Any],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in profile.get("facet_evidence") or []:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("facet_type") or ""), str(row.get("facet") or ""))
        index.setdefault(key, []).append(row)
    return index


def _evidence_sources(
    index: dict[tuple[str, str], list[dict[str, Any]]],
    facet_field: str,
    value: str,
) -> set[str]:
    return {
        str(row.get("source_type") or "")
        for row in index.get((facet_field, value), [])
        if row.get("source_type")
    }


def _information_status(
    profile: dict[str, Any],
    report: CandidateConstraintReport,
) -> dict[str, Any]:
    """§4.2 '정보 상태': 확인한 출처와 시점, 아직 확인하지 못한 항목."""

    price = profile.get("price") if isinstance(profile.get("price"), dict) else {}
    review = profile.get("recent_review_summary")
    review = review if isinstance(review, dict) else {}
    checked: list[dict[str, Any]] = []
    for source, checked_at in (
        ("Steam 상품 정보", profile.get("collected_at")),
        ("Steam 인기 태그", profile.get("popular_tags_collected_at")),
        ("Steam 가격", price.get("collected_at") or profile.get("price_collected_at")),
        ("Steam 최근 리뷰 표본", review.get("collected_at")),
    ):
        if checked_at:
            checked.append({"source": source, "checked_at": str(checked_at)})
    return {
        "checked_sources": checked,
        "unverified_items": [item.label for item in report.must_unverified],
        "interpreted_items": [
            item.label for item in report.must_satisfied if item.confidence == INTERPRETED
        ],
        "review_sample_size": review.get("sample_size"),
    }


def summarize_constraint_gate(
    reports: Iterable[CandidateConstraintReport],
) -> dict[str, Any]:
    """Aggregate view used by the answer layer and by evaluation logs."""

    rows = list(reports)
    verified = [item for item in rows if item.fully_verified]
    partial = [item for item in rows if item.passes and not item.fully_verified]
    rejected = [item for item in rows if not item.passes]
    return {
        "evaluated": len(rows),
        "fully_verified": [item.appid for item in verified],
        "unverified_conditions": {
            item.appid: [verdict.label for verdict in item.must_unverified] for item in partial
        },
        "rejected": {
            item.appid: [verdict.label for verdict in item.must_violated] for item in rejected
        },
        "status": "verified" if verified else ("unverified_only" if partial else "no_match"),
    }


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
