"""Spoiler scoping applied to retrieval, not only to wording.

기획안 §8:

    스포일러는 답변 문구만 조심해서 해결하지 않는다. 검색할 자료 자체를
    진행도·퀘스트·노출 수준에 따라 제한하고, 결과의 제목·이미지·인용문도
    검사한다. 진행도가 불분명한 상태에서 스토리 정보가 필요한 질문이면 짧게
    확인한다. 확인된 스포일러 구분이 없는 공략 자료는 안전한 범위가 확인되기
    전까지 상세 답변에 쓰지 않는다.

이 모듈은 검색 결과를 실제로 걸러내고, 걸러낸 이유를 남긴다. 차단 사유는
평가 로그(§14.2 '스포일러' 축)에서 그대로 사용한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from steam_rag.common.models import SearchResult
from steam_rag.game_expert.support_scope import GameExpertProfile, Milestone


#: 스토리 노출 위험이 큰 문서 구간.
STORY_SECTIONS = frozenset({"story", "plot", "ending", "narrative", "walkthrough", "quest"})

#: 스포일러 구분이 확인된 출처 유형. 그 외 공략 자료는 상세 답변에 쓰지 않는다.
SPOILER_LABELED_SOURCE_TYPES = frozenset(
    {"steam_official", "steam_store", "steam_news", "steam_corpus", "expert_verified"}
)

STORY_QUESTION_PATTERN = re.compile(
    r"스토리|이야기|결말|엔딩|왜\s|정체|배신|진엔딩|비밀|누구(?:야|인가|였)|플롯|서사"
)


@dataclass(frozen=True, slots=True)
class SpoilerDecision:
    allowed: bool
    reason: str = ""
    milestone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason, "milestone": self.milestone}


@dataclass(slots=True)
class SpoilerPolicy:
    """Resolved spoiler rules for one user, one game, and one playthrough."""

    level: str
    progress_order: int
    progress_label: str
    milestones: tuple[Milestone, ...] = ()
    blocked: list[dict[str, Any]] = field(default_factory=list)

    @property
    def progress_known(self) -> bool:
        return self.progress_order > 0

    def allowed_order(self, *, story_sensitive: bool) -> int:
        if self.level == "all":
            return 10**6
        if story_sensitive:
            # 스토리 자료는 'no_spoiler'에서 전면 차단하고, 진행도가 없으면
            # 'progress'에서도 열지 않는다.
            return 0 if self.level == "no_spoiler" else self.progress_order
        # "스포일러 없이 초반에 알아야 할 것만"은 진행도가 없어도 답할 수 있어야
        # 하므로, 스토리 비중이 없는 구간 자료는 첫 구간까지 허용한다(§4.4 예시).
        return max(self.progress_order, 1)

    def classify(self, text: str) -> Milestone | None:
        matched = [item for item in self.milestones if item.matches(text)]
        return max(matched, key=lambda item: item.order) if matched else None

    def check_document(self, metadata: dict[str, Any], content: str) -> SpoilerDecision:
        """Decide whether one retrieved document may be used."""

        if self.level == "all":
            return SpoilerDecision(True)

        declared = str(metadata.get("spoiler_level") or "").strip().casefold()
        if declared in {"heavy", "ending", "late"}:
            return SpoilerDecision(
                False, "문서가 후반 스포일러로 표시돼 있습니다.", declared
            )

        source_type = str(metadata.get("source_type") or "steam_corpus").strip()
        labeled = bool(metadata.get("spoiler_labeled")) or source_type in SPOILER_LABELED_SOURCE_TYPES
        section = str(metadata.get("section") or "").strip().casefold()
        if not labeled and section in STORY_SECTIONS:
            return SpoilerDecision(
                False,
                "스포일러 구분이 확인되지 않은 공략 자료라 상세 답변에 사용하지 않았습니다.",
            )

        title = str(metadata.get("item_title") or "")
        milestone = self.classify(f"{title}\n{content[:1200]}")
        if milestone is None:
            if self.level == "no_spoiler" and section in STORY_SECTIONS:
                return SpoilerDecision(False, "스포일러 없이 답하기로 설정된 상태의 스토리 문서입니다.")
            return SpoilerDecision(True)

        limit = self.allowed_order(story_sensitive=milestone.story_sensitive)
        if milestone.order > limit:
            if not self.progress_known and milestone.story_sensitive:
                reason = "진행도가 확인되지 않아 이후 구간 자료를 사용하지 않았습니다."
            else:
                reason = f"현재 진행 구간({self.progress_label or '미확인'}) 이후 내용입니다."
            return SpoilerDecision(False, reason, milestone.label)
        return SpoilerDecision(True, milestone=milestone.label)

    def screen_text(self, text: str) -> list[str]:
        """Return milestone labels that would leak if ``text`` were shown."""

        leaks: list[str] = []
        for milestone in self.milestones:
            if not milestone.matches(text):
                continue
            if milestone.order > self.allowed_order(story_sensitive=milestone.story_sensitive):
                leaks.append(milestone.label)
        return leaks

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "progress_order": self.progress_order,
            "progress_label": self.progress_label,
            "progress_known": self.progress_known,
            "blocked": self.blocked,
        }


def build_spoiler_policy(
    profile: GameExpertProfile | None,
    *,
    spoiler_level: str,
    progress: str,
) -> SpoilerPolicy:
    """Resolve the user's progress text into an ordered milestone."""

    milestones = profile.milestones if profile else ()
    matched = None
    if progress.strip() and milestones:
        candidates = [item for item in milestones if item.matches(progress)]
        matched = max(candidates, key=lambda item: item.order) if candidates else None
    return SpoilerPolicy(
        level=spoiler_level if spoiler_level in {"no_spoiler", "progress", "all"} else "no_spoiler",
        progress_order=matched.order if matched else 0,
        progress_label=matched.label if matched else progress.strip(),
        milestones=tuple(milestones),
    )


def filter_results(
    policy: SpoilerPolicy,
    results: Sequence[SearchResult],
) -> tuple[list[SearchResult], list[dict[str, Any]]]:
    """Drop documents the policy does not allow and report why."""

    allowed: list[SearchResult] = []
    blocked: list[dict[str, Any]] = []
    for result in results:
        decision = policy.check_document(result.document.metadata, result.document.page_content)
        if decision.allowed:
            allowed.append(result)
            continue
        blocked.append(
            {
                "title": str(result.document.metadata.get("item_title") or "")[:120],
                "section": str(result.document.metadata.get("section") or ""),
                "reason": decision.reason,
                "milestone": decision.milestone,
            }
        )
    for rank, result in enumerate(allowed, start=1):
        result.rank = rank
    policy.blocked = blocked
    return allowed, blocked


def needs_progress_confirmation(question: str, policy: SpoilerPolicy) -> bool:
    """§8: story question + unknown progress means ask a short question first."""

    if policy.level == "all" or policy.progress_known:
        return False
    return bool(STORY_QUESTION_PATTERN.search(question))


def redact_leaks(text: str, policy: SpoilerPolicy) -> tuple[str, list[str]]:
    """Screen a generated answer's own wording before it reaches the user."""

    leaks = policy.screen_text(text)
    if not leaks:
        return text, []
    notice = (
        "\n\n※ 허용한 스포일러 범위를 넘는 내용("
        + ", ".join(sorted(set(leaks)))
        + ")은 제외했습니다. 더 보려면 스포일러 설정을 바꿔 주세요."
    )
    return text + notice, sorted(set(leaks))


def spoiler_notice(policy: SpoilerPolicy) -> str:
    if policy.level == "all":
        return "스포일러 제한 없이 답했습니다."
    if not policy.progress_known:
        return "진행도가 확인되지 않아 스토리 관련 자료는 사용하지 않았습니다."
    return f"현재 진행 구간({policy.progress_label})까지의 자료만 사용했습니다."


def milestone_labels(milestones: Iterable[Milestone]) -> list[str]:
    return [item.label for item in sorted(milestones, key=lambda value: value.order)]
