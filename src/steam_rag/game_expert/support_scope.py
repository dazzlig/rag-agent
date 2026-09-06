"""Per-game expert configuration loaded by one shared expert implementation.

기획안 §6.2: "게임이 1만 개라고 해서 모델이나 실행 중인 프로세스 1만 개를
유지할 필요는 없다. 공통 전문가 코드에 게임별 설정과 자료를 불러오는 방식을
제안한다."

따라서 게임별 전문가는 클래스가 아니라 **데이터**다. 이 모듈은
``data/game_experts/*.json``에 저장된 게임 프로필과 지원 범위를 읽어
:class:`GameExpertProfile` 로 만든다. 지원 범위 밖의 질문은 같은 수준의 답을
약속하지 않고 그 사실을 먼저 알린다(§9.1).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_EXPERT_DIR = Path("data/game_experts")

#: §9.1에서 정한 초기 지원 주제. 각 게임은 이 중 실제로 검증한 것만 선언한다.
SUPPORT_TOPICS = {
    "system": "핵심 시스템 설명",
    "early_guide": "초반 진행 가이드",
    "boss": "대표 난관 공략",
    "build": "장비와 빌드",
    "progression": "성장과 자원 관리",
    "update": "업데이트 영향",
}


def has_final_consonant(word: str) -> bool:
    """True when the last Korean syllable ends with a 받침."""

    for char in reversed(word.strip()):
        if "가" <= char <= "힣":
            return (ord(char) - 0xAC00) % 28 != 0
        if char.isalnum():
            # 숫자와 라틴 문자는 읽는 방식이 갈리므로 받침 없음으로 둔다.
            return False
    return False


def with_topic_particle(word: str) -> str:
    """Attach 은/는 so generated Korean sentences read correctly."""

    return f"{word}{'은' if has_final_consonant(word) else '는'}"


@dataclass(frozen=True, slots=True)
class KeySystem:
    """One mechanic the expert can explain, with the source that confirmed it."""

    system_id: str
    name: str
    summary: str
    source: str = ""
    verified_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "name": self.name,
            "summary": self.summary,
            "source": self.source,
            "verified_at": self.verified_at,
        }


@dataclass(frozen=True, slots=True)
class Milestone:
    """One ordered progress point used for spoiler scoping (§8)."""

    order: int
    milestone_id: str
    label: str
    keywords: tuple[str, ...] = ()
    story_sensitive: bool = True

    def matches(self, text: str) -> bool:
        lowered = text.casefold()
        if self.milestone_id.casefold() in lowered or self.label.casefold() in lowered:
            return True
        return any(keyword.casefold() in lowered for keyword in self.keywords if keyword)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "milestone_id": self.milestone_id,
            "label": self.label,
            "keywords": list(self.keywords),
            "story_sensitive": self.story_sensitive,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeSource:
    """§10.1 '이용 가능한 공략·위키·제작자 자료'의 이용 조건 기록."""

    title: str
    url: str = ""
    usage: str = "unverified"
    version: str = ""
    collected_at: str = ""
    spoiler_labeled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "usage": self.usage,
            "version": self.version,
            "collected_at": self.collected_at,
            "spoiler_labeled": self.spoiler_labeled,
        }


@dataclass(frozen=True, slots=True)
class SupportScope:
    """확인된 공략 주제·구간·버전과 마지막 검토 시점 (§6.2)."""

    topics: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    verified_version: str = ""
    verified_platforms: tuple[str, ...] = ()
    last_reviewed: str = ""
    out_of_scope_note: str = ""

    def covers_topic(self, topic: str) -> bool:
        return topic in self.topics

    def to_dict(self) -> dict[str, Any]:
        return {
            "topics": list(self.topics),
            "topic_labels": [SUPPORT_TOPICS.get(topic, topic) for topic in self.topics],
            "sections": list(self.sections),
            "verified_version": self.verified_version,
            "verified_platforms": list(self.verified_platforms),
            "last_reviewed": self.last_reviewed,
            "out_of_scope_note": self.out_of_scope_note,
        }


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    """Whether the expert promises verified quality for this request."""

    supported: bool
    topic: str
    reason: str
    verified_version: str = ""
    last_reviewed: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "topic": self.topic,
            "topic_label": SUPPORT_TOPICS.get(self.topic, self.topic),
            "reason": self.reason,
            "verified_version": self.verified_version,
            "last_reviewed": self.last_reviewed,
        }


@dataclass(frozen=True, slots=True)
class GameExpertProfile:
    """게임 ID, 별칭, 플랫폼·판본, 주요 시스템, 사용 가능한 도구 (§6.2)."""

    appid: int
    game_key: str
    name: str
    aliases: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    editions: tuple[str, ...] = ()
    key_systems: tuple[KeySystem, ...] = ()
    milestones: tuple[Milestone, ...] = ()
    knowledge_sources: tuple[KnowledgeSource, ...] = ()
    tools: tuple[str, ...] = ()
    support: SupportScope = field(default_factory=SupportScope)

    def matches_name(self, text: str) -> bool:
        normalized = _normalize(text)
        for candidate in (self.name, *self.aliases):
            token = _normalize(candidate)
            if token and token in normalized:
                return True
        return bool(re.search(rf"\b{self.appid}\b", text))

    def milestone_for(self, text: str) -> Milestone | None:
        """Return the furthest milestone mentioned in ``text``."""

        matched = [item for item in self.milestones if item.matches(text)]
        return max(matched, key=lambda item: item.order) if matched else None

    def decide_scope(self, topic: str) -> ScopeDecision:
        label = SUPPORT_TOPICS.get(topic, topic)
        if self.support.covers_topic(topic):
            return ScopeDecision(
                supported=True,
                topic=topic,
                reason=f"{with_topic_particle(label)} 검증된 지원 범위입니다.",
                verified_version=self.support.verified_version,
                last_reviewed=self.support.last_reviewed,
            )
        covered = ", ".join(SUPPORT_TOPICS.get(item, item) for item in self.support.topics)
        return ScopeDecision(
            supported=False,
            topic=topic,
            reason=(
                f"{with_topic_particle(label)} 아직 검증한 지원 범위가 아닙니다. "
                f"검증된 범위는 {covered or '없음'}입니다."
            ),
            verified_version=self.support.verified_version,
            last_reviewed=self.support.last_reviewed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "appid": self.appid,
            "game_key": self.game_key,
            "name": self.name,
            "aliases": list(self.aliases),
            "platforms": list(self.platforms),
            "editions": list(self.editions),
            "key_systems": [item.to_dict() for item in self.key_systems],
            "milestones": [item.to_dict() for item in self.milestones],
            "knowledge_sources": [item.to_dict() for item in self.knowledge_sources],
            "tools": list(self.tools),
            "support": self.support.to_dict(),
        }


class GameExpertRegistry:
    """Load the per-game configuration files once and resolve games by name."""

    def __init__(self, profiles: Sequence[GameExpertProfile]) -> None:
        self.profiles = list(profiles)
        self._by_appid = {profile.appid: profile for profile in self.profiles}

    @classmethod
    def load(cls, directory: Path = DEFAULT_EXPERT_DIR) -> "GameExpertRegistry":
        profiles: list[GameExpertProfile] = []
        if directory.exists():
            for path in sorted(directory.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                profile = build_expert_profile(payload)
                if profile is not None:
                    profiles.append(profile)
        return cls(profiles)

    def get(self, appid: int) -> GameExpertProfile | None:
        try:
            return self._by_appid.get(int(appid))
        except (TypeError, ValueError):
            return None

    def resolve(self, text: str) -> GameExpertProfile | None:
        for profile in self.profiles:
            if profile.matches_name(text):
                return profile
        return None

    def supported_appids(self) -> list[int]:
        return [profile.appid for profile in self.profiles]

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "appid": profile.appid,
                "name": profile.name,
                "topics": list(profile.support.topics),
                "verified_version": profile.support.verified_version,
                "last_reviewed": profile.support.last_reviewed,
            }
            for profile in self.profiles
        ]


def build_expert_profile(payload: dict[str, Any]) -> GameExpertProfile | None:
    try:
        appid = int(payload["appid"])
    except (KeyError, TypeError, ValueError):
        return None
    name = str(payload.get("name") or "").strip()
    if appid <= 0 or not name:
        return None
    support = payload.get("support") if isinstance(payload.get("support"), dict) else {}
    return GameExpertProfile(
        appid=appid,
        game_key=str(payload.get("game_key") or _normalize(name).replace(" ", "_")),
        name=name,
        aliases=_strings(payload.get("aliases")),
        platforms=_strings(payload.get("platforms")),
        editions=_strings(payload.get("editions")),
        key_systems=tuple(
            KeySystem(
                system_id=str(item.get("system_id") or item.get("name") or ""),
                name=str(item.get("name") or ""),
                summary=str(item.get("summary") or ""),
                source=str(item.get("source") or ""),
                verified_at=str(item.get("verified_at") or ""),
            )
            for item in payload.get("key_systems") or []
            if isinstance(item, dict) and item.get("name")
        ),
        milestones=tuple(
            sorted(
                (
                    Milestone(
                        order=int(item.get("order") or index),
                        milestone_id=str(item.get("milestone_id") or f"m{index}"),
                        label=str(item.get("label") or ""),
                        keywords=_strings(item.get("keywords")),
                        story_sensitive=bool(item.get("story_sensitive", True)),
                    )
                    for index, item in enumerate(payload.get("milestones") or [], start=1)
                    if isinstance(item, dict)
                ),
                key=lambda item: item.order,
            )
        ),
        knowledge_sources=tuple(
            KnowledgeSource(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                usage=str(item.get("usage") or "unverified"),
                version=str(item.get("version") or ""),
                collected_at=str(item.get("collected_at") or ""),
                spoiler_labeled=bool(item.get("spoiler_labeled", False)),
            )
            for item in payload.get("knowledge_sources") or []
            if isinstance(item, dict) and item.get("title")
        ),
        tools=_strings(payload.get("tools")),
        support=SupportScope(
            topics=_strings(support.get("topics")),
            sections=_strings(support.get("sections")),
            verified_version=str(support.get("verified_version") or ""),
            verified_platforms=_strings(support.get("verified_platforms")),
            last_reviewed=str(support.get("last_reviewed") or ""),
            out_of_scope_note=str(support.get("out_of_scope_note") or ""),
        ),
    )


def classify_topic(question: str) -> str:
    """Map a walkthrough question to one supported topic id."""

    lowered = question.casefold()
    if re.search(r"보스|막혔|못\s*깨|처치|공략법|패턴|클리어\s*방법", lowered):
        return "boss"
    # "시스템"의 '템'이 장비 질문으로 잘못 분류되지 않도록 앞 글자를 확인한다.
    if re.search(r"장비|무기|방어구|빌드|스킬\s*트리|특성|세팅|(?<!시스)템", lowered):
        return "build"
    if re.search(r"초반|처음|시작|입문|뭐부터|먼저\s*(?:할|해야)", lowered):
        return "early_guide"
    if re.search(r"업데이트|패치|버전|바뀐|변경", lowered):
        return "update"
    if re.search(r"성장|레벨|자원|파밍|경제|돈|재화", lowered):
        return "progression"
    return "system"


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", str(value).casefold()).strip()


def iter_topics(topics: Iterable[str]) -> list[str]:
    return [SUPPORT_TOPICS.get(topic, topic) for topic in topics]
