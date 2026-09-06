"""The shared per-game expert implementation.

기획안 §6.1 / §6.2 / §8.

하나의 실행 코드가 게임별 프로필·지원 범위·자료·사용자 상태를 불러와 전문가
역할을 수행한다. "당신은 이 게임의 전문가입니다"라는 프롬프트만으로 전문성을
확보했다고 보지 않는다(§6.2).

답변 원칙(§8):

1. 게임과 적용 범위를 식별한다.
2. 현재 상황에서 필요한 정보만 가져오고, 부족한 핵심 정보만 묻는다.
3. 확인된 시스템과 그로부터 제안하는 전술을 구분한다.
4. 실행 가능한 단위로 답한다.
5. 사용자가 시도한 방법과 결과를 상태에 반영한다.

기본 답변 순서는 **현재 상황 진단 → 바로 시도할 행동 → 필요한 이유 → 추가
힌트**다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from steam_rag.common.interfaces import AnswerGenerator, Embedder
from steam_rag.common.models import SearchResult
from steam_rag.game_expert.spoiler import (
    SpoilerPolicy,
    build_spoiler_policy,
    filter_results,
    needs_progress_confirmation,
    redact_leaks,
    spoiler_notice,
)
from steam_rag.game_expert.support_scope import (
    SUPPORT_TOPICS,
    GameExpertProfile,
    ScopeDecision,
    classify_topic,
)
from steam_rag.rag_search.hybrid_retriever import HybridTimeAwareRetriever, augment_query, detect_intent
from steam_rag.rag_search.reranker import Reranker
from steam_rag.user_workspace.store import GameState


#: 주제별로 답이 달라지는 상태 항목만 요구한다(§4.4, §8-2).
REQUIRED_STATE_FIELDS = {
    "boss": ("progress", "equipment"),
    "build": ("character_build",),
    "progression": ("progress",),
    "early_guide": (),
    "update": (),
    "system": (),
}

STATE_QUESTIONS = {
    "progress": "지금 어디까지 진행했는지 한 줄로 알려주실 수 있나요?",
    "equipment": "현재 사용 중인 주요 장비나 무기를 알려주시면 더 정확히 볼 수 있습니다.",
    "character_build": "선택한 직업이나 빌드 방향을 알려주세요.",
}

RETRY_PATTERN = re.compile(
    r"알려준\s*대로|말한\s*대로|시켰던\s*대로|해\s*봤(?:는데|지만)|"
    r"안\s*(?:됐|되|먹|통)|여전히|또\s*(?:죽|실패)|소용\s*없"
)

#: §8 재검토 대상. 실패했을 때 무엇을 다시 볼지 명시한다.
RETRY_ASSUMPTIONS = ("장비", "전략", "조작", "공략 버전")


@dataclass(slots=True)
class ExpertAnswer:
    """Structured expert output (§12).

    답변 외에 근거, 적용 버전, 미확인 사항, 추가 질문, 상태 변경 제안을
    구조화해 반환한다. 통합 에이전트는 이 결과를 전달할 뿐 사실이나 스포일러
    범위를 바꾸지 않는다.
    """

    answer: str
    scope: ScopeDecision
    evidence: list[SearchResult] = field(default_factory=list)
    applied_scope: dict[str, Any] = field(default_factory=dict)
    unverified: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    state_change_proposals: list[dict[str, Any]] = field(default_factory=list)
    spoiler: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    is_retry: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "scope": self.scope.to_dict(),
            "applied_scope": self.applied_scope,
            "unverified": self.unverified,
            "follow_up_questions": self.follow_up_questions,
            "state_change_proposals": self.state_change_proposals,
            "spoiler": self.spoiler,
            "trace": self.trace,
            "is_retry": self.is_retry,
        }


class GameExpertAgent:
    """One implementation shared by every supported game (§6.2)."""

    def __init__(
        self,
        profile: GameExpertProfile,
        retriever: HybridTimeAwareRetriever,
        embedder: Embedder,
        answer_generator: AnswerGenerator,
        *,
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        self.profile = profile
        self.retriever = retriever
        self.embedder = embedder
        self.answer_generator = answer_generator
        self.reranker = reranker
        self.rerank_candidates = max(1, int(rerank_candidates))

    def answer(
        self,
        question: str,
        *,
        state: GameState,
        attempts: Sequence[dict[str, Any]] = (),
        thread_messages: Sequence[dict[str, Any]] = (),
        k: int = 6,
    ) -> ExpertAnswer:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must not be empty")

        trace: list[dict[str, Any]] = []
        topic = classify_topic(cleaned)
        scope = self.profile.decide_scope(topic)
        trace.append(
            {
                "agent": "Game Expert Scope",
                "status": "supported" if scope.supported else "out_of_scope",
                "detail": f"topic={SUPPORT_TOPICS.get(topic, topic)}, version={scope.verified_version or '미확인'}",
            }
        )

        policy = build_spoiler_policy(
            self.profile, spoiler_level=state.spoiler_level, progress=state.progress
        )
        is_retry = bool(RETRY_PATTERN.search(cleaned))
        missing = _missing_state_fields(topic, state)

        if needs_progress_confirmation(cleaned, policy):
            trace.append(
                {
                    "agent": "Spoiler Policy",
                    "status": "needs_progress",
                    "detail": "스토리 질문인데 진행도가 확인되지 않았습니다.",
                }
            )
            return ExpertAnswer(
                answer=(
                    "스토리와 관련된 질문이라 진행 구간을 먼저 확인해야 합니다. "
                    "지금 어디까지 진행하셨는지 알려주시면 그 범위 안에서만 설명하겠습니다."
                ),
                scope=scope,
                follow_up_questions=[STATE_QUESTIONS["progress"]],
                spoiler=policy.to_dict(),
                trace=trace,
                is_retry=is_retry,
            )

        results = self._retrieve(cleaned, topic, k=k)
        allowed, blocked = filter_results(policy, results)
        trace.append(
            {
                "agent": "Game Knowledge Search",
                "status": "completed" if allowed else "no_evidence",
                "detail": f"evidence={len(allowed)}, spoiler_blocked={len(blocked)}",
            }
        )

        unverified = _unverified_items(scope, missing, blocked, allowed)
        follow_ups = [STATE_QUESTIONS[name] for name in missing if name in STATE_QUESTIONS]

        answer_text = self._generate(
            cleaned,
            topic=topic,
            scope=scope,
            state=state,
            results=allowed,
            attempts=list(attempts),
            thread_messages=list(thread_messages),
            policy=policy,
            is_retry=is_retry,
            missing=missing,
        )
        answer_text, leaks = redact_leaks(answer_text, policy)
        if leaks:
            trace.append(
                {
                    "agent": "Spoiler Policy",
                    "status": "redacted",
                    "detail": f"answer_leaks={len(leaks)}",
                }
            )
        trace.append(
            {"agent": "Game Expert Answer", "status": "completed", "detail": f"sources={len(allowed)}"}
        )
        return ExpertAnswer(
            answer=answer_text,
            scope=scope,
            evidence=allowed,
            applied_scope={
                "appid": self.profile.appid,
                "game": self.profile.name,
                "topic": topic,
                "topic_label": SUPPORT_TOPICS.get(topic, topic),
                "platform": state.platform or (self.profile.platforms[0] if self.profile.platforms else ""),
                "verified_version": scope.verified_version,
                "user_game_version": state.game_version,
                "playthrough": state.playthrough,
                "last_reviewed": scope.last_reviewed,
            },
            unverified=unverified,
            follow_up_questions=follow_ups,
            state_change_proposals=_state_change_proposals(cleaned, state, is_retry),
            spoiler={**policy.to_dict(), "notice": spoiler_notice(policy)},
            trace=trace,
            is_retry=is_retry,
        )

    def _retrieve(self, question: str, topic: str, *, k: int) -> list[SearchResult]:
        intent = detect_intent(question)
        spec = self.retriever.build_search_spec(question)
        embedding = self.embedder.embed_query(augment_query(question, intent))
        candidate_k = max(k, self.rerank_candidates) if self.reranker else k
        results = self.retriever.retrieve(
            question,
            embedding,
            k=candidate_k,
            search_spec=spec,
            allowed_appids=[self.profile.appid],
        )
        if self.reranker:
            results = self.reranker.rerank(question, results, top_n=k)
        return results[:k]

    def _generate(
        self,
        question: str,
        *,
        topic: str,
        scope: ScopeDecision,
        state: GameState,
        results: Sequence[SearchResult],
        attempts: Sequence[dict[str, Any]],
        thread_messages: Sequence[dict[str, Any]],
        policy: SpoilerPolicy,
        is_retry: bool,
        missing: Sequence[str],
    ) -> str:
        """Ask the answer Agent, falling back to a deterministic draft in place.

        저장소 규칙에 따라 별도 formatter 서비스를 만들지 않고, 이 함수 안에서
        API 오류나 빈 응답을 결정론적 답변으로 대체한다.
        """

        if not results:
            return _no_evidence_answer(self.profile, scope, policy, missing)
        if hasattr(self.answer_generator, "generate_expert_answer"):
            try:
                generated = self.answer_generator.generate_expert_answer(  # type: ignore[attr-defined]
                    question,
                    results,
                    {
                        "game": self.profile.name,
                        "appid": self.profile.appid,
                        "topic": SUPPORT_TOPICS.get(topic, topic),
                        "in_support_scope": scope.supported,
                        "scope_reason": scope.reason,
                        "verified_version": scope.verified_version,
                        "key_systems": [item.to_dict() for item in self.profile.key_systems],
                        "game_state": state.to_dict(),
                        "attempts": list(attempts),
                        "thread_messages": list(thread_messages)[-6:],
                        "spoiler_notice": spoiler_notice(policy),
                        "is_retry": is_retry,
                        "retry_assumptions": list(RETRY_ASSUMPTIONS),
                        "missing_state_fields": list(missing),
                    },
                )
                if str(generated or "").strip():
                    return str(generated).strip()
            except Exception:
                pass
        return _deterministic_expert_answer(
            self.profile, scope, state, results, policy, is_retry=is_retry, attempts=attempts
        )


def _missing_state_fields(topic: str, state: GameState) -> list[str]:
    """Ask only for the state that changes this topic's answer (§8-2)."""

    required = REQUIRED_STATE_FIELDS.get(topic, ())
    missing: list[str] = []
    for name in required:
        value = getattr(state, name, None)
        if not value:
            missing.append(name)
    return missing


def _unverified_items(
    scope: ScopeDecision,
    missing: Sequence[str],
    blocked: Sequence[dict[str, Any]],
    allowed: Sequence[SearchResult],
) -> list[str]:
    items: list[str] = []
    if not scope.supported:
        items.append(scope.reason)
    if not allowed:
        items.append("이 질문에 사용할 수 있는 검증된 자료를 찾지 못했습니다.")
    for name in missing:
        items.append(f"사용자 상태 미확인: {name}")
    if blocked:
        items.append(f"스포일러 범위 밖 자료 {len(blocked)}건은 사용하지 않았습니다.")
    return items


def _state_change_proposals(
    question: str, state: GameState, is_retry: bool
) -> list[dict[str, Any]]:
    """Propose (never silently apply) updates to the stored game state (§11)."""

    proposals: list[dict[str, Any]] = []
    progress_match = re.search(
        r"(?:지금|현재|막)?\s*([가-힣A-Za-z0-9 ]{2,30}?)\s*(?:에서\s*)?(?:막혔|진행\s*중|깼|클리어했)",
        question,
    )
    if progress_match:
        candidate = progress_match.group(1).strip()
        if candidate and candidate != state.progress:
            proposals.append(
                {
                    "field": "progress",
                    "value": candidate,
                    "reason": "이번 질문에서 언급한 진행 구간입니다. 확인 후 저장합니다.",
                }
            )
    if is_retry:
        proposals.append(
            {
                "field": "attempt",
                "value": question.strip()[:200],
                "reason": "시도했지만 해결되지 않은 방법으로 기록합니다.",
            }
        )
    return proposals


def _no_evidence_answer(
    profile: GameExpertProfile,
    scope: ScopeDecision,
    policy: SpoilerPolicy,
    missing: Sequence[str],
) -> str:
    lines = [
        f"**{profile.name}**에서 이 질문에 사용할 수 있는 검증된 자료를 찾지 못했습니다.",
        "",
        f"- 확인한 지원 범위: {scope.reason}",
        f"- 스포일러 설정: {spoiler_notice(policy)}",
    ]
    for name in missing:
        question = STATE_QUESTIONS.get(name)
        if question:
            lines.append(f"- 추가로 확인이 필요합니다: {question}")
    lines.append("")
    lines.append("근거 없이 추측한 공략은 제공하지 않았습니다.")
    return "\n".join(lines)


def _deterministic_expert_answer(
    profile: GameExpertProfile,
    scope: ScopeDecision,
    state: GameState,
    results: Sequence[SearchResult],
    policy: SpoilerPolicy,
    *,
    is_retry: bool,
    attempts: Sequence[dict[str, Any]],
) -> str:
    """§8 순서를 유지하는 결정론적 답변."""

    diagnosis = [f"**{profile.name}** 기준으로 확인했습니다."]
    if state.progress:
        diagnosis.append(f"현재 진행 구간은 '{state.progress}'로 저장돼 있습니다.")
    if state.character_build:
        diagnosis.append(f"빌드는 '{state.character_build}'입니다.")
    if not scope.supported:
        diagnosis.append(scope.reason)

    lines = ["### 현재 상황 진단", " ".join(diagnosis), "", "### 바로 시도할 행동"]
    for index, result in enumerate(results[:3], start=1):
        metadata = result.document.metadata
        title = str(metadata.get("item_title") or metadata.get("section") or f"근거 {index}")
        snippet = re.sub(r"\s+", " ", result.document.page_content).strip()
        lines.append(f"{index}. {title}: {snippet[:180]}{'…' if len(snippet) > 180 else ''} [근거 {index}]")
    lines.extend(["", "### 필요한 이유"])
    lines.append(
        "위 항목은 검색된 공식·검증 자료에서 확인한 내용이며, 실제 전술 제안과는 구분됩니다."
    )
    if is_retry:
        previous = ", ".join(str(item.get("action") or "")[:40] for item in attempts[-3:] if item.get("action"))
        lines.extend(["", "### 추가 힌트"])
        lines.append(
            "이전에 시도한 방법("
            + (previous or "기록 없음")
            + ")은 반복하지 않았습니다. "
            + ", ".join(RETRY_ASSUMPTIONS)
            + " 중 어떤 가정이 틀렸는지 하나씩 확인해 주세요."
        )
    else:
        lines.extend(["", "### 추가 힌트"])
        lines.append("잘되지 않으면 어떤 단계에서 막혔는지 알려주시면 가정을 바꿔 다시 확인하겠습니다.")
    lines.extend(["", spoiler_notice(policy)])
    return "\n".join(lines)
