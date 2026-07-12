from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .interfaces import AnswerGenerator, Embedder
from .models import RAGAnswer, SearchResult
from .rerank import Reranker
from .retrieval import HybridTimeAwareRetriever, augment_query, detect_intent


@dataclass(slots=True)
class AgenticRAGConfig:
    """Runtime knobs for the coordinator layer.

    HyDE is intentionally enabled only through this agentic path because it
    costs an extra LLM call per search step.
    """

    max_steps: int = 3
    per_step_k: int = 5
    min_sources: int = 3
    use_hyde: bool = True


@dataclass(slots=True)
class SearchGoal:
    name: str
    query: str
    reason: str


@dataclass(slots=True)
class AgenticStep:
    step: int
    goal: SearchGoal
    hyde: str = ""
    hyde_error: str = ""
    results: list[SearchResult] = field(default_factory=list)
    sufficient: bool = False

    def to_dict(self) -> dict[str, Any]:
        sections = sorted({str(result.document.metadata.get("section", "")) for result in self.results})
        return {
            "step": self.step,
            "goal": self.goal.name,
            "query": self.goal.query,
            "reason": self.goal.reason,
            "hyde_enabled": bool(self.hyde),
            "hyde_preview": self.hyde[:280] + ("..." if len(self.hyde) > 280 else ""),
            "hyde_error": self.hyde_error,
            "results": len(self.results),
            "sections": [section for section in sections if section],
            "sufficient": self.sufficient,
        }


class AgenticRAGCoordinator:
    """Plan, retrieve with HyDE, check coverage, and refine before generation."""

    def __init__(
        self,
        retriever: HybridTimeAwareRetriever,
        embedder: Embedder,
        answer_generator: AnswerGenerator,
        *,
        config: AgenticRAGConfig | None = None,
        reranker: Reranker | None = None,
        rerank_candidates: int = 24,
    ) -> None:
        self.retriever = retriever
        self.embedder = embedder
        self.answer_generator = answer_generator
        self.config = config or AgenticRAGConfig()
        self.reranker = reranker
        self.rerank_candidates = max(1, int(rerank_candidates))

    def search(self, question: str, *, k: int = 5) -> tuple[list[SearchResult], dict[str, Any]]:
        if not question.strip():
            raise ValueError("question must not be empty")

        intent = detect_intent(question)
        plan = self._plan(question, intent)[: max(1, self.config.max_steps)]
        all_results: list[SearchResult] = []
        steps: list[AgenticStep] = []

        for step_index, goal in enumerate(plan, start=1):
            hyde, hyde_error = self._hyde(question, goal) if self.config.use_hyde else ("", "")
            embedding_text = self._embedding_query(goal.query, hyde)
            embedding = self.embedder.embed_query(embedding_text)
            results = self.retriever.retrieve(goal.query, embedding, k=max(k, self.config.per_step_k))
            all_results.extend(results)

            ranked = self._rank_unique(all_results, k=max(k, self.config.min_sources), question=question, intent=intent)
            sufficient = self._is_sufficient(question, intent, ranked)
            steps.append(
                AgenticStep(
                    step=step_index,
                    goal=goal,
                    hyde=hyde,
                    hyde_error=hyde_error,
                    results=results,
                    sufficient=sufficient,
                )
            )
            if sufficient:
                break

        final_k = max(k, self.rerank_candidates) if self.reranker else k
        final_results = self._rank_unique(all_results, k=final_k, question=question, intent=intent)
        if self.reranker:
            final_results = self.reranker.rerank(question, final_results, top_n=k)
        metadata = {
            "strategy": "agentic_hyde" if self.config.use_hyde else "agentic",
            "reranker": self.reranker.model_name if self.reranker else "",
            "intent": intent,
            "plan": [
                {"name": goal.name, "query": goal.query, "reason": goal.reason}
                for goal in plan
            ],
            "steps": [step.to_dict() for step in steps],
            "stopped_after_step": steps[-1].step if steps else 0,
            "sufficient": steps[-1].sufficient if steps else False,
        }
        return final_results, metadata

    def ask(self, question: str, *, k: int = 5) -> RAGAnswer:
        results, metadata = self.search(question, k=k)
        if not results:
            return RAGAnswer(question, "검색된 근거가 없어 답변할 수 없습니다.", [], metadata=metadata)

        if hasattr(self.answer_generator, "generate_agentic"):
            answer = self.answer_generator.generate_agentic(question, results, metadata)  # type: ignore[attr-defined]
        else:
            answer = self.answer_generator.generate(question, results)
        return RAGAnswer(question, answer, results, metadata=metadata)

    def _plan(self, question: str, intent: str) -> list[SearchGoal]:
        base = question.strip()
        target = self._target_prefix(question)
        if self._needs_playstyle(question) and (self._needs_reviews(question) or self._needs_updates(question)):
            goals = [
                SearchGoal("playstyle_profile", f"{base} gameplay combat perspective dimension tags facets", "플레이스타일/전투/시점/차원 근거 확인"),
            ]
            if self._needs_updates(question):
                goals.append(SearchGoal("latest_update", f"{target} latest patch update hotfix news", "최신 업데이트/패치 근거 확인"))
            if self._needs_reviews(question):
                goals.append(SearchGoal("recent_reviews", f"{target} recent user reviews sentiment pros cons", "최근 리뷰/평가 근거 수집"))
            goals.append(SearchGoal("store_context", f"{target} store summary about genre gameplay", "공식 설명 기반 맥락 보강"))
            return goals
        if intent == "after_update":
            return [
                SearchGoal("latest_update", f"{base} latest patch update hotfix", "최신 업데이트/패치 근거 확인"),
                SearchGoal("post_update_reviews", f"{base} reviews after update player reaction sentiment", "업데이트 이후 유저 반응 확인"),
                SearchGoal("store_context", f"{base} game overview genre gameplay", "게임 기본 맥락 보강"),
            ]
        if intent == "news":
            return [
                SearchGoal("latest_news", f"{base} latest update patch notes announcement", "최신 뉴스/패치 우선 검색"),
                SearchGoal("release_context", f"{base} store summary current state", "스토어 요약으로 맥락 보강"),
            ]
        if intent == "review":
            return [
                SearchGoal("recent_reviews", f"{base} recent user reviews sentiment pros cons", "최근 리뷰/평가 근거 수집"),
                SearchGoal("review_context", f"{base} gameplay issues strengths weaknesses", "평가의 이유가 되는 플레이 경험 확인"),
            ]
        if intent == "gameplay":
            return [
                SearchGoal("playstyle_profile", f"{base} gameplay combat perspective dimension tags facets", "플레이스타일 facet 근거 검색"),
                SearchGoal("gameplay_details", f"{base} controls progression exploration combat loop", "전투/탐험/성장 루프 보강"),
                SearchGoal("player_feel", f"{base} player reviews feel difficulty pacing", "유저가 체감한 플레이 감각 확인"),
            ]
        return [
            SearchGoal("overview", f"{base} store summary genre gameplay", "기본 설명과 장르 확인"),
            SearchGoal("playstyle", f"{base} gameplay combat perspective playstyle", "플레이스타일 정보 확인"),
            SearchGoal("reviews", f"{base} user reviews sentiment", "유저 평가 보강"),
        ]

    def _hyde(self, question: str, goal: SearchGoal) -> tuple[str, str]:
        if not hasattr(self.answer_generator, "generate_hyde"):
            return "", ""
        try:
            hyde = self.answer_generator.generate_hyde(question, goal.query, goal.reason)  # type: ignore[attr-defined]
        except Exception as exc:  # HyDE failure should not block normal retrieval.
            return "", str(exc)
        return hyde.strip(), ""

    def _embedding_query(self, query: str, hyde: str) -> str:
        intent = detect_intent(query)
        if not hyde:
            return augment_query(query, intent)
        return augment_query(
            f"{query}\n\nHypothetical answer for retrieval:\n{hyde}",
            intent,
        )

    def _rank_unique(
        self,
        results: Sequence[SearchResult],
        *,
        k: int,
        question: str = "",
        intent: str = "",
    ) -> list[SearchResult]:
        best: dict[tuple[object, ...], SearchResult] = {}
        for index, result in enumerate(results):
            key = self._result_key(result)
            adjusted = result.score + max(0.0, 0.03 - 0.002 * index)
            if key not in best or adjusted > best[key].score:
                result.score = adjusted
                best[key] = result
        ranked_all = sorted(best.values(), key=lambda item: item.score, reverse=True)
        priority_sections = self._priority_sections(question, intent)
        ranked: list[SearchResult] = []
        used_keys: set[tuple[object, ...]] = set()
        used_sections: set[str] = set()

        for section in priority_sections:
            if len(ranked) >= k or section in used_sections:
                continue
            candidate = next(
                (
                    result
                    for result in ranked_all
                    if str(result.document.metadata.get("section", "")) == section
                ),
                None,
            )
            if candidate is None:
                continue
            key = self._result_key(candidate)
            ranked.append(candidate)
            used_keys.add(key)
            used_sections.add(section)

        for result in ranked_all:
            if len(ranked) >= k:
                break
            key = self._result_key(result)
            if key in used_keys:
                continue
            ranked.append(result)
            used_keys.add(key)

        ranked = ranked[:k]
        for rank, result in enumerate(ranked, start=1):
            result.rank = rank
        return ranked

    def _is_sufficient(self, question: str, intent: str, results: Sequence[SearchResult]) -> bool:
        if len(results) < self.config.min_sources:
            return False
        sections = {str(result.document.metadata.get("section", "")) for result in results}
        if self._needs_playstyle(question) and not bool({"metadata", "store_summary", "about"} & sections):
            return False
        if self._needs_updates(question) and "news" not in sections:
            return False
        if self._needs_reviews(question) and "review" not in sections:
            return False
        if intent == "after_update":
            return "news" in sections and "review" in sections
        if intent == "news":
            return "news" in sections
        if intent == "review":
            return "review" in sections
        if intent == "gameplay":
            return bool({"about", "store_summary", "metadata"} & sections)
        return bool({"store_summary", "about", "metadata"} & sections)

    def _result_key(self, result: SearchResult) -> tuple[object, ...]:
        metadata = result.document.metadata
        return (
            metadata.get("game_key"),
            metadata.get("section"),
            metadata.get("item_title"),
            metadata.get("chunk_id"),
        )

    def _priority_sections(self, question: str, intent: str) -> list[str]:
        sections: list[str] = []
        if self._needs_playstyle(question) or intent in {"gameplay", "general"}:
            sections.extend(["metadata", "store_summary", "about"])
        if self._needs_updates(question) or intent in {"news", "after_update"}:
            sections.append("news")
        if self._needs_reviews(question) or intent in {"review", "after_update"}:
            sections.append("review")
        if not sections:
            sections = ["store_summary", "about", "metadata", "review", "news"]
        deduped: list[str] = []
        for section in sections:
            if section not in deduped:
                deduped.append(section)
        return deduped

    def _needs_playstyle(self, question: str) -> bool:
        lowered = question.casefold()
        return bool(
            re.search(
                r"2\.5d|2d|3d|quarter|isometric|camera|perspective|combat|gameplay|playstyle|facet|"
                r"시점|차원|전투|플레이스타일|플레이 스타일|쿼터뷰|분류|로그라이크|액션",
                lowered,
            )
        )

    def _needs_reviews(self, question: str) -> bool:
        lowered = question.casefold()
        return bool(re.search(r"review|sentiment|reaction|평가|리뷰|반응|최근 평가", lowered))

    def _needs_updates(self, question: str) -> bool:
        lowered = question.casefold()
        return bool(re.search(r"update|patch|hotfix|news|업데이트|패치|핫픽스|최근", lowered))

    def _target_prefix(self, question: str) -> str:
        appid = re.search(
            r"(?:store\.steampowered\.com/app/|\bappid\s*[:=#]?\s*)(\d{2,10})\s*([A-Za-z0-9:'’&_. -]{0,80})",
            question,
            flags=re.IGNORECASE,
        )
        if appid:
            title = re.sub(r"[_-]+", " ", appid.group(2)).strip(" /-_.")
            title = re.sub(r"\s{2,}", " ", title)
            return f"appid: {appid.group(1)} {title}".strip()
        return question.strip()
