from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from steam_rag.agents.agentic_rag import AgenticRAGConfig, GameResearchAgent
from steam_rag.common.interfaces import AnswerGenerator, Embedder
from steam_rag.common.models import Document, RAGAnswer, SearchResult
from steam_rag.rag_search.hybrid_retriever import HybridTimeAwareRetriever, augment_query, detect_intent
from steam_rag.rag_search.reranker import Reranker
from steam_rag.rag_search.search_spec import SearchSpec, evaluate_evidence_coverage


class MultiAgentState(TypedDict, total=False):
    question: str
    k: int
    use_hyde: bool
    query_variants: list[str]
    search_spec: SearchSpec
    results: list[SearchResult]
    research_metadata: dict[str, Any]
    evidence_coverage: dict[str, Any]
    needs_refinement: bool
    answer: str
    trace: list[dict[str, Any]]
    allowed_appids: list[int]


@dataclass(slots=True)
class MultiAgentConfig:
    max_steps: int = 3
    per_step_k: int = 5
    min_claim_coverage: float = 0.8
    max_query_variants: int = 4
    use_hyde: bool = True


class QueryPlannerAgent:
    """Convert a user question into a deterministic SearchSpec."""

    def __init__(self, retriever: HybridTimeAwareRetriever) -> None:
        self.retriever = retriever

    def __call__(self, state: MultiAgentState) -> dict[str, Any]:
        spec = self.retriever.build_search_spec(state["question"])
        trace = list(state.get("trace", []))
        trace.append(
            {
                "agent": "Query Planner Agent",
                "status": "completed",
                "detail": f"intent={spec.intent}, claims={len(spec.claims)}",
            }
        )
        return {"search_spec": spec, "trace": trace}


class QueryExpansionAgent:
    """Generate aliases and alternate search expressions without inventing AppIDs."""

    def __init__(self, answer_generator: AnswerGenerator, max_variants: int = 4) -> None:
        self.answer_generator = answer_generator
        self.max_variants = max(1, max_variants)

    def expand(self, question: str) -> list[str]:
        variants = [question.strip()]
        if hasattr(self.answer_generator, "expand_search_queries"):
            try:
                generated = self.answer_generator.expand_search_queries(question)  # type: ignore[attr-defined]
            except Exception:
                generated = []
            if isinstance(generated, dict):
                generated = generated.get("query_variants", [])
            if isinstance(generated, Sequence) and not isinstance(generated, (str, bytes)):
                variants.extend(str(item).strip() for item in generated if str(item).strip())
        return _dedupe_text(variants)[: self.max_variants]

    def __call__(self, state: MultiAgentState) -> dict[str, Any]:
        variants = state.get("query_variants") or self.expand(state["question"])
        variants = _dedupe_text(variants)[: self.max_variants]
        trace = list(state.get("trace", []))
        trace.append(
            {
                "agent": "Query Expansion Agent",
                "status": "completed",
                "detail": f"variants={len(variants)}",
                "query_variants": variants,
            }
        )
        return {"query_variants": variants, "trace": trace}


class EvidenceCriticAgent:
    """Check claim-level evidence coverage and request one bounded refinement."""

    def __init__(self, min_coverage: float) -> None:
        self.min_coverage = min_coverage

    def __call__(self, state: MultiAgentState) -> dict[str, Any]:
        allowed = {str(appid) for appid in state.get("allowed_appids", [])}
        original_results = list(state.get("results", []))
        results = [
            result
            for result in original_results
            if not allowed
            or str(result.document.metadata.get("appid") or "") in allowed
        ]
        report = evaluate_evidence_coverage(state["search_spec"], results)
        needs_refinement = report.coverage_ratio < self.min_coverage
        trace = list(state.get("trace", []))
        trace.append(
            {
                "agent": "Evidence Critic Agent",
                "status": "refine" if needs_refinement else "sufficient",
                "detail": (
                    f"claim_coverage={report.coverage_ratio:.2f}, "
                    f"appid_filtered={len(original_results) - len(results)}"
                ),
            }
        )
        return {
            "evidence_coverage": report.to_dict(),
            "needs_refinement": needs_refinement,
            "results": results,
            "trace": trace,
        }


class SteamMultiAgentWorkflow:
    """LangGraph workflow for planning, game research, critique, and answering.

    Deterministic collectors/indexers remain services.  Only components that
    plan, choose searches, judge evidence, or compose an answer are called
    agents.
    """

    def __init__(
        self,
        retriever: HybridTimeAwareRetriever,
        embedder: Embedder,
        answer_generator: AnswerGenerator,
        *,
        config: MultiAgentConfig | None = None,
        reranker: Reranker | None = None,
        rerank_candidates: int = 24,
    ) -> None:
        self.retriever = retriever
        self.embedder = embedder
        self.answer_generator = answer_generator
        self.config = config or MultiAgentConfig()
        self.reranker = reranker
        self.rerank_candidates = max(1, rerank_candidates)
        self.planner_agent = QueryPlannerAgent(retriever)
        self.expansion_agent = QueryExpansionAgent(answer_generator, self.config.max_query_variants)
        self.game_research_agent = GameResearchAgent(
            retriever,
            embedder,
            answer_generator,
            config=AgenticRAGConfig(
                max_steps=self.config.max_steps,
                per_step_k=self.config.per_step_k,
                use_hyde=self.config.use_hyde,
                min_claim_coverage=self.config.min_claim_coverage,
            ),
            reranker=reranker,
            rerank_candidates=rerank_candidates,
        )
        self.critic_agent = EvidenceCriticAgent(self.config.min_claim_coverage)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(MultiAgentState)
        graph.add_node("query_planner_agent", self.planner_agent)
        graph.add_node("query_expansion_agent", self.expansion_agent)
        graph.add_node("game_research_agent", self._research)
        graph.add_node("evidence_critic_agent", self.critic_agent)
        graph.add_node("research_refinement_agent", self._refine)
        graph.add_node("web_research_agent", self._web_research)
        graph.add_node("answer_agent", self._answer)
        graph.add_edge(START, "query_planner_agent")
        graph.add_edge("query_planner_agent", "query_expansion_agent")
        graph.add_edge("query_expansion_agent", "game_research_agent")
        graph.add_edge("game_research_agent", "evidence_critic_agent")
        graph.add_conditional_edges(
            "evidence_critic_agent",
            lambda state: "refine" if state.get("needs_refinement") else "web",
            {"refine": "research_refinement_agent", "web": "web_research_agent"},
        )
        graph.add_edge("research_refinement_agent", "web_research_agent")
        graph.add_edge("web_research_agent", "answer_agent")
        graph.add_edge("answer_agent", END)
        return graph.compile()

    def invoke(
        self,
        question: str,
        *,
        k: int = 5,
        query_variants: Sequence[str] | None = None,
        allowed_appids: Sequence[int] | None = None,
    ) -> RAGAnswer:
        initial: MultiAgentState = {
            "question": question.strip(),
            "k": max(1, k),
            "use_hyde": self.config.use_hyde,
            "trace": [],
            "allowed_appids": list(dict.fromkeys(int(appid) for appid in (allowed_appids or []))),
        }
        if query_variants:
            initial["query_variants"] = list(query_variants)
        state = self.graph.invoke(initial)
        results = list(state.get("results", []))
        metadata = dict(state.get("research_metadata", {}))
        metadata.update(
            {
                "orchestrator": "langgraph",
                "workflow": "steam_multi_agent",
                "query_variants": state.get("query_variants", []),
                "evidence_coverage": state.get("evidence_coverage", {}),
                "agent_trace": state.get("trace", []),
                "allowed_appids": state.get("allowed_appids", []),
            }
        )
        return RAGAnswer(question, state.get("answer", "근거를 찾지 못했습니다."), results, metadata)

    def _research(self, state: MultiAgentState) -> dict[str, Any]:
        question = state["question"]
        k = state["k"]
        results, metadata = self.game_research_agent.search(
            question,
            k=k,
            allowed_appids=state.get("allowed_appids", []),
        )

        # Alias variants improve recall without multiplying HyDE calls.  The
        # first/original query receives the full agentic loop; extra variants
        # run one bounded hybrid retrieval and are merged by document identity.
        for variant in state.get("query_variants", [])[1:]:
            intent = detect_intent(variant)
            embedding = self.embedder.embed_query(augment_query(variant, intent))
            results.extend(
                self.retriever.retrieve(
                    variant,
                    embedding,
                    k=max(k, self.config.per_step_k),
                    search_spec=state["search_spec"],
                    allowed_appids=state.get("allowed_appids", []),
                )
            )
        final = self.game_research_agent._rank_unique(  # noqa: SLF001 - shared ranking contract
            results,
            k=max(k, self.rerank_candidates) if self.reranker else k,
            question=question,
            intent=state["search_spec"].intent,
        )
        if self.reranker:
            final = self.reranker.rerank(question, final, top_n=k)
        else:
            final = final[:k]
        trace = list(state.get("trace", []))
        trace.append(
            {
                "agent": "Game Research Agent",
                "status": "completed",
                "detail": f"evidence={len(final)}, steps={metadata.get('stopped_after_step', 0)}",
            }
        )
        return {"results": final, "research_metadata": metadata, "trace": trace}

    def _refine(self, state: MultiAgentState) -> dict[str, Any]:
        coverage = state.get("evidence_coverage", {})
        missing_claims = [
            str(claim.get("text") or claim.get("claim_id"))
            for claim in coverage.get("claims", [])
            if isinstance(claim, dict) and not claim.get("supported")
        ]
        if not missing_claims:
            return {"needs_refinement": False}
        query = f"{state['question']} {' '.join(missing_claims[:3])}"
        embedding = self.embedder.embed_query(augment_query(query, state["search_spec"].intent))
        extra = self.retriever.retrieve(
            query,
            embedding,
            k=max(state["k"], self.config.per_step_k),
            search_spec=state["search_spec"],
            allowed_appids=state.get("allowed_appids", []),
        )
        merged = self.game_research_agent._rank_unique(  # noqa: SLF001
            [*state.get("results", []), *extra],
            k=state["k"],
            question=state["question"],
            intent=state["search_spec"].intent,
        )
        report = evaluate_evidence_coverage(state["search_spec"], merged)
        trace = list(state.get("trace", []))
        trace.append(
            {
                "agent": "Research Refinement Agent",
                "status": "completed",
                "detail": f"claim_coverage={report.coverage_ratio:.2f}",
            }
        )
        return {
            "results": merged,
            "evidence_coverage": report.to_dict(),
            "needs_refinement": False,
            "trace": trace,
        }

    def _answer(self, state: MultiAgentState) -> dict[str, Any]:
        results = state.get("results", [])
        if not results:
            answer = "확인할 수 있는 근거를 찾지 못했습니다. 게임 이름이나 조건을 조금 더 구체적으로 입력해 주세요."
        else:
            metadata = dict(state.get("research_metadata", {}))
            metadata["evidence_coverage"] = state.get("evidence_coverage", {})
            metadata["agent_trace"] = state.get("trace", [])
            if hasattr(self.answer_generator, "generate_service_answer"):
                answer = self.answer_generator.generate_service_answer(  # type: ignore[attr-defined]
                    state["question"], results, metadata
                )
            elif hasattr(self.answer_generator, "generate_agentic"):
                answer = self.answer_generator.generate_agentic(  # type: ignore[attr-defined]
                    state["question"], results, metadata
                )
            else:
                answer = self.answer_generator.generate(state["question"], results)
        trace = list(state.get("trace", []))
        trace.append(
            {"agent": "Answer Agent", "status": "completed", "detail": f"sources={len(results)}"}
        )
        return {"answer": answer, "trace": trace}

    def _web_research(self, state: MultiAgentState) -> dict[str, Any]:
        trace = list(state.get("trace", []))
        coverage = state.get("evidence_coverage", {})
        if not _should_use_web_supplement(state["question"], coverage) or not hasattr(
            self.answer_generator, "research_web_evidence"
        ):
            trace.append(
                {
                    "agent": "Web Research Agent",
                    "status": "skipped",
                    "detail": "Steam 근거가 충분하고 외부 자료 요청이 없음",
                }
            )
            return {"trace": trace}

        missing_claims = [
            str(claim.get("text") or claim.get("claim_id"))
            for claim in coverage.get("claims", [])
            if isinstance(claim, dict) and not claim.get("supported")
        ]
        game_names = list(
            dict.fromkeys(
                str(result.document.metadata.get("game_name") or "").strip()
                for result in state.get("results", [])
                if str(result.document.metadata.get("game_name") or "").strip()
            )
        )
        try:
            rows = self.answer_generator.research_web_evidence(  # type: ignore[attr-defined]
                state["question"],
                game_names=game_names,
                missing_claims=missing_claims,
                limit=3,
            )
        except Exception as exc:
            trace.append(
                {
                    "agent": "Web Research Agent",
                    "status": "fallback",
                    "detail": f"웹 보조 검색 실패: {type(exc).__name__}",
                }
            )
            return {"trace": trace}

        allowed = state.get("allowed_appids", [])
        web_results: list[SearchResult] = []
        for row in rows:
            snippet = str(row.get("snippet") or row.get("claim") or "").strip()
            if not snippet:
                continue
            metadata: dict[str, Any] = {
                "section": "web",
                "source_type": str(row.get("source_type") or "web"),
                "game_name": str(row.get("game") or (game_names[0] if len(game_names) == 1 else "")),
                "item_title": str(row.get("title") or "웹 보조 근거"),
                "source_date": str(row.get("published_at") or ""),
                "publisher": str(row.get("publisher") or ""),
                "url": str(row.get("url") or ""),
            }
            if len(allowed) == 1:
                metadata["appid"] = allowed[0]
            web_results.append(SearchResult(Document(snippet, metadata), score=0.5, role="supplement"))

        combined = [*state.get("results", []), *web_results]
        for rank, result in enumerate(combined, start=1):
            result.rank = rank
        trace.append(
            {
                "agent": "Web Research Agent",
                "status": "completed" if web_results else "no_evidence",
                "detail": f"supplemental_sources={len(web_results)}",
            }
        )
        metadata = dict(state.get("research_metadata", {}))
        metadata["web_supplement_count"] = len(web_results)
        return {"results": combined, "research_metadata": metadata, "trace": trace}


def _dedupe_text(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _should_use_web_supplement(question: str, coverage: dict[str, Any]) -> bool:
    explicitly_external = bool(
        re.search(r"웹|공식\s*(?:발표|사이트|자료)|개발사|퍼블리셔|인터뷰|로드맵|보도", question.casefold())
    )
    return explicitly_external
