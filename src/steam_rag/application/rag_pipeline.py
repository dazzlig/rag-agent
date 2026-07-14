from __future__ import annotations

from pathlib import Path

from steam_rag.agents.agentic_rag import AgenticRAGConfig, AgenticRAGCoordinator
from steam_rag.agents.multi_agent_workflow import MultiAgentConfig, SteamMultiAgentWorkflow
from steam_rag.common.interfaces import AnswerGenerator, Embedder
from steam_rag.common.models import RAGAnswer, SearchResult
from steam_rag.rag_search.hybrid_retriever import HybridTimeAwareRetriever, augment_query, detect_intent
from steam_rag.rag_search.reranker import Reranker
from steam_rag.rag_search.vector_store import VectorIndex


class RAGPipeline:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        answer_generator: AnswerGenerator | None = None,
        reranker: Reranker | None = None,
        rerank_candidates: int = 24,
    ) -> None:
        if index.embedding_model != embedder.model_name:
            raise ValueError(
                f"Index uses {index.embedding_model!r}, but embedder uses {embedder.model_name!r}"
            )
        self.index = index
        self.embedder = embedder
        self.answer_generator = answer_generator
        self.reranker = reranker
        self.rerank_candidates = max(1, int(rerank_candidates))
        self.retriever = HybridTimeAwareRetriever(index)

    @classmethod
    def from_path(
        cls,
        path: Path,
        embedder: Embedder,
        answer_generator: AnswerGenerator | None = None,
        reranker: Reranker | None = None,
        rerank_candidates: int = 24,
    ) -> "RAGPipeline":
        return cls(VectorIndex.load(path), embedder, answer_generator, reranker, rerank_candidates)

    def search(self, question: str, *, k: int = 5) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("question must not be empty")
        intent = detect_intent(question)
        search_spec = self.retriever.build_search_spec(question)
        embedding = self.embedder.embed_query(augment_query(question, intent))
        candidate_k = max(k, self.rerank_candidates) if self.reranker else k
        results = self.retriever.retrieve(
            question,
            embedding,
            k=candidate_k,
            search_spec=search_spec,
        )
        if self.reranker:
            return self.reranker.rerank(question, results, top_n=k)
        return results

    def ask(self, question: str, *, k: int = 5) -> RAGAnswer:
        if self.answer_generator is None:
            raise RuntimeError("answer_generator is required for ask()")
        results = self.search(question, k=k)
        if not results:
            return RAGAnswer(question, "검색된 근거가 없어 답변할 수 없습니다.", [])
        answer = self.answer_generator.generate(question, results)
        return RAGAnswer(question, answer, results)

    def search_agentic(
        self,
        question: str,
        *,
        k: int = 5,
        max_steps: int = 3,
        use_hyde: bool = True,
    ) -> tuple[list[SearchResult], dict[str, object]]:
        if self.answer_generator is None:
            raise RuntimeError("answer_generator is required for agentic search")
        coordinator = AgenticRAGCoordinator(
            self.retriever,
            self.embedder,
            self.answer_generator,
            config=AgenticRAGConfig(
                max_steps=max_steps,
                per_step_k=max(k, self.rerank_candidates) if self.reranker else k,
                use_hyde=use_hyde,
            ),
            reranker=self.reranker,
            rerank_candidates=self.rerank_candidates,
        )
        return coordinator.search(question, k=k)

    def ask_agentic(
        self,
        question: str,
        *,
        k: int = 5,
        max_steps: int = 3,
        use_hyde: bool = True,
    ) -> RAGAnswer:
        if self.answer_generator is None:
            raise RuntimeError("answer_generator is required for agentic ask()")
        coordinator = AgenticRAGCoordinator(
            self.retriever,
            self.embedder,
            self.answer_generator,
            config=AgenticRAGConfig(
                max_steps=max_steps,
                per_step_k=max(k, self.rerank_candidates) if self.reranker else k,
                use_hyde=use_hyde,
            ),
            reranker=self.reranker,
            rerank_candidates=self.rerank_candidates,
        )
        return coordinator.ask(question, k=k)

    def ask_multi_agent(
        self,
        question: str,
        *,
        k: int = 5,
        max_steps: int = 3,
        use_hyde: bool = True,
        query_variants: list[str] | None = None,
        allowed_appids: list[int] | None = None,
    ) -> RAGAnswer:
        """Run the production-facing LangGraph multi-agent workflow."""

        if self.answer_generator is None:
            raise RuntimeError("answer_generator is required for multi-agent ask()")
        workflow = SteamMultiAgentWorkflow(
            self.retriever,
            self.embedder,
            self.answer_generator,
            config=MultiAgentConfig(
                max_steps=max_steps,
                per_step_k=max(k, self.rerank_candidates) if self.reranker else k,
                use_hyde=use_hyde,
            ),
            reranker=self.reranker,
            rerank_candidates=self.rerank_candidates,
        )
        return workflow.invoke(
            question,
            k=k,
            query_variants=query_variants,
            allowed_appids=allowed_appids,
        )
