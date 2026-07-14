from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import _bootstrap  # noqa: F401
from steam_rag.application.rag_pipeline import RAGPipeline
from steam_rag.common.models import Document, SearchResult
from steam_rag.rag_search.vector_store import VectorIndex, build_index


class FakeEmbedder:
    model_name = "fake-embedding"

    def __init__(self) -> None:
        self.last_query = ""
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, float("patch" in text.casefold())] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.last_query = text
        self.queries.append(text)
        return [1.0, float("패치" in text or "patch" in text.casefold())]


class FakeGenerator:
    def generate(self, question: str, results: Sequence[SearchResult]) -> str:
        return f"{len(results)}개 근거를 사용한 답변 [근거 1]"


class FakeAgenticGenerator(FakeGenerator):
    def __init__(self) -> None:
        self.hyde_calls: list[tuple[str, str, str]] = []
        self.agentic_metadata: dict[str, object] = {}

    def generate_hyde(self, question: str, search_query: str, reason: str) -> str:
        self.hyde_calls.append((question, search_query, reason))
        return "Hollow Knight is a 2D side-scroller metroidvania with melee combat and exploration."

    def generate_agentic(
        self,
        question: str,
        results: Sequence[SearchResult],
        metadata: dict[str, object],
    ) -> str:
        self.agentic_metadata = metadata
        return f"agentic 답변: {len(results)}개 근거 [근거 1]"


class FakeReranker:
    model_name = "fake-reranker"

    def rerank(self, question: str, results: Sequence[SearchResult], *, top_n: int) -> list[SearchResult]:
        scored = list(results)
        for result in scored:
            title = str(result.document.metadata.get("item_title", ""))
            result.rerank_score = 10.0 if "Preferred" in title else 0.1
        scored.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        for rank, result in enumerate(scored[:top_n], start=1):
            result.rank = rank
        return scored[:top_n]


class PipelineTests(unittest.TestCase):
    def test_index_roundtrip_and_answer_generation(self) -> None:
        documents = [
            Document(
                "Patch notes now live.",
                {
                    "game_key": "hollow_knight",
                    "section": "news",
                    "item_title": "Patch Now Live",
                    "source_date": "2026-06-01",
                    "relevance_type": "valid_update_or_patch",
                    "chunk_id": "patch",
                },
            )
        ]
        index = build_index(documents, FakeEmbedder())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            index.save(path)
            loaded = VectorIndex.load(path)

        embedder = FakeEmbedder()
        pipeline = RAGPipeline(loaded, embedder, FakeGenerator())
        answer = pipeline.ask("할로우 나이트 패치", k=1)
        self.assertEqual(len(answer.sources), 1)
        self.assertIn("[근거 1]", answer.answer)
        self.assertIn("patch notes", embedder.last_query)

    def test_embedding_model_mismatch_is_rejected(self) -> None:
        index = VectorIndex([], [], "another-model")
        with self.assertRaises(ValueError):
            RAGPipeline(index, FakeEmbedder())

    def test_pipeline_applies_optional_reranker_after_retrieval(self) -> None:
        documents = [
            Document(
                "Generic result with lexical overlap.",
                {
                    "game_key": "example",
                    "game_name": "Example",
                    "section": "about",
                    "item_title": "Generic",
                    "chunk_id": "generic",
                },
            ),
            Document(
                "Preferred result selected by cross encoder.",
                {
                    "game_key": "example",
                    "game_name": "Example",
                    "section": "about",
                    "item_title": "Preferred",
                    "chunk_id": "preferred",
                },
            ),
        ]
        embedder = FakeEmbedder()
        index = build_index(documents, embedder)
        pipeline = RAGPipeline(index, embedder, reranker=FakeReranker(), rerank_candidates=2)

        results = pipeline.search("example gameplay", k=1)

        self.assertEqual(results[0].document.metadata["item_title"], "Preferred")
        self.assertEqual(results[0].rerank_score, 10.0)

    def test_agentic_rag_uses_hyde_and_returns_trace_metadata(self) -> None:
        documents = [
            Document(
                "Hollow Knight has side-scrolling melee combat, exploration, and metroidvania progression.",
                {
                    "game_key": "hollow_knight",
                    "game_name": "Hollow Knight",
                    "section": "about",
                    "item_title": "About",
                    "chunk_id": "about",
                },
            ),
            Document(
                "Players praise the precise combat and challenging boss fights.",
                {
                    "game_key": "hollow_knight",
                    "game_name": "Hollow Knight",
                    "section": "review",
                    "item_title": "Review 1",
                    "chunk_id": "review",
                    "source_date": "2026-06-01",
                },
            ),
        ]
        embedder = FakeEmbedder()
        index = build_index(documents, embedder)
        generator = FakeAgenticGenerator()
        pipeline = RAGPipeline(index, embedder, generator)

        answer = pipeline.ask_agentic("Hollow Knight 전투와 탐험 특징", k=2, max_steps=2)

        self.assertIn("agentic 답변", answer.answer)
        self.assertEqual(answer.metadata["strategy"], "agentic_hyde")
        self.assertTrue(answer.metadata["steps"])
        self.assertTrue(generator.hyde_calls)
        self.assertIn("Hypothetical answer", embedder.queries[-1])
        self.assertEqual(generator.agentic_metadata["strategy"], "agentic_hyde")

    def test_agentic_mixed_playstyle_recent_question_requires_diverse_sections(self) -> None:
        documents = [
            Document(
                "combat_facets real_time perspective_facets isometric dimension_facets 3d playstyle_facets exploration",
                {
                    "game_key": "hades_ii_1145350",
                    "game_name": "Hades II",
                    "section": "metadata",
                    "item_title": "Metadata",
                    "chunk_id": "metadata",
                },
            ),
            Document(
                "Post-launch patch hotfix changed balance and fixes.",
                {
                    "game_key": "hades_ii_1145350",
                    "game_name": "Hades II",
                    "section": "news",
                    "item_title": "Post-Launch Patch",
                    "chunk_id": "news",
                    "source_date": "2026-06-10",
                    "relevance_type": "valid_update_or_patch",
                },
            ),
            Document(
                "Recent players praise the weapon variety and combat loop.",
                {
                    "game_key": "hades_ii_1145350",
                    "game_name": "Hades II",
                    "section": "review",
                    "item_title": "Review 1",
                    "chunk_id": "review",
                    "source_date": "2026-06-28",
                },
            ),
        ]
        embedder = FakeEmbedder()
        index = build_index(documents, embedder)
        generator = FakeAgenticGenerator()
        pipeline = RAGPipeline(index, embedder, generator)

        results, metadata = pipeline.search_agentic(
            "appid: 1145350 Hades II는 2.5D 쿼터뷰 액션 로그라이크야? 최근 업데이트와 리뷰도 포함해줘.",
            k=3,
            max_steps=3,
        )

        sections = {result.document.metadata["section"] for result in results}
        self.assertTrue({"metadata", "news", "review"} <= sections)
        self.assertFalse(metadata["sufficient"])
        self.assertLess(metadata["evidence_coverage"]["coverage_ratio"], 1.0)
        self.assertTrue(
            any(
                claim["claim_id"] == "facet_dimension_facets_2_5d" and not claim["supported"]
                for claim in metadata["evidence_coverage"]["claims"]
            )
        )


if __name__ == "__main__":
    unittest.main()
