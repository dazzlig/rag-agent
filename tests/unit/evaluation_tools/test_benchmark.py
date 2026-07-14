from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import _bootstrap  # noqa: F401
from steam_rag.evaluation_tools.benchmark import (
    GoldenCase,
    Stage4BenchmarkRunner,
    load_golden_set,
    save_benchmark,
    summarize_records,
    score_case,
)
from steam_rag.common.models import Document, SearchResult
from steam_rag.rag_search.hybrid_retriever import HybridTimeAwareRetriever
from steam_rag.rag_search.search_spec import evaluate_evidence_coverage
from steam_rag.rag_search.vector_store import build_index


class FakeEmbedder:
    model_name = "fake"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, float("patch" in text.casefold()), float(len(text) % 7)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeGenerator:
    def generate(self, question: str, results: Sequence[SearchResult]) -> str:
        return "패치와 평가를 설명합니다. [근거 1]"

    def generate_agentic(self, question: str, results: Sequence[SearchResult], metadata: dict) -> str:
        return self.generate(question, results)

    def generate_hyde(self, question: str, search_query: str, reason: str) -> str:
        return "hypothetical patch review evidence"


class FakeReranker:
    model_name = "fake-reranker"

    def rerank(self, question: str, results: Sequence[SearchResult], *, top_n: int) -> list[SearchResult]:
        output = list(results)[:top_n]
        for rank, result in enumerate(output, start=1):
            result.rank = rank
            result.rerank_score = 1.0 / rank
        return output


class Stage4EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document(
                "Post-Launch Patch hotfix update fixes controls.",
                {
                    "appid": 1145350,
                    "game_key": "hades_ii_1145350",
                    "game_name": "Hades II",
                    "section": "news",
                    "item_title": "Patch",
                    "chunk_id": "news",
                    "source_date": "2026-06-10",
                },
            ),
            Document(
                "Recent review positive combat and story evaluation.",
                {
                    "appid": 1145350,
                    "game_key": "hades_ii_1145350",
                    "game_name": "Hades II",
                    "section": "review",
                    "item_title": "Review",
                    "chunk_id": "review",
                    "source_date": "2026-06-12",
                },
            ),
            Document(
                "patch_date 2026-06-10 before_sample_size 9 after_sample_size 11 positive_ratio_delta 0 confidence insufficient",
                {
                    "appid": 1145350,
                    "game_key": "hades_ii_1145350",
                    "game_name": "Hades II",
                    "section": "analysis",
                    "item_title": "Patch Impact Analysis",
                    "chunk_id": "analysis",
                    "patch_date": "2026-06-10",
                },
            ),
        ]
        self.embedder = FakeEmbedder()
        self.index = build_index(self.documents, self.embedder)

    def test_golden_set_contains_exactly_fifty_unique_cases(self) -> None:
        cases = load_golden_set(Path("data/eval/stage4_golden_set.jsonl"))
        self.assertEqual(len(cases), 50)
        self.assertEqual(len({case.case_id for case in cases}), 50)
        self.assertGreaterEqual(len({case.category for case in cases}), 6)

    def test_search_spec_and_claim_coverage_are_claim_level(self) -> None:
        retriever = HybridTimeAwareRetriever(self.index)
        question = "Hades II 최신 패치 이후 리뷰 평가는 좋아졌어?"
        spec = retriever.build_search_spec(question)
        results = [SearchResult(self.documents[0], 1.0, rank=1)]
        report = evaluate_evidence_coverage(spec, results)

        self.assertIn("analysis", spec.primary_sections)
        self.assertIn("review", spec.primary_sections)
        self.assertIn("review_change_after_update", {claim.claim_id for claim in spec.claims})
        self.assertGreaterEqual(report.claim_count, 2)
        self.assertLess(report.coverage_ratio, 1.0)
        self.assertTrue(any(claim.claim_id == "player_sentiment" and not claim.supported for claim in report.claims))

    def test_agentic_and_hyde_are_the_default_stage4_comparison(self) -> None:
        case = GoldenCase(
            "T1",
            "after_update",
            "Hades II 최신 패치 이후 평가는?",
            "after_update",
            ("hades_ii_1145350",),
            ("analysis",),
            ("positive_ratio",),
            ("패치", "평가"),
            (),
            "2026-06-10",
        )
        runner = Stage4BenchmarkRunner(
            self.index,
            self.embedder,
            generator=FakeGenerator(),
            reranker=FakeReranker(),
            top_k=3,
        )
        records = runner.run([case])

        self.assertEqual({record.strategy for record in records}, {"agentic", "agentic_hyde"})
        for record in records:
            self.assertEqual(
                set(record.metrics),
                {"retrieval", "generation", "citation", "temporal", "recommendation", "operations"},
            )
        summary = summarize_records(records)
        self.assertEqual(len(summary), 2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_benchmark(records, details_path=root / "details.jsonl", summary_path=root / "summary.csv")
            self.assertTrue((root / "details.jsonl").exists())
            self.assertIn("retrieval.claim_evidence_coverage", (root / "summary.csv").read_text(encoding="utf-8-sig"))

    def test_grouped_citations_count_for_claim_coverage(self) -> None:
        case = GoldenCase(
            "T2", "gameplay", "Hades II 전투는?", "gameplay",
            ("hades_ii_1145350",), ("analysis",), (), ("전투",), (), "", "",
        )
        retriever = HybridTimeAwareRetriever(self.index)
        spec = retriever.build_search_spec(case.question)
        result = SearchResult(self.documents[2], 1.0, rank=1)

        metrics = score_case(case, spec, [result], "전투는 실시간입니다. [근거 1, 2]", 1.0)

        self.assertEqual(metrics["citation"]["claim_citation_coverage"], 1.0)
        self.assertEqual(metrics["citation"]["citation_validity"], 0.5)


if __name__ == "__main__":
    unittest.main()
