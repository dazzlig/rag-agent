from __future__ import annotations

import unittest
from datetime import date

import _bootstrap  # noqa: F401
from steam_rag.common.models import Document
from steam_rag.rag_search.hybrid_retriever import HybridTimeAwareRetriever, detect_intent
from steam_rag.rag_search.vector_store import VectorIndex


def document(content: str, **metadata: object) -> Document:
    return Document(content, {"game_key": "hollow_knight", **metadata})


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            document(
                "Old patch notes with bug fixes and balance changes.",
                section="news",
                item_title="Patch 1",
                source_date="2025-01-01",
                relevance_type="valid_update_or_patch",
                chunk_id="old-news",
            ),
            document(
                "Latest patch now live with performance fixes and balance changes.",
                section="news",
                item_title="Patch 2 Now Live",
                source_date="2026-05-20",
                relevance_type="valid_update_or_patch",
                chunk_id="new-news",
            ),
            document(
                "Before the patch, players reported several performance problems.",
                section="review",
                item_title="Review Before",
                source_date="2026-04-01",
                weighted_vote_score="0.8",
                chunk_id="review-before",
            ),
            document(
                "After the patch, players praise the improved performance and combat.",
                section="review",
                item_title="Review After",
                source_date="2026-06-01",
                weighted_vote_score="0.8",
                chunk_id="review-after",
            ),
            document(
                "Explore a hand-drawn world with precise combat and interconnected areas.",
                section="about",
                item_title="About The Game",
                chunk_id="about",
            ),
        ]
        embeddings = [[1.0, 0.0] for _ in self.documents]
        index = VectorIndex(self.documents, embeddings, "fake-embedding")
        self.retriever = HybridTimeAwareRetriever(index, reference_date=date(2026, 6, 21))

    def test_latest_patch_wins_news_query(self) -> None:
        results = self.retriever.retrieve("할로우 나이트 최근 패치 알려줘", [1.0, 0.0], k=2)
        self.assertEqual(results[0].document.metadata["item_title"], "Patch 2 Now Live")
        self.assertGreater(results[0].relative_recency_score, results[1].relative_recency_score)

    def test_sale_news_is_not_treated_as_patch_evidence(self) -> None:
        documents = [
            document(
                "Summer sale discount is live now. Wishlist and buy with a deal.",
                section="news",
                item_title="Summer Sale",
                source_date="2026-06-20",
                news_type="sale_promo",
                relevance_type="store_or_sales_related",
                chunk_id="sale-news",
            ),
            document(
                "Patch notes with hotfixes and balance changes.",
                section="news",
                item_title="Hotfix Patch",
                source_date="2026-06-01",
                news_type="hotfix",
                relevance_type="valid_update_or_patch",
                chunk_id="patch-news",
            ),
        ]
        retriever = HybridTimeAwareRetriever(
            VectorIndex(documents, [[1.0, 0.0], [1.0, 0.0]], "fake-embedding"),
            reference_date=date(2026, 6, 21),
        )

        patch_results = retriever.retrieve("할로우 나이트 최근 패치 알려줘", [1.0, 0.0], k=2)
        price_results = retriever.retrieve("할로우 나이트 할인 가격 알려줘", [1.0, 0.0], k=2)

        self.assertEqual(patch_results[0].document.metadata["item_title"], "Hotfix Patch")
        self.assertEqual(price_results[0].document.metadata["item_title"], "Summer Sale")

    def test_price_query_prefers_metadata_price_fields(self) -> None:
        documents = [
            document(
                "Price metadata.",
                section="metadata",
                item_title="Metadata",
                chunk_id="metadata",
                price_available="True",
                price_currency="USD",
                price_final_formatted="$14.99",
                price_discount_percent="50",
                price_collected_at="2026-06-30T00:00:00+00:00",
            ),
            document(
                "General gameplay information.",
                section="about",
                item_title="About",
                chunk_id="about-price-test",
            ),
        ]
        retriever = HybridTimeAwareRetriever(VectorIndex(documents, [[1.0, 0.0], [1.0, 0.0]], "fake-embedding"))

        results = retriever.retrieve("할로우 나이트 지금 할인 가격은?", [1.0, 0.0], k=1)

        self.assertEqual(results[0].document.metadata["section"], "metadata")
        self.assertGreater(results[0].content_bonus, 0)

    def test_after_update_query_prefers_post_patch_review(self) -> None:
        self.assertEqual(detect_intent("업데이트 이후 유저 평가가 어때?"), "after_update")
        results = self.retriever.retrieve(
            "할로우 나이트 업데이트 이후 유저 평가가 어때?", [1.0, 0.0], k=5
        )
        reviews = [result for result in results if result.document.metadata["section"] == "review"]
        self.assertEqual(reviews[0].document.metadata["item_title"], "Review After")
        self.assertEqual(reviews[0].latest_patch_date, "2026-05-20")

    def test_after_update_query_prefers_structured_analysis(self) -> None:
        documents = self.documents + [
            document(
                "패치 전 60개 리뷰 긍정률 50%, 패치 후 60개 리뷰 긍정률 90%, +40%p improved high confidence.",
                section="analysis",
                item_title="Patch Impact Analysis",
                source_date="2026-05-20",
                patch_date="2026-05-20",
                before_sample_size="60",
                after_sample_size="60",
                positive_ratio_delta_pp="40.0",
                change_direction="improved",
                confidence_label="high",
                chunk_id="patch-analysis",
            )
        ]
        retriever = HybridTimeAwareRetriever(
            VectorIndex(documents, [[1.0, 0.0] for _ in documents], "fake-embedding"),
            reference_date=date(2026, 6, 21),
        )

        results = retriever.retrieve(
            "할로우 나이트 업데이트 이후 평가가 좋아졌어?", [1.0, 0.0], k=3
        )

        self.assertEqual(results[0].document.metadata["section"], "analysis")
        self.assertGreater(results[0].content_bonus, 0.9)

    def test_stale_analysis_does_not_override_latest_patch_reviews(self) -> None:
        documents = self.documents + [
            document(
                "Old structured patch analysis.",
                section="analysis",
                item_title="Patch Impact Analysis",
                source_date="2025-01-01",
                patch_date="2025-01-01",
                change_direction="improved",
                confidence_label="high",
                chunk_id="stale-analysis",
            )
        ]
        retriever = HybridTimeAwareRetriever(
            VectorIndex(documents, [[1.0, 0.0] for _ in documents], "fake-embedding"),
            reference_date=date(2026, 6, 21),
        )

        results = retriever.retrieve(
            "할로우 나이트 업데이트 이후 평가가 좋아졌어?", [1.0, 0.0], k=3
        )

        self.assertNotEqual(results[0].document.metadata["section"], "analysis")

    def test_gameplay_query_uses_about_section(self) -> None:
        results = self.retriever.retrieve("할로우 나이트 전투와 플레이 특징", [1.0, 0.0], k=1)
        self.assertEqual(results[0].document.metadata["section"], "about")

    def test_explicit_appid_limits_retrieval_to_that_game(self) -> None:
        documents = [
            Document(
                "PEAK online co-op climbing survival gameplay.",
                {"appid": "3527290", "game_key": "peak_3527290", "game_name": "PEAK", "section": "about", "chunk_id": "peak"},
            ),
            Document(
                "Hades II single-player action roguelike gameplay.",
                {"appid": "1145350", "game_key": "hades_ii_1145350", "game_name": "Hades II", "section": "about", "chunk_id": "hades"},
            ),
        ]
        retriever = HybridTimeAwareRetriever(
            VectorIndex(documents, [[1.0, 0.0], [1.0, 0.0]], "fake-embedding")
        )

        results = retriever.retrieve(
            "appid: 3527290 친구와 협동 플레이", [1.0, 0.0], k=2
        )

        self.assertTrue(results)
        self.assertTrue(all(result.document.metadata["appid"] == "3527290" for result in results))

    def test_game_name_with_korean_particle_limits_retrieval(self) -> None:
        documents = [
            Document(
                "PEAK online co-op climbing survival gameplay.",
                {"appid": "3527290", "game_key": "peak_3527290", "game_name": "PEAK", "section": "about", "chunk_id": "peak-name"},
            ),
            Document(
                "Hades II single-player action roguelike gameplay.",
                {"appid": "1145350", "game_key": "hades_ii_1145350", "game_name": "Hades II", "section": "about", "chunk_id": "hades-name"},
            ),
        ]
        retriever = HybridTimeAwareRetriever(
            VectorIndex(documents, [[1.0, 0.0], [1.0, 0.0]], "fake-embedding")
        )

        results = retriever.retrieve(
            "PEAK는 친구와 협동하기 좋아?", [1.0, 0.0], k=2
        )

        self.assertTrue(results)
        self.assertTrue(all(result.document.metadata["appid"] == "3527290" for result in results))

    def test_explicit_playstyle_facets_affect_ranking(self) -> None:
        documents = [
            Document(
                "A three dimensional action role-playing game.",
                {
                    "game_key": "matching_game",
                    "section": "about",
                    "item_title": "About",
                    "chunk_id": "matching",
                    "combat_facets": ["real_time", "direct_control"],
                    "perspective_facets": ["third_person"],
                    "dimension_facets": ["3d"],
                    "playstyle_facets": ["character_progression"],
                },
            ),
            Document(
                "A two dimensional turn-based role-playing game.",
                {
                    "game_key": "conflicting_game",
                    "section": "about",
                    "item_title": "About",
                    "chunk_id": "conflicting",
                    "combat_facets": ["turn_based", "command_based"],
                    "perspective_facets": ["side_view"],
                    "dimension_facets": ["2d"],
                    "playstyle_facets": ["character_progression"],
                },
            ),
        ]
        retriever = HybridTimeAwareRetriever(
            VectorIndex(documents, [[1.0, 0.0], [1.0, 0.0]], "fake-embedding")
        )
        results = retriever.retrieve(
            "3D third-person direct control action RPG gameplay", [1.0, 0.0], k=2
        )
        self.assertEqual(results[0].document.metadata["game_key"], "matching_game")
        self.assertGreater(results[0].facet_score, results[1].facet_score)
        self.assertIn("dimension_facets:3d", results[0].matched_facets)


if __name__ == "__main__":
    unittest.main()
