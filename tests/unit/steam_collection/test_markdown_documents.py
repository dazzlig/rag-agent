from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from steam_rag.steam_collection.markdown_documents import chunk_documents, parse_markdown


SAMPLE = """# Example Game

## Metadata
- game_key: example_game
- appid: 123
- name: Example Game
- release_date: Jan 1, 2024
- genres: ['Action']
- categories: ['Single-player', 'Online Co-op']
- steam_tags: ['3D', 'Third Person']
- playstyle_profile_source: curated_mvp_v1
- playstyle_profile_confidence: high
- dimension_facets: ['2d']

## Store Summary
This is a sufficiently detailed store summary for the example game.

## Recent Steam Reviews

### Review 1
- review_created_at: 2026-06-01
- sentiment: positive
- weighted_vote_score: 0.8

The combat and exploration are excellent and work well together.

## Steam News

### News 1: Patch 2 Now Live
- news_date: 2026-05-20
- relevance_type: valid_update_or_patch

Patch 2 improves performance and fixes several combat issues.
"""


class IngestTests(unittest.TestCase):
    def test_metadata_and_dates_survive_parsing_and_chunking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example_game.md"
            path.write_text(SAMPLE, encoding="utf-8")
            documents = parse_markdown(path)

        by_section = {document.metadata["section"]: document for document in documents}
        self.assertEqual(by_section["review"].metadata["source_date"], "2026-06-01")
        self.assertEqual(by_section["news"].metadata["source_date"], "2026-05-20")
        self.assertEqual(by_section["news"].metadata["relevance_type"], "valid_update_or_patch")
        self.assertIn("action", by_section["store_summary"].metadata["steam_genres_normalized"])
        self.assertIn("co_op", by_section["store_summary"].metadata["playstyle_facets"])
        self.assertEqual(by_section["store_summary"].metadata["dimension_facets"], ["3d"])
        self.assertEqual(
            by_section["store_summary"].metadata["playstyle_profile_source"],
            "steam_popular_tags_store_text_and_reviews",
        )
        chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=20)
        self.assertTrue(all(chunk.metadata.get("chunk_id") for chunk in chunks))
        self.assertTrue(all(len(chunk.page_content) <= 100 for chunk in chunks))

    def test_invalid_chunk_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_documents([], chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    unittest.main()
