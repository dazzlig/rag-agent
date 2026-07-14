from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401
from steam_rag.game_metadata.playstyle import (
    build_playstyle_metadata,
    extract_query_facets,
    facet_match_score,
    normalize_steam_tags,
)


class PlaystyleTests(unittest.TestCase):
    def test_steam_tags_are_normalized(self) -> None:
        tags = normalize_steam_tags(["Action RPG", "Third-Person", "Online Co-op", "Souls-like"])
        self.assertEqual(tags, ["action_rpg", "online_co_op", "souls_like", "third_person"])

    def test_korean_query_is_mapped_to_facets(self) -> None:
        facets = extract_query_facets("3D 3인칭 시점에서 직접 조작하는 실시간 액션 RPG")
        self.assertIn("3d", facets["dimension_facets"])
        self.assertIn("third_person", facets["perspective_facets"])
        self.assertIn("direct_control", facets["combat_facets"])
        self.assertIn("real_time", facets["combat_facets"])

    def test_metadata_is_inferred_from_real_steam_sources(self) -> None:
        metadata = build_playstyle_metadata(
            "Explore an open world with co-op play.",
            genres=["RPG"],
            categories=["Online Co-op"],
            steam_tags=["3D", "Third Person"],
            includes_reviews=True,
        )
        self.assertIn("rpg", metadata["steam_genres_normalized"])
        self.assertIn("co_op", metadata["playstyle_facets"])
        self.assertIn("3d", metadata["dimension_facets"])
        self.assertIn("third_person", metadata["perspective_facets"])
        self.assertEqual(
            metadata["playstyle_profile_source"],
            "steam_popular_tags_store_text_and_reviews",
        )
        self.assertIn("steam_recent_reviews", metadata["playstyle_evidence_sources"])

    def test_facet_score_rewards_matches_and_penalizes_conflicts(self) -> None:
        query = {"dimension_facets": ["3d"], "combat_facets": ["real_time"]}
        matched_score, matched, _ = facet_match_score(
            query, {"dimension_facets": ["3d"], "combat_facets": ["real_time"]}
        )
        conflict_score, _, conflicts = facet_match_score(
            query, {"dimension_facets": ["2d"], "combat_facets": ["turn_based"]}
        )
        self.assertGreater(matched_score, conflict_score)
        self.assertIn("dimension_facets:3d", matched)
        self.assertTrue(conflicts)


if __name__ == "__main__":
    unittest.main()
