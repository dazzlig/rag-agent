from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401
from steam_rag.game_recommendation.similarity_ranker import (
    adapt_similarity_spec_to_question,
    build_similarity_spec,
    canonicalize_semantic_tag,
    rank_similar_profiles,
    resolve_reference_game,
    score_profile_similarity,
)


def profile(
    appid: int,
    name: str,
    tags: list[str],
    *,
    combat: list[str] | None = None,
    perspective: list[str] | None = None,
    dimension: list[str] | None = None,
    playstyle: list[str] | None = None,
    genres: list[str] | None = None,
    is_free: bool = False,
) -> dict:
    return {
        "appid": appid,
        "name": name,
        "app_type": "game",
        "popular_user_tags": [
            {"name": tag, "normalized": tag, "rank": rank}
            for rank, tag in enumerate(tags, start=1)
        ],
        "steam_tags_normalized": tags,
        "steam_genres_normalized": genres or [],
        "steam_categories_normalized": [],
        "combat_facets": combat or [],
        "perspective_facets": perspective or [],
        "dimension_facets": dimension or [],
        "playstyle_facets": playstyle or [],
        "inferred_facets": {},
        "price": {"is_free": is_free},
        "store_summary": "",
    }


WUTHERING_WAVES = profile(
    3513350,
    "Wuthering Waves",
    ["오픈 월드", "애니메이션", "무료 플레이", "액션", "RPG", "탐험", "3인칭"],
    genres=["액션", "RPG"],
    combat=["real_time", "direct_control", "melee"],
    perspective=["third_person"],
    dimension=["3d"],
    playstyle=["open_world", "exploration", "story_rich"],
    is_free=True,
)

TOWER_OF_FANTASY = profile(
    2064650,
    "Tower of Fantasy",
    ["Anime", "Open World", "Free to Play", "Action RPG", "Exploration", "Third Person"],
    genres=["Action", "RPG"],
    combat=["real_time", "direct_control", "melee"],
    perspective=["third_person"],
    dimension=["3d"],
    playstyle=["open_world", "exploration"],
    is_free=True,
)

CORE_KEEPER = profile(
    1621690,
    "Core Keeper",
    ["Sandbox", "Survival", "Crafting", "Pixel Graphics", "Co-op", "Exploration"],
    genres=["Adventure", "Indie"],
    combat=["real_time"],
    perspective=["top_down"],
    dimension=["2d"],
    playstyle=["survival", "crafting", "co_op", "exploration"],
)


class SimilarityTests(unittest.TestCase):
    def test_semantic_tag_canonicalization_handles_korean_and_english(self) -> None:
        self.assertEqual(canonicalize_semantic_tag("애니메이션"), "anime")
        self.assertEqual(canonicalize_semantic_tag("Anime Style"), "anime")
        self.assertEqual(canonicalize_semantic_tag("무료 플레이"), "free_to_play")
        self.assertEqual(canonicalize_semantic_tag("Real-Time Combat"), "real_time")

    def test_korean_service_alias_resolves_only_to_local_verified_profile(self) -> None:
        resolved = resolve_reference_game(
            "명조 같은 서브컬처 게임 중 Steam 게임 추천해줘",
            [CORE_KEEPER, WUTHERING_WAVES],
        )

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.appid, 3513350)
        self.assertEqual(resolved.name, "Wuthering Waves")
        self.assertEqual(resolved.source, "high_confidence_alias")

        self.assertIsNone(resolve_reference_game("명조 같은 게임", [CORE_KEEPER]))

    def test_generic_subtitle_alias_resolves_without_a_curated_game_profile(self) -> None:
        expedition = profile(1903340, "클레르 옵스퀴르: 33 원정대", ["RPG"])

        resolved = resolve_reference_game("33 원정대 같은 RPG 추천", [expedition])

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.appid, 1903340)
        self.assertEqual(resolved.source, "generated_title_alias")

    def test_wuthering_spec_uses_evidence_derived_hard_gates(self) -> None:
        reference = resolve_reference_game("명조 같은 게임", [WUTHERING_WAVES])
        assert reference is not None

        spec = build_similarity_spec(WUTHERING_WAVES, reference=reference)

        self.assertEqual(spec.must_have, ("anime", "rpg", "real_time"))
        self.assertIn("open_world", spec.should_have)
        self.assertIn("third_person", spec.should_have)
        self.assertIn("survival_sandbox", spec.excluded)

    def test_tower_of_fantasy_ranks_above_core_keeper_and_core_fails_gate(self) -> None:
        spec = build_similarity_spec(WUTHERING_WAVES)

        tower = score_profile_similarity(TOWER_OF_FANTASY, spec)
        core = score_profile_similarity(CORE_KEEPER, spec)
        ranked = rank_similar_profiles([CORE_KEEPER, TOWER_OF_FANTASY], spec)

        self.assertTrue(tower.passed_hard_gate)
        self.assertFalse(core.passed_hard_gate)
        self.assertIn("anime", core.missing_must_have)
        self.assertIn("survival_sandbox", core.excluded_matches)
        self.assertEqual([item.name for item in ranked], ["Tower of Fantasy"])
        self.assertGreater(tower.score, core.score)
        self.assertIn("애니풍 비주얼", tower.matched_aspects)
        self.assertIn("실시간 액션 전투", tower.matched_aspects)

    def test_subculture_word_keeps_identity_gate_but_softens_combat_gate(self) -> None:
        spec = adapt_similarity_spec_to_question(
            build_similarity_spec(WUTHERING_WAVES),
            "명조 같은 서브컬처 게임 추천해줘",
        )

        self.assertEqual(spec.must_have, ("anime", "rpg"))
        self.assertIn("real_time", spec.should_have)
        self.assertNotIn("turn_based", spec.excluded)
        self.assertIn("character_collection", spec.should_have)

    def test_followup_delta_preserves_seed_contract_and_adds_new_constraints(self) -> None:
        initial = adapt_similarity_spec_to_question(
            build_similarity_spec(WUTHERING_WAVES),
            "명조 같은 서브컬처 게임 추천해줘",
        )

        refined = adapt_similarity_spec_to_question(
            initial,
            "그중 협동 가능한 것만, 턴제는 빼고 골라줘",
        )

        self.assertIn("anime", refined.must_have)
        self.assertIn("rpg", refined.must_have)
        self.assertIn("co_op", refined.must_have)
        self.assertIn("turn_based", refined.excluded)


if __name__ == "__main__":
    unittest.main()
