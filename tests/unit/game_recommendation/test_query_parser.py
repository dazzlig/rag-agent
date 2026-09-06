from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401
from steam_rag.game_recommendation.query_parser import (
    OpenAIRecommendationQueryStructurer,
    RecommendationProfileIndex,
    RecommendationQuery,
    parse_recommendation_query,
)


def profile(
    appid: int,
    name: str,
    *,
    tags: list[tuple[str, int]],
    genres: list[str] | None = None,
    categories: list[str] | None = None,
    combat: list[str] | None = None,
    perspective: list[str] | None = None,
    dimension: list[str] | None = None,
    playstyle: list[str] | None = None,
    inferred: dict[str, list[str]] | None = None,
    positive_ratio: float | None = None,
    price_final: int = 30_000 * 100,
    discount_percent: int = 0,
    is_free: bool = False,
    app_type: str = "game",
    release_coming_soon: bool = False,
) -> dict:
    return {
        "appid": appid,
        "name": name,
        "app_type": app_type,
        "release_coming_soon": release_coming_soon,
        "steam_genres_normalized": genres or [],
        "steam_categories_normalized": categories or [],
        "popular_user_tags": [
            {"name": tag, "normalized": tag, "rank": rank} for tag, rank in tags
        ],
        "combat_facets": combat or [],
        "perspective_facets": perspective or [],
        "dimension_facets": dimension or [],
        "playstyle_facets": playstyle or [],
        "inferred_facets": inferred or {},
        "recent_review_summary": {"positive_ratio": positive_ratio},
        "price": {
            "currency": "KRW",
            "final": price_final,
            "discount_percent": discount_percent,
            "is_free": is_free,
        },
        "searchable_terms": [tag for tag, _ in tags],
    }


class RecommendationTests(unittest.TestCase):
    def test_rule_parser_structures_korean_conditions(self) -> None:
        query = parse_recommendation_query(
            "3D 3인칭 액션 RPG 중 최근 평가가 좋고 4만원 이하인 싱글 플레이어 게임"
        )

        self.assertEqual(query.genres, ["action", "rpg"])
        self.assertIn("action_rpg", query.required_tags)
        self.assertIn("third_person", query.perspective)
        self.assertIn("3d", query.dimension)
        self.assertIn("singleplayer", query.categories)
        self.assertTrue(query.recent_rating_required)
        self.assertEqual(query.price_max_krw, 40_000)

    def test_explicitly_rejected_condition_becomes_an_exclusion(self) -> None:
        """기획안 4.1: '턴제보다는 액션'을 턴제 요구로 읽지 않는다."""

        query = parse_recommendation_query(
            "그림체가 예쁘고 스토리가 있는 게임을 찾고 있어. 턴제보다는 직접 움직이고 공격하는 액션이 좋아."
        )

        self.assertNotIn("turn_based", query.combat)
        self.assertIn("turn_based_combat", query.excluded_conditions)
        self.assertEqual(query.combat, ["direct_control"])

    def test_action_alone_does_not_confirm_real_time_combat(self) -> None:
        query = parse_recommendation_query("액션 게임 추천해줘")

        self.assertNotIn("real_time", query.combat)

    def test_negation_forms_are_recognized(self) -> None:
        for question, excluded in (
            ("턴제 전투 말고 실시간 액션 RPG 추천", "turn_based_combat"),
            ("협동은 싫어. 혼자 하는 스토리 게임 추천", "co_op"),
            ("로그라이크는 별로야. 스토리 게임 추천", "roguelike"),
        ):
            with self.subTest(question=question):
                self.assertIn(excluded, parse_recommendation_query(question).excluded_conditions)

    def test_positive_turn_based_request_is_still_a_requirement(self) -> None:
        query = parse_recommendation_query("턴제 RPG 추천해줘")

        self.assertIn("turn_based", query.combat)
        self.assertEqual(query.excluded_conditions, [])

    def test_rule_parser_preserves_sale_and_upcoming_requirements(self) -> None:
        sale = parse_recommendation_query("현재 세일 중인 스토리 RPG 5개 추천")
        upcoming = parse_recommendation_query("앞으로 나올 신작 RPG 기대작 5개")
        playable = parse_recommendation_query("Steam에서 지금 할 수 있는 서브컬처 게임 추천")

        self.assertTrue(sale.sale_required)
        self.assertTrue(upcoming.upcoming_required)
        self.assertTrue(playable.currently_playable_required)

    def test_hard_filter_and_top20_to_top5_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(25):
                payload = profile(
                    1000 + index,
                    f"Game {index:02d}",
                    tags=[("action_rpg", index % 10 + 1), ("third_person", 2), ("3d", 3)],
                    genres=["action", "rpg"],
                    categories=["singleplayer"],
                    combat=["real_time"],
                    perspective=["third_person"],
                    dimension=["3d"],
                    playstyle=["character_progression"],
                    positive_ratio=0.70 + index / 100,
                )
                (root / f"game_{index}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            rejected = profile(
                9999,
                "First Person Game",
                tags=[("action_rpg", 1), ("first_person", 2), ("3d", 3)],
                genres=["action", "rpg"],
                perspective=["first_person"],
                dimension=["3d"],
                combat=["real_time"],
            )
            (root / "rejected.json").write_text(json.dumps(rejected), encoding="utf-8")

            query = RecommendationQuery(
                genres=["action", "rpg"],
                required_tags=["action_rpg", "third_person", "3d"],
                combat=["real_time"],
                perspective=["third_person"],
                dimension=["3d"],
                recent_rating_required=True,
            )
            selection = RecommendationProfileIndex.load(root).search(
                "question", query, candidate_limit=20, detail_limit=5
            )

            self.assertEqual(selection.scanned_profiles, 26)
            self.assertEqual(selection.hard_filter_matches, 25)
            self.assertEqual(len(selection.candidates), 20)
            self.assertEqual(len(selection.detail_targets), 5)
            self.assertEqual(selection.detail_targets[0].name, "Game 20")
            self.assertNotIn("First Person Game", [item.name for item in selection.candidates])

    def test_rare_high_rank_tag_receives_more_weight(self) -> None:
        profiles = [
            (
                Path("rare.json"),
                profile(1, "Rare", tags=[("party_based_rpg", 1), ("rpg", 2)]),
            ),
            (
                Path("common.json"),
                profile(2, "Common", tags=[("rpg", 1)]),
            ),
            (
                Path("common2.json"),
                profile(3, "Common 2", tags=[("rpg", 1)]),
            ),
        ]
        index = RecommendationProfileIndex(profiles)
        selection = index.search("RPG 추천", RecommendationQuery(), candidate_limit=3)

        self.assertEqual(selection.candidates[0].name, "Rare")
        self.assertGreater(
            selection.candidates[0].tag_rank_rarity_score,
            selection.candidates[1].tag_rank_rarity_score,
        )

    def test_llm_structurer_returns_parsed_schema(self) -> None:
        parsed = RecommendationQuery(genres=["RPG"], perspective=["third_person"])
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    parse=lambda **_: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
                    )
                )
            )
        )
        result = OpenAIRecommendationQueryStructurer(client=fake_client).structure("추천")

        self.assertEqual(result.genres, ["rpg"])
        self.assertEqual(result.perspective, ["third_person"])

    def test_allowed_appids_confines_ranking_to_verified_web_candidates(self) -> None:
        profiles = [
            (Path("verified.json"), profile(1, "Verified", tags=[("rpg", 1)])),
            (Path("legacy.json"), profile(2, "Legacy Winner", tags=[("rpg", 1)])),
        ]

        selection = RecommendationProfileIndex(profiles).search(
            "웹에서 찾은 후보만 추천",
            RecommendationQuery(required_tags=["rpg"]),
            allowed_appids={1},
        )

        self.assertEqual([item.appid for item in selection.candidates], [1])

    def test_sale_upcoming_and_game_type_are_hard_filters(self) -> None:
        profiles = [
            (Path("sale.json"), profile(1, "Sale Game", tags=[], discount_percent=30)),
            (Path("full.json"), profile(2, "Full Price", tags=[])),
            (Path("upcoming.json"), profile(3, "NTE", tags=[], release_coming_soon=True)),
            (Path("dlc.json"), profile(4, "Expansion", tags=[], app_type="dlc", discount_percent=50)),
            (Path("generic.json"), profile(5, "Games", tags=[], discount_percent=50)),
            (
                Path("free.json"),
                profile(
                    6,
                    "Free To Play",
                    tags=[],
                    discount_percent=100,
                    is_free=True,
                    price_final=0,
                ),
            ),
        ]
        index = RecommendationProfileIndex(profiles)

        sale = index.search("현재 세일", RecommendationQuery(sale_required=True))
        upcoming = index.search("출시 예정", RecommendationQuery(upcoming_required=True))
        playable = index.search(
            "Steam에서 지금 할 수 있는 게임",
            RecommendationQuery(currently_playable_required=True),
        )

        self.assertEqual([item.appid for item in sale.candidates], [1])
        self.assertEqual([item.appid for item in upcoming.candidates], [3])
        self.assertNotIn(3, [item.appid for item in playable.candidates])
        self.assertIn(6, [item.appid for item in playable.candidates])


if __name__ == "__main__":
    unittest.main()
