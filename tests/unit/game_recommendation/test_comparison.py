from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401
from steam_rag.game_recommendation.comparison import (
    COMPARISON_AXES,
    compare_profiles,
    comparison_markdown,
)


def profile(appid: int, name: str, **overrides) -> dict:
    payload = {
        "appid": appid,
        "name": name,
        "combat_facets": [],
        "perspective_facets": [],
        "dimension_facets": [],
        "playstyle_facets": [],
        "popular_user_tags": [],
    }
    payload.update(overrides)
    return payload


class ComparisonTests(unittest.TestCase):
    def test_axes_describe_experience_not_genre_names(self) -> None:
        labels = [axis.label for axis in COMPARISON_AXES]

        self.assertEqual(
            labels,
            ["전투 조작", "화면 시점", "표현 방식", "스토리 진행", "반복 요소", "난도 성향", "협동 방식"],
        )

    def test_missing_data_is_reported_as_unverified_instead_of_guessed(self) -> None:
        table = compare_profiles(
            [
                profile(1, "A", combat_facets=["real_time", "direct_control"]),
                profile(2, "B"),
            ]
        )
        combat = next(axis for axis in table.axes if axis.axis_id == "combat_control")
        story = next(axis for axis in table.axes if axis.axis_id == "story")

        self.assertEqual(combat.cells[0].display, "실시간 전투 · 직접 조작")
        self.assertEqual(combat.cells[1].display, "미확인")
        self.assertEqual(story.unverified_games, ["A", "B"])

    def test_differing_axes_surface_the_choice_changing_difference(self) -> None:
        table = compare_profiles(
            [
                profile(1, "Real", combat_facets=["real_time"], perspective_facets=["third_person"]),
                profile(2, "Turn", combat_facets=["turn_based"], perspective_facets=["third_person"]),
            ]
        )

        self.assertIn("combat_control", [axis.axis_id for axis in table.differing_axes])
        self.assertNotIn("perspective", [axis.axis_id for axis in table.differing_axes])

    def test_popular_tags_answer_axes_without_facet_fields(self) -> None:
        table = compare_profiles(
            [
                profile(
                    1,
                    "Hard",
                    popular_user_tags=[{"name": "Difficult", "normalized": "difficult", "rank": 1}],
                ),
                profile(2, "Unknown"),
            ]
        )
        difficulty = next(axis for axis in table.axes if axis.axis_id == "difficulty")

        self.assertEqual(difficulty.cells[0].display, "높은 난도")
        self.assertTrue(difficulty.cells[0].verified)
        self.assertFalse(difficulty.cells[1].verified)

    def test_markdown_needs_two_games_and_names_unverified_axes(self) -> None:
        single = compare_profiles([profile(1, "Only")])
        pair = compare_profiles(
            [profile(1, "A", combat_facets=["real_time"]), profile(2, "B", combat_facets=["turn_based"])]
        )
        rendered = comparison_markdown(pair)

        self.assertIn("2개 이상", comparison_markdown(single))
        self.assertIn("| 비교 축 | A | B |", rendered)
        self.assertIn("아직 확인하지 못한 축", rendered)
        self.assertIn("선택을 바꿀 수 있는 차이: 전투 조작", rendered)

    def test_comparison_is_capped_at_three_games(self) -> None:
        table = compare_profiles([profile(index, f"G{index}") for index in range(1, 6)])

        self.assertEqual(len(table.games), 3)


if __name__ == "__main__":
    unittest.main()
