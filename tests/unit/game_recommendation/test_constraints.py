from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401
from steam_rag.game_recommendation.constraints import (
    SATISFIED,
    UNVERIFIED,
    VIOLATED,
    evaluate_candidate_conditions,
    summarize_constraint_gate,
)
from pathlib import Path

from steam_rag.game_recommendation.query_parser import (
    RecommendationProfileIndex,
    RecommendationQuery,
    parse_recommendation_query,
)


def profile(appid: int, name: str, **overrides) -> dict:
    payload = {
        "appid": appid,
        "name": name,
        "app_type": "game",
        "steam_genres_normalized": ["action"],
        "steam_categories_normalized": [],
        "popular_user_tags": [{"name": "action", "normalized": "action", "rank": 1}],
        "combat_facets": [],
        "perspective_facets": [],
        "dimension_facets": [],
        "playstyle_facets": [],
        "collected_at": "2026-09-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


class ConstraintVerdictTests(unittest.TestCase):
    def test_confirmed_opposite_value_is_a_violation_not_a_miss(self) -> None:
        query = RecommendationQuery(combat=["real_time"])
        report = evaluate_candidate_conditions(
            profile(1, "Turn Based", combat_facets=["turn_based"]), query
        )

        self.assertEqual(report.status, VIOLATED)
        self.assertFalse(report.passes)
        self.assertIn("턴제 전투", report.must_violated[0].note)

    def test_missing_metadata_is_unverified_and_never_counts_as_satisfied(self) -> None:
        query = RecommendationQuery(combat=["real_time"])
        report = evaluate_candidate_conditions(profile(2, "Unknown"), query)

        self.assertEqual(report.status, UNVERIFIED)
        self.assertTrue(report.passes)
        self.assertFalse(report.fully_verified)
        self.assertEqual(report.fit_reasons, [])
        self.assertIn("전투 방식: 실시간 전투 미확인", report.checks_before_choosing)

    def test_official_evidence_marks_the_condition_confirmed(self) -> None:
        query = RecommendationQuery(combat=["turn_based"])
        report = evaluate_candidate_conditions(
            profile(
                3,
                "Confirmed",
                combat_facets=["turn_based"],
                facet_evidence=[
                    {
                        "facet_type": "combat_facets",
                        "facet": "turn_based",
                        "source_type": "steam_popular_user_tag",
                    }
                ],
            ),
            query,
        )

        self.assertEqual(report.status, SATISFIED)
        self.assertTrue(report.fully_verified)
        self.assertEqual(report.must_satisfied[0].confidence, "confirmed")
        self.assertEqual(report.checks_before_choosing, [])

    def test_store_text_only_evidence_stays_a_check_before_choosing(self) -> None:
        query = RecommendationQuery(combat=["real_time"])
        report = evaluate_candidate_conditions(
            profile(
                4,
                "Interpreted",
                combat_facets=["real_time"],
                facet_evidence=[
                    {
                        "facet_type": "combat_facets",
                        "facet": "real_time",
                        "source_type": "steam_store_text",
                    }
                ],
            ),
            query,
        )

        self.assertEqual(report.status, SATISFIED)
        self.assertEqual(report.must_satisfied[0].confidence, "interpreted")
        self.assertIn(
            "전투 방식: 실시간 전투 근거가 스토어 설명 해석뿐", report.checks_before_choosing
        )

    def test_excluded_condition_is_violated_only_when_confirmed_present(self) -> None:
        query = RecommendationQuery(excluded_conditions=["turn_based_combat"])
        present = evaluate_candidate_conditions(
            profile(
                5,
                "Has Turn Based",
                popular_user_tags=[
                    {"name": "Turn-Based Combat", "normalized": "turn_based_combat", "rank": 2}
                ],
            ),
            query,
        )
        absent = evaluate_candidate_conditions(profile(6, "No Turn Based"), query)

        self.assertFalse(present.passes)
        self.assertTrue(absent.passes)

    def test_information_status_lists_sources_and_unverified_items(self) -> None:
        query = RecommendationQuery(combat=["real_time"], recent_rating_required=True)
        report = evaluate_candidate_conditions(
            profile(
                7,
                "Partial",
                recent_review_summary={
                    "positive_ratio": 0.9,
                    "sample_size": 120,
                    "collected_at": "2026-09-02T00:00:00+00:00",
                },
            ),
            query,
        )
        status = report.information_status

        self.assertIn(
            {"source": "Steam 상품 정보", "checked_at": "2026-09-01T00:00:00+00:00"},
            status["checked_sources"],
        )
        self.assertEqual(status["review_sample_size"], 120)
        self.assertIn("전투 방식: 실시간 전투", status["unverified_items"])


class ConstraintGateSelectionTests(unittest.TestCase):
    def test_search_drops_violations_and_keeps_flagged_unverified_candidates(self) -> None:
        profiles = [
            (Path("real.json"), profile(1, "Real Time", combat_facets=["real_time"])),
            (Path("turn.json"), profile(2, "Turn Based", combat_facets=["turn_based"])),
            (Path("unknown.json"), profile(3, "Unknown Combat")),
        ]

        selection = RecommendationProfileIndex(profiles).search(
            "실시간 전투 액션 추천", RecommendationQuery(combat=["real_time"])
        )
        appids = [item.appid for item in selection.candidates]

        self.assertEqual(appids, [1, 3])
        self.assertEqual([item.appid for item in selection.verified_candidates], [1])
        self.assertEqual([item.appid for item in selection.unverified_candidates], [3])
        self.assertEqual([item.appid for item in selection.rejected], [2])

    def test_selection_dict_reports_the_condition_gate(self) -> None:
        profiles = [
            (Path("unknown.json"), profile(3, "Unknown Combat")),
            (Path("turn.json"), profile(2, "Turn Based", combat_facets=["turn_based"])),
        ]

        payload = (
            RecommendationProfileIndex(profiles)
            .search("실시간 전투 추천", RecommendationQuery(combat=["real_time"]))
            .to_dict()
        )

        self.assertEqual(payload["verified_matches"], 0)
        self.assertEqual(payload["unverified_matches"], 1)
        self.assertEqual(payload["constraint_gate"]["status"], "unverified_only")
        self.assertIn(2, payload["constraint_gate"]["rejected"])

    def test_founding_case_pretty_art_but_turn_based_combat_is_rejected(self) -> None:
        """기획안 2.1: 그림체는 마음에 들지만 전투가 턴제인 게임을 구매 전에 걸러낸다."""

        request = parse_recommendation_query(
            "그림체가 예쁘고 스토리가 있는 게임을 찾고 있어. 턴제보다는 직접 움직이고 공격하는 액션이 좋아."
        )
        pretty_but_turn_based = profile(
            2852190,
            "Turn-Based Story RPG",
            combat_facets=["turn_based"],
            playstyle_facets=["story_rich"],
            popular_user_tags=[
                {"name": "Story Rich", "normalized": "story_rich", "rank": 1},
                {"name": "Turn-Based Combat", "normalized": "turn_based_combat", "rank": 2},
            ],
        )

        report = evaluate_candidate_conditions(pretty_but_turn_based, request)
        selection = RecommendationProfileIndex(
            [(Path("mhs3.json"), pretty_but_turn_based)]
        ).search("실시간 액션", request)

        # 4.1: "액션"만으로 실시간 전투를 확정하지 않고, 사용자가 배제한 턴제는 조건에서 뺀다.
        self.assertEqual(request.combat, ["direct_control"])
        self.assertNotIn("turn_based", request.combat)
        self.assertIn("turn_based_combat", request.excluded_conditions)
        self.assertFalse(report.passes)
        self.assertEqual(
            sorted(verdict.label for verdict in report.must_violated),
            ["전투 방식: 직접 조작", "제외 조건: 턴제 전투"],
        )
        self.assertEqual(selection.candidates, [])
        self.assertEqual(selection.to_dict()["constraint_gate"]["status"], "no_match")

    def test_summary_reports_no_match_when_every_candidate_violates(self) -> None:
        query = RecommendationQuery(combat=["real_time"])
        reports = [
            evaluate_candidate_conditions(profile(1, "A", combat_facets=["turn_based"]), query),
            evaluate_candidate_conditions(profile(2, "B", combat_facets=["turn_based"]), query),
        ]

        self.assertEqual(summarize_constraint_gate(reports)["status"], "no_match")


if __name__ == "__main__":
    unittest.main()
