from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401
from steam_rag.application.service_runtime import (
    CANDIDATE_FEEDBACK_PATTERN,
    ServicePaths,
    SteamServiceRuntime,
    _candidate_payload,
    _recommendation_markdown,
)
from steam_rag.common.models import Document, SearchResult
from steam_rag.game_recommendation.constraints import evaluate_candidate_conditions
from steam_rag.game_recommendation.query_parser import CandidateScore, RecommendationQuery
from steam_rag.tools.game_tools import ToolBudget


def profile(appid: int, name: str, **overrides) -> dict:
    payload = {
        "appid": appid,
        "name": name,
        "app_type": "game",
        "steam_genres_normalized": ["action"],
        "popular_user_tags": [{"name": "Action", "normalized": "action", "rank": 1}],
        "combat_facets": [],
        "perspective_facets": [],
        "dimension_facets": [],
        "playstyle_facets": [],
        "collected_at": "2026-09-01T00:00:00+00:00",
        "store_summary": "직접 조작하는 액션 게임입니다.",
        "price": {"currency": "KRW", "final": 3_000_000, "discount_percent": 0},
    }
    payload.update(overrides)
    return payload


def candidate(appid: int, name: str, query: RecommendationQuery, **overrides) -> CandidateScore:
    payload = profile(appid, name, **overrides)
    score = CandidateScore(appid=appid, name=name, score=1.0, profile_path="", profile=payload)
    score.constraints = evaluate_candidate_conditions(payload, query)
    return score


def evidence(appid: int, title: str) -> SearchResult:
    return SearchResult(
        Document(
            "실시간 전투로 직접 회피하고 공격한다.",
            {
                "appid": appid,
                "game_name": "Unknown Combat",
                "section": "gameplay",
                "item_title": title,
                "source_type": "steam_corpus",
            },
        ),
        score=1.0,
        rank=1,
    )


class CandidateCardTests(unittest.TestCase):
    def test_card_carries_fit_checks_and_information_status(self) -> None:
        query = RecommendationQuery(combat=["real_time"], genres=["action"])
        payload = _candidate_payload(candidate(1, "Unknown Combat", query))

        self.assertEqual(payload["condition_status"], "unverified")
        self.assertIn("공식 장르: 액션", payload["fit_reasons"])
        self.assertIn("전투 방식: 실시간 전투 미확인", payload["checks_before_choosing"])
        self.assertIn(
            {"source": "Steam 상품 정보", "checked_at": "2026-09-01T00:00:00+00:00"},
            payload["information_status"]["checked_sources"],
        )

    def test_answer_says_when_no_candidate_is_fully_verified(self) -> None:
        query = RecommendationQuery(combat=["real_time"])
        cards = [_candidate_payload(candidate(1, "Unknown Combat", query))]

        answer = _recommendation_markdown(
            "실시간 전투 게임 추천",
            cards,
            {},
            SimpleNamespace(sale_required=False, upcoming_required=False),
        )

        self.assertIn("모든 필수 조건을 확인한 게임은 아직 없습니다", answer)
        self.assertIn("선택 전 확인: 전투 방식: 실시간 전투 미확인", answer)

    def test_verified_candidate_card_has_no_check_list(self) -> None:
        query = RecommendationQuery(combat=["real_time"])
        verified = candidate(
            2,
            "Confirmed Action",
            query,
            combat_facets=["real_time"],
            facet_evidence=[
                {
                    "facet_type": "combat_facets",
                    "facet": "real_time",
                    "source_type": "steam_popular_user_tag",
                }
            ],
        )
        payload = _candidate_payload(verified)

        self.assertEqual(payload["condition_status"], "satisfied")
        self.assertEqual(payload["checks_before_choosing"], [])


class ConditionInvestigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        self.paths = ServicePaths(
            docs_dir=root / "docs",
            index_path=root / "index",
            raw_dir=root / "raw",
            catalog_path=root / "catalog.json",
            profiles_dir=root / "profiles",
            service_db=root / "service.db",
            time_analysis_dir=root / "time",
            workspace_db=root / "workspace.db",
            expert_dir=root / "experts",
        )
        self.runtime = SteamServiceRuntime(paths=self.paths, enable_reranker=False)
        self.query = RecommendationQuery(combat=["real_time"])

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_no_extra_search_when_every_condition_is_verified(self) -> None:
        verified = candidate(
            1,
            "Confirmed",
            self.query,
            combat_facets=["real_time"],
            facet_evidence=[
                {
                    "facet_type": "combat_facets",
                    "facet": "real_time",
                    "source_type": "steam_genre",
                }
            ],
        )
        budget = ToolBudget()

        report = self.runtime._investigate_unverified_conditions([verified], budget=budget)

        self.assertEqual(report["status"], "not_required")
        self.assertEqual(budget.expert_calls, 0)

    def test_price_only_gaps_do_not_open_the_index(self) -> None:
        """가격·할인은 문서 검색으로 확인할 수 없으므로 추가 조사를 하지 않는다."""

        self.paths.index_path.mkdir(parents=True)
        query = RecommendationQuery(sale_required=True)
        item = candidate(1, "No Price Data", query, price={})
        budget = ToolBudget()

        with patch(
            "steam_rag.application.service_runtime.RAGPipeline.from_path"
        ) as from_path:
            report = self.runtime._investigate_unverified_conditions([item], budget=budget)

        self.assertEqual(report["status"], "not_required")
        from_path.assert_not_called()

    def test_missing_index_is_reported_instead_of_claiming_verification(self) -> None:
        budget = ToolBudget()
        report = self.runtime._investigate_unverified_conditions(
            [candidate(1, "Unknown Combat", self.query)], budget=budget
        )

        self.assertEqual(report["status"], "no_corpus")
        self.assertEqual(report["resolved"], [])
        self.assertEqual(budget.expert_calls, 0)

    def test_investigation_stops_at_the_expert_call_limit(self) -> None:
        self.paths.index_path.mkdir(parents=True)
        candidates = [candidate(index, f"Unknown {index}", self.query) for index in range(1, 6)]
        pipeline = SimpleNamespace(
            index=SimpleNamespace(
                documents=[Document("d", {"appid": item.appid}) for item in candidates]
            ),
            search=lambda question, k=4: [evidence(1, "전투 설명")],
        )
        budget = ToolBudget()

        with (
            patch("steam_rag.application.service_runtime.OpenAIEmbedder"),
            patch(
                "steam_rag.application.service_runtime.RAGPipeline.from_path",
                return_value=pipeline,
            ),
        ):
            report = self.runtime._investigate_unverified_conditions(candidates, budget=budget)

        self.assertEqual(budget.expert_calls, 3)
        self.assertEqual(len(report["resolved"]), 3)
        self.assertEqual(report["status"], "completed")
        self.assertIn("expert_call:verify:4", budget.to_dict()["denied"])


class CandidateFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        self.runtime = SteamServiceRuntime(
            paths=ServicePaths(
                docs_dir=root / "docs",
                index_path=root / "index",
                raw_dir=root / "raw",
                catalog_path=root / "catalog.json",
                profiles_dir=root / "profiles",
                service_db=root / "service.db",
                time_analysis_dir=root / "time",
                workspace_db=root / "workspace.db",
                expert_dir=root / "experts",
            ),
            enable_reranker=False,
        )
        self.games = [{"appid": 1, "name": "Alpha"}, {"appid": 2, "name": "Beta"}]

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_pattern_matches_aspect_level_rejection(self) -> None:
        self.assertTrue(CANDIDATE_FEEDBACK_PATTERN.search("A의 그림체는 좋은데 전투는 B가 좋아"))
        self.assertTrue(CANDIDATE_FEEDBACK_PATTERN.search("반복 플레이가 많은 건 별로야"))
        self.assertIsNone(CANDIDATE_FEEDBACK_PATTERN.search("이 게임 가격 알려줘"))

    def test_rejection_reason_becomes_exclusions_and_preferences(self) -> None:
        parsed = {
            "rejected": [{"appid": 1, "aspect": "combat", "reason": "전투가 턴제라 싫음"}],
            "liked": [{"appid": 1, "aspect": "art_style"}],
            "new_exclude": ["turn_based"],
            "preferred_aspects": ["art_style"],
            "already_played": [2],
            "needs_new_candidates": True,
        }
        with patch(
            "steam_rag.application.service_runtime.OpenAIAnswerGenerator"
        ) as generator_type:
            generator_type.return_value.interpret_candidate_feedback.return_value = parsed
            feedback = self.runtime._candidate_feedback(
                "Alpha의 그림체는 좋은데 전투는 별로야",
                self.games,
                user_id="tester",
                session_id="disc_1",
            )

        stored = {(item.kind, item.value) for item in self.runtime.workspace.list_preferences("tester")}

        self.assertEqual(feedback["excluded_appids"], [1, 2])
        self.assertEqual(feedback["preferred_aspects"], ["art_style"])
        self.assertIn(("dislike", "turn_based"), stored)
        self.assertIn(("like", "art_style"), stored)

    def test_session_feedback_is_not_stored_as_permanent_taste(self) -> None:
        with patch(
            "steam_rag.application.service_runtime.OpenAIAnswerGenerator"
        ) as generator_type:
            generator_type.return_value.interpret_candidate_feedback.return_value = {
                "new_exclude": ["turn_based"],
                "rejected": [],
            }
            self.runtime._candidate_feedback(
                "전투는 별로야", self.games, user_id="tester", session_id="disc_1"
            )

        persistent = self.runtime.workspace.list_preferences("tester", scope="persistent")
        session = self.runtime.workspace.list_preferences(
            "tester", scope="session", session_id="disc_1"
        )

        self.assertEqual(persistent, [])
        self.assertEqual([item.value for item in session], ["turn_based"])

    def test_unrelated_message_skips_the_feedback_agent(self) -> None:
        with patch(
            "steam_rag.application.service_runtime.OpenAIAnswerGenerator"
        ) as generator_type:
            feedback = self.runtime._candidate_feedback(
                "이 게임 가격 알려줘", self.games, user_id="tester", session_id="disc_1"
            )

        self.assertEqual(feedback, {})
        generator_type.return_value.interpret_candidate_feedback.assert_not_called()

    def test_hallucinated_appids_are_dropped(self) -> None:
        with patch(
            "steam_rag.application.service_runtime.OpenAIAnswerGenerator"
        ) as generator_type:
            generator_type.return_value.interpret_candidate_feedback.return_value = {
                "rejected": [{"appid": 999, "aspect": "combat", "reason": "없는 게임"}],
                "already_played": [888],
            }
            feedback = self.runtime._candidate_feedback(
                "전투가 별로야", self.games, user_id="tester", session_id=""
            )

        self.assertEqual(feedback["excluded_appids"], [])


if __name__ == "__main__":
    unittest.main()
