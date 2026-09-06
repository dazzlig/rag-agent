from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from steam_rag.game_recommendation.profile_store import SteamProfileStore
from steam_rag.game_recommendation.query_parser import RecommendationQuery
from steam_rag.tools.game_tools import (
    ToolBudget,
    compare_candidates,
    get_game_facts,
    get_user_game_state,
)
from steam_rag.user_workspace.store import WorkspaceStore


def profile(appid: int, name: str, **overrides) -> dict:
    payload = {
        "appid": appid,
        "name": name,
        "app_type": "game",
        "steam_genres_normalized": ["action"],
        "popular_user_tags": [{"name": "Action", "normalized": "action", "rank": 1}],
        "combat_facets": ["real_time"],
        "perspective_facets": [],
        "dimension_facets": [],
        "playstyle_facets": [],
        "collected_at": "2026-09-01T00:00:00+00:00",
        "price": {"currency": "KRW", "final": 3_000_000, "discount_percent": 0},
    }
    payload.update(overrides)
    return payload


class ToolBudgetTests(unittest.TestCase):
    def test_limits_follow_the_planned_operating_values(self) -> None:
        budget = ToolBudget()

        self.assertTrue(budget.take_extra_search("a"))
        self.assertTrue(budget.take_extra_search("b"))
        self.assertFalse(budget.take_extra_search("c"))
        self.assertTrue(all(budget.take_expert_call(str(index)) for index in range(3)))
        self.assertFalse(budget.take_expert_call("fourth"))
        self.assertTrue(budget.exhausted)
        self.assertEqual(budget.to_dict()["extra_searches"], 2)
        self.assertEqual(budget.to_dict()["expert_calls"], 3)
        self.assertIn("extra_search:c", budget.to_dict()["denied"])


class GameToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        self.store = SteamProfileStore(root / "service.db")
        for payload in (profile(1, "Real Time"), profile(2, "Turn Based", combat_facets=["turn_based"])):
            path = root / f"{payload['appid']}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.store.sync_registry([{"appid": payload["appid"], "name": payload["name"], "type": "game"}])
            self.store.upsert_core_profile(payload, profile_path=path)
        self.workspace = WorkspaceStore(root / "workspace.db")

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_game_facts_include_sources_and_condition_verdicts(self) -> None:
        facts = get_game_facts(self.store, 2, query=RecommendationQuery(combat=["real_time"]))

        self.assertTrue(facts["found"])
        self.assertEqual(facts["name"], "Turn Based")
        self.assertEqual(facts["collected_at"], "2026-09-01T00:00:00+00:00")
        self.assertEqual(facts["conditions"]["status"], "violated")

    def test_game_facts_report_missing_profiles(self) -> None:
        self.assertEqual(get_game_facts(self.store, 999), {"appid": 999, "found": False})

    def test_compare_reports_games_it_could_not_load(self) -> None:
        table, missing = compare_candidates(self.store, [1, 2, 999])

        self.assertEqual([game["appid"] for game in table.games], [1, 2])
        self.assertEqual(missing, [999])

    def test_user_game_state_is_scoped_to_user_game_and_playthrough(self) -> None:
        self.workspace.update_game_state("user_a", 1, progress="초반")
        self.workspace.open_play_thread("user_a", appid=1, topic="boss")

        state = get_user_game_state(self.workspace, "user_a", 1)
        other = get_user_game_state(self.workspace, "user_b", 1)

        self.assertTrue(state["state_known"])
        self.assertEqual(state["progress"], "초반")
        self.assertEqual(len(state["threads"]), 1)
        self.assertFalse(other["state_known"])
        self.assertEqual(other["threads"], [])


if __name__ == "__main__":
    unittest.main()
