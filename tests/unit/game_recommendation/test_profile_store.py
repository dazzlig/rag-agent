from __future__ import annotations

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401
from steam_rag.game_recommendation.candidate_service import DynamicRecommendationService, discovery_terms
from steam_rag.game_recommendation.profile_store import SteamProfileStore
from steam_rag.game_recommendation.query_parser import RecommendationQuery


def core_profile(appid: int, name: str) -> dict:
    return {
        "appid": appid,
        "name": name,
        "profile_updated_at": "2026-07-13T00:00:00+00:00",
        "profile_expires_at": "2099-01-01T00:00:00+00:00",
        "profile_completeness": 0.9,
        "steam_genres_normalized": ["action", "rpg"],
        "steam_categories_normalized": [],
        "popular_user_tags": [
            {"name": "Action RPG", "normalized": "action_rpg", "rank": 1},
            {"name": "Third Person", "normalized": "third_person", "rank": 2},
        ],
        "combat_facets": ["real_time"],
        "perspective_facets": ["third_person"],
        "dimension_facets": ["3d"],
        "playstyle_facets": ["character_progression"],
        "inferred_facets": {},
        "recent_review_summary": {"positive_ratio": None},
        "price": {"currency": "KRW", "final": 30_000 * 100},
        "searchable_terms": ["action_rpg", "third_person"],
    }


class FakeDiscoveryClient:
    def search_store(self, term: str, **_: object) -> list[dict]:
        return [
            {"appid": 101, "name": "Candidate One"},
            {"appid": 102, "name": "Candidate Two"},
        ]


class FakeTagDiscoveryClient:
    def fetch_tag_candidates(self, tag: str, **_: object) -> list[dict]:
        rows = {
            "2D": [(201, "Shared", 2), (202, "Only 2D", 1)],
            "Turn-Based Combat": [(201, "Shared", 3), (203, "Only Turn", 1)],
            "Rpg": [(201, "Shared", 4), (204, "Only RPG", 1)],
        }.get(tag, [])
        return [
            {"appid": appid, "name": name, "tag_rank": rank, "discovery_term": tag}
            for appid, name, rank in rows
        ]

    def search_store(self, term: str, **_: object) -> list[dict]:
        return []


class ProfileStoreTests(unittest.TestCase):
    def test_registry_profile_cache_and_resumable_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SteamProfileStore(Path(directory) / "service.db")
            synced = store.sync_registry(
                [
                    {"appid": 1, "name": "One"},
                    {"appid": 2, "name": "Two"},
                ]
            )
            store.upsert_core_profile(core_profile(1, "One"))
            store.enqueue(2, priority=80)
            job = store.claim_next()

            self.assertEqual(synced, 2)
            self.assertEqual(job.appid, 2)
            store.mark_failed(
                job.job_id,
                "temporary",
                status="transient_failed",
                retry_after=timedelta(seconds=0),
            )
            retried = store.claim_next()
            self.assertEqual(retried.appid, 2)
            store.mark_completed(retried.job_id)
            summary = store.summary()

        self.assertEqual(summary["registry_count"], 2)
        self.assertEqual(summary["core_profile_count"], 1)
        self.assertEqual(summary["job_statuses"], {"completed": 1})

    def test_candidate_shortage_discovers_and_collects_core_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles_dir = root / "profiles"
            store = SteamProfileStore(root / "service.db")

            def fake_collect(_client, game, *, profiles_dir: Path, **_kwargs) -> Path:
                path = profiles_dir / f"game_{game.appid}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(core_profile(game.appid, game.name)),
                    encoding="utf-8",
                )
                return path

            service = DynamicRecommendationService(
                client=FakeDiscoveryClient(),
                store=store,
                profiles_dir=profiles_dir,
            )
            query = RecommendationQuery(
                genres=["action", "rpg"],
                perspective=["third_person"],
                dimension=["3d"],
            )
            with patch(
                "steam_rag.game_recommendation.candidate_service.collect_recommendation_profile",
                side_effect=fake_collect,
            ):
                run = service.recommend(
                    "3D 3인칭 액션 RPG",
                    query,
                    min_candidates=2,
                    max_new_profiles=2,
                    discovery_per_term=2,
                )

        self.assertEqual(run.initial_candidate_count, 0)
        self.assertEqual(run.discovered_app_count, 2)
        self.assertEqual(run.new_core_profiles, [101, 102])
        self.assertEqual(run.selection.hard_filter_matches, 2)
        self.assertEqual(run.store_summary["core_profile_count"], 2)

    def test_discovery_terms_include_combination_and_specific_facets(self) -> None:
        terms = discovery_terms(
            RecommendationQuery(
                required_tags=["action_rpg"],
                perspective=["third_person"],
                dimension=["3d"],
            )
        )

        self.assertEqual(terms[0], "Action RPG Third Person 3D")
        self.assertIn("Third Person", terms)

    def test_multi_tag_candidate_is_collected_before_single_tag_hits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SteamProfileStore(root / "service.db")

            def fake_collect(_client, game, *, profiles_dir: Path, **_kwargs) -> Path:
                path = profiles_dir / f"game_{game.appid}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(core_profile(game.appid, game.name)), encoding="utf-8")
                return path

            service = DynamicRecommendationService(
                client=FakeTagDiscoveryClient(),
                store=store,
                profiles_dir=root / "profiles",
            )
            with patch(
                "steam_rag.game_recommendation.candidate_service.collect_recommendation_profile",
                side_effect=fake_collect,
            ):
                run = service.recommend(
                    "2D 턴제 RPG",
                    RecommendationQuery(
                        genres=["rpg"], combat=["turn_based"], dimension=["2d"]
                    ),
                    min_candidates=1,
                    max_new_profiles=1,
                )

        self.assertEqual(run.new_core_profiles, [201])


if __name__ == "__main__":
    unittest.main()
