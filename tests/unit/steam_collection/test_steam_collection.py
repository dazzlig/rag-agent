from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import _bootstrap  # noqa: F401
from steam_rag.common.models import Document
from steam_rag.rag_search.vector_store import VectorIndex
from steam_rag.steam_collection.corpus_manager import OnDemandCorpusManager, SteamCatalog, _resolve_via_store_search
from steam_rag.steam_collection.markdown_documents import parse_markdown
from steam_rag.steam_collection.steam_client import (
    SteamAPIClient,
    SteamGame,
    classify_news,
    classify_news_type,
    collect_game,
    parse_popular_tags_html,
    parse_tag_candidates_html,
    save_catalog,
)


class FakeSteamTransport:
    def __call__(self, url: str, params: dict[str, object]) -> dict:
        if "IStoreService/GetAppList" in url:
            return {
                "response": {
                    "apps": [{"appid": 42, "name": "Example Game"}],
                    "last_appid": 42,
                    "have_more_results": False,
                }
            }
        if "appdetails" in url:
            appid = str(params["appids"])
            return {
                appid: {
                    "success": True,
                    "data": {
                        "type": "game",
                        "steam_appid": int(appid),
                        "name": "Example Game",
                        "header_image": "https://example.invalid/header.jpg",
                        "short_description": "A 3D third-person action game about exploration and hunting.",
                        "about_the_game": (
                            "Explore a large three dimensional world, hunt dangerous monsters, "
                            "and fight them through direct real-time action combat with friends."
                        ),
                        "release_date": {"date": "Jan 1, 2025", "coming_soon": False},
                        "developers": ["Example Studio"],
                        "publishers": ["Example Studio"],
                        "genres": [{"description": "Action"}],
                        "categories": [{"description": "Online Co-op"}],
                        "is_free": False,
                        "price_overview": {
                            "currency": "USD",
                            "initial": 2999,
                            "final": 1499,
                            "discount_percent": 50,
                            "initial_formatted": "$29.99",
                            "final_formatted": "$14.99",
                        },
                    },
                }
            }
        if "appreviews" in url:
            return {
                "cursor": params["cursor"],
                "reviews": [
                    {
                        "recommendationid": "1",
                        "timestamp_created": 1_750_000_000,
                        "timestamp_updated": 1_750_000_100,
                        "language": "english",
                        "voted_up": True,
                        "review": "The combat and exploration are excellent.",
                        "weighted_vote_score": "0.8",
                        "votes_up": 3,
                        "author": {"playtime_at_review": 120},
                    }
                ],
            }
        if "GetNewsForApp" in url:
            return {
                "appnews": {
                    "newsitems": [
                        {
                            "gid": "news-1",
                            "date": 1_750_000_200,
                            "title": "Patch 1 Now Live",
                            "contents": "This patch fixes combat bugs.",
                            "url": "https://example.invalid/news-1",
                            "feedname": "steam_community_announcements",
                            "feedlabel": "Community Announcements",
                        }
                    ]
                }
            }
        raise AssertionError(f"Unexpected URL: {url}")


class FakeSteamHtmlTransport:
    def __call__(self, url: str, params: dict[str, object]) -> str:
        if "store.steampowered.com/app/42" not in url:
            raise AssertionError(f"Unexpected URL: {url}")
        return """
        <div class="glance_tags popular_tags">
          <a class="app_tag" href="/tags/en/Action%20RPG/">Action RPG</a>
          <a class="app_tag" href="/tags/en/Third%20Person/">Third Person</a>
          <a class="app_tag" href="/tags/en/3D/">3D</a>
          <a class="app_tag" href="/tags/en/Hunting/">Hunting</a>
          <a class="app_tag" href="/tags/en/Online%20Co-op/">Online Co-op</a>
        </div>
        """


class FakeEmbedder:
    model_name = "fake-embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, float(len(text))]


class SteamCollectionTests(unittest.TestCase):
    def client(self) -> SteamAPIClient:
        return SteamAPIClient(
            transport=FakeSteamTransport(),
            text_transport=FakeSteamHtmlTransport(),
            request_delay_seconds=0,
        )

    def test_collect_game_creates_parseable_timeaware_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = collect_game(
                self.client(),
                SteamGame(42, "Example Game"),
                docs_dir=root / "docs",
                raw_dir=root / "raw",
                profiles_dir=root / "profiles",
                max_reviews=1,
                news_count=1,
            )
            documents = parse_markdown(result.markdown_path)
            markdown = result.markdown_path.read_text(encoding="utf-8")

            self.assertTrue(result.markdown_path.exists())
            self.assertIsNotNone(result.profile_path)
            self.assertTrue(result.profile_path.exists())
            self.assertEqual(result.review_count, 1)
            self.assertEqual(result.news_count, 1)
            self.assertIn("- combat_facets:", markdown)
            self.assertIn("- price_currency: USD", markdown)
            self.assertIn("- price_discount_percent: 50", markdown)
            self.assertIn("- news_type: patch_note", markdown)
            self.assertIn("- steam_tags: ['Action RPG', 'Third Person', '3D', 'Hunting', 'Online Co-op']", markdown)
            self.assertIn("- popular_tags_source: store.html.app_tag", markdown)
            self.assertIn("- popular_user_tags_source: store_html.app_tag", markdown)
            self.assertIn("- popular_user_tags_language: koreana", markdown)
            self.assertIn("- perspective_facets:", markdown)
            self.assertIn("- dimension_facets:", markdown)
            self.assertIn("- playstyle_facets:", markdown)
            self.assertEqual({doc.metadata["section"] for doc in documents}, {"metadata", "store_summary", "about", "review", "news"})
            self.assertTrue(all(str(doc.metadata["appid"]) == "42" for doc in documents))
            self.assertTrue(all("real_time" in doc.metadata["combat_facets"] for doc in documents))
            self.assertTrue(all("third_person" in doc.metadata["perspective_facets"] for doc in documents))
            self.assertTrue(all("3d" in doc.metadata["dimension_facets"] for doc in documents))
            metadata_doc = next(doc for doc in documents if doc.metadata["section"] == "metadata")
            self.assertEqual(metadata_doc.metadata["price_currency"], "USD")
            self.assertEqual(metadata_doc.metadata["price_discount_percent"], "50")
            self.assertEqual(metadata_doc.metadata["popular_tags_source"], "store.html.app_tag")
            self.assertIn("action_rpg", metadata_doc.metadata["steam_tags_normalized"])
            self.assertIn("hunting", metadata_doc.metadata["playstyle_facets"])
            news_doc = next(doc for doc in documents if doc.metadata["section"] == "news")
            self.assertEqual(news_doc.metadata["news_type"], "patch_note")

            self.assertEqual(news_doc.metadata["relevance_type"], "valid_update_or_patch")
            profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
            self.assertEqual(profile["language"], "koreana")
            self.assertEqual(profile["country"], "KR")
            self.assertEqual(profile["app_type"], "game")
            self.assertFalse(profile["release_coming_soon"])
            self.assertEqual(profile["header_image"], "https://example.invalid/header.jpg")
            self.assertEqual(profile["popular_user_tags"][0]["rank"], 1)
            self.assertEqual(profile["popular_user_tags"][0]["name"], "Action RPG")
            self.assertTrue(
                any(
                    item["facet"] == "third_person"
                    and item["source_type"] == "steam_popular_user_tag"
                    and item["source_rank"] == 2
                    for item in profile["facet_evidence"]
                )
            )

    def test_parse_tag_candidates_from_client_rendered_sale_sections(self) -> None:
        markup = (
            'data-section_39096_3_*="{&quot;appids&quot;:[101,202,101]}" '
            'data-section_39096_4_*="{&quot;appids&quot;:[303]}"'
        )

        rows = parse_tag_candidates_html(markup, max_apps=3)

        self.assertEqual([row["appid"] for row in rows], [101, 202, 303])
        self.assertEqual([row["tag_rank"] for row in rows], [1, 2, 3])

    def test_collection_api_defaults_request_korean_locale(self) -> None:
        json_calls: list[tuple[str, dict[str, object]]] = []
        text_calls: list[tuple[str, dict[str, object]]] = []

        def json_transport(url: str, params: dict[str, object]) -> dict:
            json_calls.append((url, dict(params)))
            if "appdetails" in url:
                return {"42": {"success": True, "data": {"steam_appid": 42, "name": "게임"}}}
            if "appreviews" in url:
                return {"reviews": [], "cursor": "*"}
            raise AssertionError(url)

        def text_transport(url: str, params: dict[str, object]) -> str:
            text_calls.append((url, dict(params)))
            return '<a class="app_tag">액션 RPG</a>'

        client = SteamAPIClient(
            transport=json_transport,
            text_transport=text_transport,
            request_delay_seconds=0,
        )
        client.fetch_app_details(42)
        client.fetch_popular_tags(42)
        client.fetch_reviews(42)

        self.assertEqual(json_calls[0][1]["l"], "koreana")
        self.assertEqual(json_calls[0][1]["cc"], "KR")
        self.assertEqual(text_calls[0][1]["l"], "koreana")
        self.assertEqual(text_calls[0][1]["cc"], "KR")
        self.assertEqual(json_calls[1][1]["language"], "koreana")

    def test_review_date_range_paginates_until_start_date(self) -> None:
        calls: list[str] = []

        def unix(day: str) -> int:
            return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())

        def transport(url: str, params: dict[str, object]) -> dict:
            cursor = str(params["cursor"])
            calls.append(cursor)
            if cursor == "*":
                return {
                    "cursor": "page-2",
                    "reviews": [
                        {"recommendationid": "future", "timestamp_created": unix("2026-07-02")},
                        {"recommendationid": "june-20", "timestamp_created": unix("2026-06-20")},
                    ],
                }
            return {
                "cursor": "page-3",
                "reviews": [
                    {"recommendationid": "june-05", "timestamp_created": unix("2026-06-05")},
                    {"recommendationid": "may-30", "timestamp_created": unix("2026-05-30")},
                ],
            }

        result = SteamAPIClient(transport=transport, request_delay_seconds=0).fetch_reviews_by_date_range(
            42,
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 30),
        )

        self.assertEqual(calls, ["*", "page-2"])
        self.assertEqual(
            [review["recommendationid"] for review in result.reviews],
            ["june-05", "june-20"],
        )
        self.assertTrue(result.reached_start)
        self.assertFalse(result.truncated)

    def test_popular_tags_are_parsed_from_store_html(self) -> None:
        html = """
        <a class="app_tag" href="/tags/en/Metroidvania/"> Metroidvania </a>
        <a class="btnv6_blue_hoverfade btn_small_tall app_tag_modal">+</a>
        <a class="app_tag" href="/tags/en/Souls-like/">Souls-like</a>
        <a class="app_tag" href="/tags/en/2D/">2D</a>
        """

        self.assertEqual(parse_popular_tags_html(html), ["Metroidvania", "Souls-like", "2D"])

    def test_tag_page_candidates_preserve_rank(self) -> None:
        html = """
        <a class="tab_item" data-ds-appid="101" href="/app/101/">
          <div class="tab_item_name">First Game</div>
        </a>
        <a class="tab_item" data-ds-appid="102" href="/app/102/">
          <div class="tab_item_name">Second Game</div>
        </a>
        """

        self.assertEqual(
            parse_tag_candidates_html(html),
            [
                {"appid": 101, "name": "First Game", "tag_rank": 1},
                {"appid": 102, "name": "Second Game", "tag_rank": 2},
            ],
        )

    def test_tag_id_combination_search_uses_filtered_results_html(self) -> None:
        requested_tags: list[str] = []

        def transport(url: str, params: dict[str, object]) -> dict:
            self.assertIn("search/results", url)
            requested_tags.append(str(params["tags"]))
            return {
                "results_html": """
                <a class="search_result_row" data-ds-appid="201">
                  <span class="tab_item_name">Matching RPG</span>
                </a>
                """
            }

        def text_transport(url: str, params: dict[str, object]) -> str:
            self.assertIn("tagdata/populartags/english", url)
            return json.dumps(
                [
                    {"tagid": 122, "name": "RPG"},
                    {"tagid": 4325, "name": "Turn-Based Combat"},
                    {"tagid": 3871, "name": "2D"},
                ]
            )

        client = SteamAPIClient(
            transport=transport,
            text_transport=text_transport,
            request_delay_seconds=0,
        )
        results = client.search_store_by_tags(["2D", "Turn-Based Combat", "RPG"])

        self.assertEqual(requested_tags, ["3871,4325,122"])
        self.assertEqual(results[0]["appid"], 201)
        self.assertEqual(results[0]["condition_hits"], 3)

    def test_news_classification_separates_patch_and_sale_noise(self) -> None:
        patch = {"title": "Patch Notes 2", "contents": "Bug fixes and balance changes are now live."}
        sale = {"title": "Summer Sale", "contents": "Discount and wishlist promotion."}
        event = {"title": "Community Event", "contents": "Join our livestream contest."}
        release = {"title": "Version 1.0 Coming September 25", "contents": "Release date announcement."}

        self.assertEqual(classify_news_type(patch), "patch_note")
        self.assertEqual(classify_news(patch), "valid_update_or_patch")
        self.assertEqual(classify_news_type(sale), "sale_promo")
        self.assertEqual(classify_news(sale), "store_or_sales_related")
        self.assertEqual(classify_news_type(event), "community_event")
        self.assertEqual(classify_news(event), "franchise_or_promotion_related")
        self.assertEqual(classify_news_type(release), "release_announcement")
        self.assertEqual(classify_news(release), "franchise_or_promotion_related")

    def test_catalog_resolves_name_and_store_url(self) -> None:
        catalog = SteamCatalog([{"appid": 42, "name": "Example Game"}])
        self.assertEqual(catalog.resolve("Example Game 전투는 어때?").appid, 42)
        self.assertEqual(catalog.resolve("https://store.steampowered.com/app/42/").appid, 42)

    def test_catalog_does_not_treat_ui_test_words_as_a_game_title(self) -> None:
        catalog = SteamCatalog([{"appid": 3404680, "name": "test"}])

        self.assertIsNone(catalog.resolve("UI 최근 질문 삭제 테스트"))
        self.assertEqual(catalog.resolve("test").appid, 3404680)

    def test_store_search_resolves_game_missing_from_local_catalog(self) -> None:
        class SearchClient:
            def search_store(self, term: str, *, count: int = 10) -> list[dict]:
                self.term = term
                return [
                    {"appid": 3527290, "name": "PEAK"},
                    {"appid": 999, "name": "Unrelated Game"},
                ]

        client = SearchClient()
        game = _resolve_via_store_search(
            client,  # type: ignore[arg-type]
            "PEAK는 친구들이랑 하기 좋은 협동 생존 게임이야?",
        )

        self.assertIsNotNone(game)
        self.assertEqual(game.appid, 3527290)
        self.assertEqual(game.name, "PEAK")
        self.assertEqual(client.term, "PEAK")

    def test_catalog_preserves_case_for_colliding_game_titles(self) -> None:
        catalog = SteamCatalog(
            [
                {"appid": 3506430, "name": "Peak"},
                {"appid": 3527290, "name": "PEAK"},
            ]
        )

        game = catalog.resolve("PEAK는 친구와 협동하기 좋아?")

        self.assertIsNotNone(game)
        self.assertEqual(game.appid, 3527290)

    def test_catalog_sync_uses_last_appid_pagination(self) -> None:
        requested_last_appids: list[int] = []

        def transport(url: str, params: dict[str, object]) -> dict:
            self.assertIn("IStoreService/GetAppList", url)
            last_appid = int(params["last_appid"])
            requested_last_appids.append(last_appid)
            if last_appid == 0:
                return {
                    "response": {
                        "apps": [{"appid": 42, "name": "Example Game"}],
                        "last_appid": 42,
                        "have_more_results": True,
                    }
                }
            return {
                "response": {
                    "apps": [{"appid": 43, "name": "Second Game"}],
                    "last_appid": 43,
                    "have_more_results": False,
                }
            }

        client = SteamAPIClient(transport=transport, request_delay_seconds=0)
        apps = client.fetch_catalog("not-a-real-key", page_size=1)
        self.assertEqual(requested_last_appids, [0, 42])
        self.assertEqual([app["appid"] for app in apps], [42, 43])

    def test_on_demand_manager_collects_and_upserts_only_missing_game(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.json"
            index_path = root / "index.json"
            save_catalog(catalog_path, [{"appid": 42, "name": "Example Game"}])
            VectorIndex(
                [Document("Existing game", {"appid": "7", "game_key": "existing", "section": "about", "chunk_id": "existing"})],
                [[1.0, 13.0]],
                "fake-embedding",
            ).save(index_path)
            manager = OnDemandCorpusManager(
                client=self.client(),
                catalog_path=catalog_path,
                docs_dir=root / "docs",
                raw_dir=root / "raw",
                index_path=index_path,
                max_age=timedelta(hours=24),
            )

            first = manager.ensure_question("Example Game gameplay", FakeEmbedder())
            second = manager.ensure_question("Example Game gameplay", FakeEmbedder())
            updated = VectorIndex.load(index_path)

            self.assertTrue(first.collected)
            self.assertTrue(first.indexed)
            self.assertFalse(second.collected)
            self.assertFalse(second.indexed)
            self.assertTrue(any(str(doc.metadata.get("appid")) == "7" for doc in updated.documents))
            self.assertTrue(any(str(doc.metadata.get("appid")) == "42" for doc in updated.documents))

    def test_explicit_appid_does_not_require_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = OnDemandCorpusManager(
                client=self.client(),
                catalog_path=root / "missing-catalog.json",
                docs_dir=root / "docs",
                raw_dir=root / "raw",
                index_path=root / "index.json",
            )
            update = manager.ensure_question("appid: 42 gameplay", FakeEmbedder())
            self.assertEqual(update.game.name, "Example Game")
            self.assertTrue(update.markdown_path.name.startswith("example_game_42"))

    def test_missing_catalog_is_synced_with_existing_steam_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "steam_catalog.json"
            manager = OnDemandCorpusManager(
                client=self.client(),
                catalog_path=catalog_path,
                docs_dir=root / "docs",
                raw_dir=root / "raw",
                index_path=root / "index.json",
            )

            with patch.dict("os.environ", {"STEAM_WEB_API_KEY": "existing-key"}):
                update = manager.ensure_question("Example Game gameplay", FakeEmbedder())

            self.assertTrue(catalog_path.exists())
            self.assertEqual(update.game.appid, 42)


if __name__ == "__main__":
    unittest.main()
