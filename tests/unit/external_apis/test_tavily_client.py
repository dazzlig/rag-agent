from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from steam_rag.external_apis.tavily_client import TavilySearchClient, compact_tavily_results


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class TavilySearchTests(unittest.TestCase):
    def test_basic_search_disables_costly_answer_and_raw_content_and_uses_cache(self) -> None:
        calls: list[dict] = []

        def opener(request, *, timeout: float):
            calls.append(
                {
                    "authorization": request.headers.get("Authorization"),
                    "payload": json.loads(request.data.decode("utf-8")),
                    "timeout": timeout,
                }
            )
            return FakeResponse(
                {
                    "results": [
                        {
                            "title": "Official game page",
                            "url": "https://example.com/game",
                            "content": "Relevant Steam game information.",
                            "score": 0.91,
                        }
                    ],
                    "usage": {"credits": 1},
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            client = TavilySearchClient(
                "tvly-test",
                cache_dir=Path(directory),
                opener=opener,
            )
            first = client.search("Steam RPG", max_results=5)
            second = client.search("Steam RPG", max_results=5)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["authorization"], "Bearer tvly-test")
        self.assertEqual(calls[0]["payload"]["search_depth"], "basic")
        self.assertFalse(calls[0]["payload"]["include_answer"])
        self.assertFalse(calls[0]["payload"]["include_raw_content"])
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])

    def test_compaction_filters_low_score_and_unattributed_results(self) -> None:
        rows = compact_tavily_results(
            {
                "results": [
                    {"title": "Strong", "url": "https://example.com/1", "content": "Useful", "score": 0.8},
                    {"title": "Weak", "url": "https://example.com/2", "content": "Noise", "score": 0.2},
                    {"title": "No URL", "url": "", "content": "Unknown", "score": 0.9},
                ]
            }
        )

        self.assertEqual([row["title"] for row in rows], ["Strong"])


if __name__ == "__main__":
    unittest.main()
