from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401
from steam_rag.common.telemetry import (
    TelemetryCollector,
    telemetry_session,
    tracked_openai_call,
)


class TelemetryTests(unittest.TestCase):
    def test_openai_usage_and_official_default_costs_are_aggregated(self) -> None:
        chat_response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                total_tokens=2_000_000,
                prompt_tokens_details=SimpleNamespace(cached_tokens=100_000),
            )
        )
        embedding_response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1_000_000, total_tokens=1_000_000)
        )

        with telemetry_session() as telemetry:
            tracked_openai_call(
                model="gpt-5-mini",
                operation="chat",
                call=lambda: chat_response,
            )
            tracked_openai_call(
                model="text-embedding-3-small",
                operation="embedding",
                call=lambda: embedding_response,
            )
            snapshot = telemetry.snapshot()

        self.assertEqual(snapshot["openai"]["call_count"], 2)
        self.assertEqual(snapshot["openai"]["chat_call_count"], 1)
        self.assertEqual(snapshot["openai"]["embedding_call_count"], 1)
        self.assertEqual(snapshot["openai"]["cached_input_tokens"], 100_000)
        self.assertEqual(snapshot["openai"]["estimated_cost_usd"], 2.2475)
        self.assertEqual(snapshot["estimated_cost_usd"], 2.2475)

    def test_tavily_steam_and_corpus_metrics_distinguish_cache_and_reuse(self) -> None:
        collector = TelemetryCollector()
        with patch.dict(os.environ, {"TAVILY_USD_PER_CREDIT": "0.008"}):
            collector.record_tavily(cache_hit=False, credits=1)
            collector.record_tavily(cache_hit=True)
        collector.record_steam_request("api.appdetails")
        collector.record_steam_attempt("api.appdetails", success=False, latency_ms=5)
        collector.record_steam_attempt("api.appdetails", success=True, latency_ms=7)
        collector.record_corpus_update(
            appid=10,
            name="Example",
            collected=False,
            indexed=False,
            reason="fresh",
        )
        snapshot = collector.snapshot()

        self.assertEqual(snapshot["tavily"]["request_count"], 2)
        self.assertEqual(snapshot["tavily"]["external_call_count"], 1)
        self.assertEqual(snapshot["tavily"]["cache_hit_rate"], 0.5)
        self.assertEqual(snapshot["tavily"]["estimated_cost_usd"], 0.008)
        self.assertEqual(snapshot["steam"]["request_count"], 1)
        self.assertEqual(snapshot["steam"]["attempt_count"], 2)
        self.assertEqual(snapshot["steam"]["error_count"], 1)
        self.assertEqual(snapshot["corpus"]["reused_count"], 1)
        self.assertEqual(snapshot["external_call_count"], 3)

    def test_nested_sessions_share_one_request_collector(self) -> None:
        with telemetry_session() as outer:
            with telemetry_session() as inner:
                self.assertIs(inner, outer)
                inner.record_tavily(cache_hit=False, credits=1)
            self.assertEqual(outer.snapshot()["tavily"]["external_call_count"], 1)


if __name__ == "__main__":
    unittest.main()
