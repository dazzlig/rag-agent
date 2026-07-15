from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, TypeVar
from urllib.parse import urlparse


TELEMETRY_SCHEMA_VERSION = "telemetry-v1"
PRICING_VERSION = "2026-07-14-official-defaults"

# USD per one million tokens. These defaults are intentionally small and
# explicit so an evaluation report remains reproducible. Override the complete
# table with STEAM_RAG_PRICING_JSON when a model or price changes.
DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "text-embedding-3-small": {
        "input": 0.02,
        "cached_input": 0.02,
        "output": 0.00,
    },
}
DEFAULT_TAVILY_USD_PER_CREDIT = 0.008

T = TypeVar("T")


def _pricing_table() -> dict[str, dict[str, float]]:
    table = {model: dict(prices) for model, prices in DEFAULT_MODEL_PRICING.items()}
    raw = os.getenv("STEAM_RAG_PRICING_JSON", "").strip()
    if not raw:
        return table
    try:
        supplied = json.loads(raw)
    except json.JSONDecodeError:
        return table
    if not isinstance(supplied, dict):
        return table
    for model, prices in supplied.items():
        if not isinstance(prices, dict):
            continue
        try:
            input_rate = float(prices.get("input", 0.0))
            cached_input_rate = float(prices.get("cached_input", input_rate))
            output_rate = float(prices.get("output", 0.0))
        except (TypeError, ValueError):
            continue
        table[str(model)] = {
            "input": input_rate,
            "cached_input": cached_input_rate,
            "output": output_rate,
        }
    return table


def _rate_for_model(model: str) -> dict[str, float] | None:
    table = _pricing_table()
    if model in table:
        return table[model]
    for name, rate in table.items():
        if model.startswith(f"{name}-"):
            return rate
    return None


def _get_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _usage_tokens(response: Any) -> tuple[int, int, int, int]:
    usage = _get_value(response, "usage")
    if usage is None:
        return 0, 0, 0, 0
    input_tokens = int(
        _get_value(usage, "prompt_tokens", _get_value(usage, "input_tokens", 0)) or 0
    )
    output_tokens = int(
        _get_value(usage, "completion_tokens", _get_value(usage, "output_tokens", 0)) or 0
    )
    total_tokens = int(_get_value(usage, "total_tokens", input_tokens + output_tokens) or 0)
    prompt_details = _get_value(
        usage,
        "prompt_tokens_details",
        _get_value(usage, "input_tokens_details"),
    )
    cached_tokens = int(_get_value(prompt_details, "cached_tokens", 0) or 0)
    return input_tokens, output_tokens, total_tokens, cached_tokens


def _round_cost(value: float) -> float:
    return round(float(value), 8)


@dataclass(slots=True)
class TelemetryCollector:
    openai_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    tavily: dict[str, Any] = field(
        default_factory=lambda: {
            "request_count": 0,
            "external_call_count": 0,
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "credits": 0.0,
            "estimated_cost_usd": 0.0,
            "error_count": 0,
        }
    )
    steam: dict[str, Any] = field(
        default_factory=lambda: {
            "request_count": 0,
            "attempt_count": 0,
            "success_count": 0,
            "error_count": 0,
            "latency_ms": 0.0,
            "endpoints": defaultdict(
                lambda: {
                    "request_count": 0,
                    "attempt_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                }
            ),
        }
    )
    corpus: dict[str, Any] = field(
        default_factory=lambda: {
            "check_count": 0,
            "collected_count": 0,
            "reused_count": 0,
            "indexed_count": 0,
            "items": [],
        }
    )

    def record_openai(
        self,
        *,
        model: str,
        operation: str,
        response: Any = None,
        error: bool = False,
        latency_ms: float = 0.0,
    ) -> None:
        input_tokens, output_tokens, total_tokens, cached_tokens = _usage_tokens(response)
        rate = _rate_for_model(model)
        estimated_cost = 0.0
        if rate is not None:
            billable_cached = min(cached_tokens, input_tokens)
            uncached_input = max(0, input_tokens - billable_cached)
            estimated_cost = (
                uncached_input * rate["input"]
                + billable_cached * rate.get("cached_input", rate["input"])
                + output_tokens * rate["output"]
            ) / 1_000_000
        row = self.openai_models.setdefault(
            model,
            {
                "call_count": 0,
                "chat_call_count": 0,
                "embedding_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "estimated_cost_usd": 0.0,
                "unknown_cost_call_count": 0,
                "error_count": 0,
                "latency_ms": 0.0,
            },
        )
        row["call_count"] += 1
        if operation == "embedding":
            row["embedding_call_count"] += 1
        else:
            row["chat_call_count"] += 1
        row["input_tokens"] += input_tokens
        row["output_tokens"] += output_tokens
        row["total_tokens"] += total_tokens
        row["cached_input_tokens"] += cached_tokens
        row["estimated_cost_usd"] += estimated_cost
        row["unknown_cost_call_count"] += int(rate is None)
        row["error_count"] += int(error)
        row["latency_ms"] += max(0.0, latency_ms)

    def record_tavily(
        self,
        *,
        cache_hit: bool,
        credits: float = 0.0,
        error: bool = False,
    ) -> None:
        credits = max(0.0, float(credits))
        self.tavily["request_count"] += 1
        self.tavily["cache_hit_count"] += int(cache_hit)
        self.tavily["cache_miss_count"] += int(not cache_hit)
        self.tavily["external_call_count"] += int(not cache_hit)
        self.tavily["credits"] += credits
        self.tavily["error_count"] += int(error)
        usd_per_credit = float(
            os.getenv("TAVILY_USD_PER_CREDIT", str(DEFAULT_TAVILY_USD_PER_CREDIT))
        )
        self.tavily["estimated_cost_usd"] += credits * usd_per_credit

    def record_steam_request(self, endpoint: str) -> None:
        self.steam["request_count"] += 1
        self.steam["endpoints"][endpoint]["request_count"] += 1

    def record_steam_attempt(
        self,
        endpoint: str,
        *,
        success: bool,
        latency_ms: float,
    ) -> None:
        self.steam["attempt_count"] += 1
        self.steam["success_count"] += int(success)
        self.steam["error_count"] += int(not success)
        self.steam["latency_ms"] += max(0.0, latency_ms)
        endpoint_row = self.steam["endpoints"][endpoint]
        endpoint_row["attempt_count"] += 1
        endpoint_row["success_count"] += int(success)
        endpoint_row["error_count"] += int(not success)

    def record_corpus_update(
        self,
        *,
        appid: int,
        name: str,
        collected: bool,
        indexed: bool,
        reason: str,
    ) -> None:
        self.corpus["check_count"] += 1
        self.corpus["collected_count"] += int(collected)
        self.corpus["reused_count"] += int(not collected)
        self.corpus["indexed_count"] += int(indexed)
        if len(self.corpus["items"]) < 20:
            self.corpus["items"].append(
                {
                    "appid": int(appid),
                    "name": str(name),
                    "collected": bool(collected),
                    "indexed": bool(indexed),
                    "reason": str(reason),
                }
            )

    def snapshot(self) -> dict[str, Any]:
        models: dict[str, dict[str, Any]] = {}
        openai_totals: dict[str, Any] = {
            "call_count": 0,
            "chat_call_count": 0,
            "embedding_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "estimated_cost_usd": 0.0,
            "unknown_cost_call_count": 0,
            "error_count": 0,
            "latency_ms": 0.0,
        }
        for model, source in sorted(self.openai_models.items()):
            row = dict(source)
            row["estimated_cost_usd"] = _round_cost(row["estimated_cost_usd"])
            row["latency_ms"] = round(row["latency_ms"], 3)
            row["cost_known"] = row["unknown_cost_call_count"] == 0
            models[model] = row
            for key in openai_totals:
                openai_totals[key] += source[key]
        openai_totals["estimated_cost_usd"] = _round_cost(
            openai_totals["estimated_cost_usd"]
        )
        openai_totals["latency_ms"] = round(openai_totals["latency_ms"], 3)
        openai_totals["models"] = models

        tavily = dict(self.tavily)
        tavily["credits"] = round(tavily["credits"], 4)
        tavily["estimated_cost_usd"] = _round_cost(tavily["estimated_cost_usd"])
        tavily["cache_hit_rate"] = (
            round(tavily["cache_hit_count"] / tavily["request_count"], 6)
            if tavily["request_count"]
            else None
        )
        steam = {
            **{key: value for key, value in self.steam.items() if key != "endpoints"},
            "latency_ms": round(self.steam["latency_ms"], 3),
            "endpoints": {
                endpoint: dict(row)
                for endpoint, row in sorted(self.steam["endpoints"].items())
            },
        }
        corpus = {**self.corpus, "items": list(self.corpus["items"])}
        estimated_cost = (
            openai_totals["estimated_cost_usd"] + tavily["estimated_cost_usd"]
        )
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "pricing_version": PRICING_VERSION,
            "cost_is_estimate": True,
            "openai": openai_totals,
            "tavily": tavily,
            "steam": steam,
            "corpus": corpus,
            "external_call_count": (
                openai_totals["call_count"]
                + tavily["external_call_count"]
                + steam["attempt_count"]
            ),
            "estimated_cost_usd": _round_cost(estimated_cost),
        }


_CURRENT: ContextVar[TelemetryCollector | None] = ContextVar(
    "steam_rag_telemetry", default=None
)


@contextmanager
def telemetry_session() -> Iterator[TelemetryCollector]:
    existing = _CURRENT.get()
    if existing is not None:
        yield existing
        return
    collector = TelemetryCollector()
    token = _CURRENT.set(collector)
    try:
        yield collector
    finally:
        _CURRENT.reset(token)


def current_telemetry() -> TelemetryCollector | None:
    return _CURRENT.get()


def tracked_openai_call(
    *,
    model: str,
    operation: str,
    call: Callable[[], T],
) -> T:
    started = time.perf_counter()
    try:
        response = call()
    except Exception:
        collector = current_telemetry()
        if collector is not None:
            collector.record_openai(
                model=model,
                operation=operation,
                error=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        raise
    collector = current_telemetry()
    if collector is not None:
        collector.record_openai(
            model=model,
            operation=operation,
            response=response,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    return response


def steam_endpoint_name(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "root"
    if "/app/" in parsed.path:
        return "store_html"
    return path.replace("/", ".")
