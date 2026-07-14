from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.request import Request, urlopen


class TavilySearchClient:
    """Small Tavily Search client with a local TTL cache and no SDK dependency."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: Path = Path("data/web_cache/tavily"),
        cache_ttl_seconds: int = 24 * 60 * 60,
        timeout_seconds: float = 20.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = str(api_key or os.getenv("TAVILY_API_KEY") or "").strip()
        self.cache_dir = cache_dir
        self.cache_ttl_seconds = max(0, int(cache_ttl_seconds))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._opener = opener or urlopen

    def search(
        self,
        query: str,
        *,
        max_results: int = 6,
        search_depth: str = "basic",
        topic: str = "general",
        include_domains: Sequence[str] = (),
        exclude_domains: Sequence[str] = (),
        time_range: str | None = None,
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            raise ValueError("Tavily 검색어가 비어 있습니다.")
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY가 설정되지 않았습니다.")
        if search_depth not in {"basic", "fast", "ultra-fast", "advanced"}:
            raise ValueError(f"지원하지 않는 Tavily search_depth: {search_depth}")
        if topic not in {"general", "news", "finance"}:
            raise ValueError(f"지원하지 않는 Tavily topic: {topic}")

        payload: dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max(1, min(int(max_results), 20)),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "auto_parameters": False,
        }
        if include_domains:
            payload["include_domains"] = list(dict.fromkeys(str(item) for item in include_domains))[:20]
        if exclude_domains:
            payload["exclude_domains"] = list(dict.fromkeys(str(item) for item in exclude_domains))[:20]
        if time_range:
            payload["time_range"] = time_range

        cache_path = self._cache_path(payload)
        cached = self._read_cache(cache_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        request = Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "SteamLens/0.1",
            },
            method="POST",
        )
        with self._opener(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError("Tavily 응답 형식이 올바르지 않습니다.")
        result["cache_hit"] = False
        self._write_cache(cache_path, result)
        return result

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.json"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        if self.cache_ttl_seconds <= 0 or not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.cache_ttl_seconds:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_cache(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            return


def compact_tavily_results(payload: dict[str, Any], *, limit: int = 6) -> list[dict[str, Any]]:
    """Keep only attributable, reasonably relevant snippets for the LLM context."""

    rows: list[dict[str, Any]] = []
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        content = " ".join(str(item.get("content") or "").split())
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if not url.startswith(("http://", "https://")) or not content or score < 0.35:
            continue
        rows.append(
            {
                "title": str(item.get("title") or url).strip()[:300],
                "url": url,
                "content": content[:1400],
                "score": round(score, 4),
                "published_date": str(item.get("published_date") or "").strip(),
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[: max(1, min(int(limit), 10))]
