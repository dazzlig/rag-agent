from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from steam_rag.game_metadata.playstyle import build_playstyle_metadata


DEFAULT_LANGUAGE = "koreana"
DEFAULT_COUNTRY = "KR"


JsonTransport = Callable[[str, dict[str, object]], dict]
TextTransport = Callable[[str, dict[str, object]], str]


@dataclass(frozen=True, slots=True)
class SteamGame:
    appid: int
    name: str
    game_key: str = ""

    def resolved_key(self) -> str:
        return self.game_key or f"{slugify(self.name)}_{self.appid}"


@dataclass(frozen=True, slots=True)
class CollectionResult:
    game: SteamGame
    markdown_path: Path
    review_count: int
    news_count: int
    collected_at: str
    profile_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ReviewRangeResult:
    reviews: list[dict]
    pages_fetched: int
    reached_start: bool
    truncated: bool


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    return slug or "steam_game"


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|li|h[1-6])\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_popular_tags_html(html_text: str, *, max_tags: int = 20) -> list[str]:
    """Extract Store page "Popular user-defined tags" from Steam HTML."""

    tags: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"<a\b[^>]*class=[\"'][^\"']*\bapp_tag\b[^\"']*[\"'][^>]*>(.*?)</a>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        tag = clean_text(match.group(1))
        tag = re.sub(r"^\s*\+\s*", "", tag).strip()
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
        if len(tags) >= max_tags:
            break
    return tags


def parse_tag_candidates_html(html_text: str, *, max_apps: int = 50) -> list[dict]:
    """Extract ranked app candidates from a Steam Store tag page."""

    candidates: list[dict] = []
    seen: set[int] = set()
    for match in re.finditer(
        r"<a\b(?P<attrs>[^>]*\bdata-ds-appid=[\"'][^\"']+[\"'][^>]*)>(?P<body>.*?)</a>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        appid_match = re.search(
            r"data-ds-appid=[\"']\s*(\d+)",
            match.group("attrs"),
            flags=re.IGNORECASE,
        )
        if not appid_match:
            continue
        appid = int(appid_match.group(1))
        if appid in seen:
            continue
        name_match = re.search(
            r"class=[\"'][^\"']*\btab_item_name\b[^\"']*[\"'][^>]*>(.*?)</",
            match.group("body"),
            flags=re.IGNORECASE | re.DOTALL,
        )
        name = clean_text(name_match.group(1)) if name_match else ""
        if not name:
            continue
        seen.add(appid)
        candidates.append({"appid": appid, "name": name, "tag_rank": len(candidates) + 1})
        if len(candidates) >= max_apps:
            break
    if candidates:
        return candidates

    # The current Steam tag hub is client-rendered. Its ranked sale sections
    # expose ordered app ids in HTML data attributes instead of tab_item links.
    for match in re.finditer(
        r'data-section_[^=\s]+="(?P<payload>[^"]+)"',
        html_text,
        flags=re.IGNORECASE,
    ):
        payload_text = html.unescape(match.group("payload"))
        try:
            payload = json.loads(payload_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for raw_appid in payload.get("appids", []):
            try:
                appid = int(raw_appid)
            except (TypeError, ValueError):
                continue
            if appid <= 0 or appid in seen:
                continue
            seen.add(appid)
            candidates.append(
                {
                    "appid": appid,
                    "name": f"Steam App {appid}",
                    "tag_rank": len(candidates) + 1,
                }
            )
            if len(candidates) >= max_apps:
                return candidates
    return candidates


def unix_date(value: object) -> str | None:
    try:
        return datetime.fromtimestamp(int(value), timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


class SteamAPIClient:
    """Small Steam HTTP client with retry, pagination, and injectable transport."""

    def __init__(
        self,
        *,
        transport: JsonTransport | None = None,
        text_transport: TextTransport | None = None,
        timeout: float = 30.0,
        retries: int = 3,
        backoff_seconds: float = 0.5,
        request_delay_seconds: float = 0.2,
    ) -> None:
        self.transport = transport
        self.text_transport = text_transport
        self.timeout = timeout
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.request_delay_seconds = request_delay_seconds
        self._tag_ids_cache: dict[str, int] | None = None

    def _default_transport(self, url: str, params: dict[str, object]) -> dict:
        request = Request(
            f"{url}?{urlencode(params)}",
            headers={"User-Agent": "rag-agent/0.1 Steam corpus collector"},
        )
        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - fixed HTTPS endpoints
            return json.loads(response.read().decode("utf-8"))

    def _default_text_transport(self, url: str, params: dict[str, object]) -> str:
        request = Request(
            f"{url}?{urlencode(params)}",
            headers={"User-Agent": "rag-agent/0.1 Steam corpus collector"},
        )
        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - fixed HTTPS endpoints
            return response.read().decode("utf-8", errors="replace")

    def _get(self, url: str, params: dict[str, object]) -> dict:
        transport = self.transport or self._default_transport
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                payload = transport(url, params)
                if not isinstance(payload, dict):
                    raise ValueError(f"Steam returned a non-object response for {url}")
                if self.request_delay_seconds:
                    time.sleep(self.request_delay_seconds)
                return payload
            except Exception as exc:  # retry network and transient decode failures
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        raise RuntimeError(f"Steam request failed after {self.retries} attempts: {url}") from last_error

    def _get_text(self, url: str, params: dict[str, object]) -> str:
        transport = self.text_transport or self._default_text_transport
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                payload = transport(url, params)
                if not isinstance(payload, str):
                    raise ValueError(f"Steam returned a non-text response for {url}")
                if self.request_delay_seconds:
                    time.sleep(self.request_delay_seconds)
                return payload
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        raise RuntimeError(f"Steam text request failed after {self.retries} attempts: {url}") from last_error

    def fetch_app_details(
        self,
        appid: int,
        *,
        language: str = DEFAULT_LANGUAGE,
        country: str = DEFAULT_COUNTRY,
    ) -> dict:
        payload = self._get(
            "https://store.steampowered.com/api/appdetails",
            {"appids": appid, "l": language, "cc": country},
        )
        app = payload.get(str(appid), {})
        if not app.get("success") or not isinstance(app.get("data"), dict):
            raise LookupError(f"Steam appdetails did not return app {appid}")
        return dict(app["data"])

    def fetch_popular_tags(
        self,
        appid: int,
        *,
        language: str = DEFAULT_LANGUAGE,
        country: str = DEFAULT_COUNTRY,
        max_tags: int = 20,
    ) -> list[str]:
        html_text = self._get_text(
            f"https://store.steampowered.com/app/{appid}/",
            {"l": language, "cc": country},
        )
        return parse_popular_tags_html(html_text, max_tags=max_tags)

    def fetch_reviews(
        self,
        appid: int,
        *,
        max_reviews: int = 50,
        language: str = DEFAULT_LANGUAGE,
        review_filter: str = "recent",
        purchase_type: str = "all",
    ) -> list[dict]:
        if review_filter not in {"recent", "updated", "all"}:
            raise ValueError("review_filter must be recent, updated, or all")
        reviews: list[dict] = []
        cursor = "*"
        while len(reviews) < max_reviews:
            payload = self._get(
                f"https://store.steampowered.com/appreviews/{appid}",
                {
                    "json": 1,
                    "filter": review_filter,
                    "language": language,
                    "purchase_type": purchase_type,
                    "num_per_page": min(100, max_reviews - len(reviews)),
                    "cursor": cursor,
                },
            )
            batch = payload.get("reviews") or []
            if not isinstance(batch, list) or not batch:
                break
            reviews.extend(item for item in batch if isinstance(item, dict))
            next_cursor = payload.get("cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = str(next_cursor)
        return reviews[:max_reviews]

    def fetch_reviews_by_date_range(
        self,
        appid: int,
        *,
        start_date: date | datetime,
        end_date: date | datetime,
        max_reviews: int = 5_000,
        max_pages: int = 100,
        language: str = DEFAULT_LANGUAGE,
        purchase_type: str = "all",
    ) -> ReviewRangeResult:
        """Page recent reviews backwards until the requested start date is reached."""

        def timestamp(value: date | datetime, *, end: bool) -> int:
            if isinstance(value, datetime):
                current = value
                if current.tzinfo is None:
                    current = current.replace(tzinfo=timezone.utc)
                return int(current.timestamp())
            boundary = datetime_time.max if end else datetime_time.min
            return int(datetime.combine(value, boundary, tzinfo=timezone.utc).timestamp())

        start_timestamp = timestamp(start_date, end=False)
        end_timestamp = timestamp(end_date, end=True)
        if start_timestamp > end_timestamp:
            raise ValueError("start_date must be on or before end_date")
        if max_reviews <= 0 or max_pages <= 0:
            raise ValueError("max_reviews and max_pages must be positive")

        cursor = "*"
        selected: list[dict] = []
        seen: set[str] = set()
        pages_fetched = 0
        reached_start = False
        while pages_fetched < max_pages and len(selected) < max_reviews:
            payload = self._get(
                f"https://store.steampowered.com/appreviews/{appid}",
                {
                    "json": 1,
                    "filter": "recent",
                    "language": language,
                    "purchase_type": purchase_type,
                    "review_type": "all",
                    "num_per_page": 100,
                    "cursor": cursor,
                },
            )
            pages_fetched += 1
            batch = payload.get("reviews") or []
            if not isinstance(batch, list) or not batch:
                break
            valid_timestamps: list[int] = []
            for item in batch:
                if not isinstance(item, dict):
                    continue
                try:
                    created = int(item.get("timestamp_created"))
                except (TypeError, ValueError):
                    continue
                valid_timestamps.append(created)
                if created < start_timestamp:
                    continue
                if created > end_timestamp:
                    continue
                key = str(item.get("recommendationid") or f"{created}:{item.get('review', '')}")
                if key in seen:
                    continue
                seen.add(key)
                selected.append(dict(item))
                if len(selected) >= max_reviews:
                    break
            if valid_timestamps and min(valid_timestamps) < start_timestamp:
                reached_start = True
                break
            next_cursor = payload.get("cursor")
            if not next_cursor or str(next_cursor) == cursor:
                break
            cursor = str(next_cursor)
        truncated = len(selected) >= max_reviews or (pages_fetched >= max_pages and not reached_start)
        selected.sort(key=lambda item: int(item.get("timestamp_created") or 0))
        return ReviewRangeResult(selected[:max_reviews], pages_fetched, reached_start, truncated)

    def fetch_news(
        self,
        appid: int,
        *,
        count: int = 20,
        maxlength: int = 0,
        feeds: str | None = "steam_community_announcements",
    ) -> list[dict]:
        params: dict[str, object] = {
            "appid": appid,
            "count": count,
            "maxlength": maxlength,
            "format": "json",
        }
        if feeds:
            params["feeds"] = feeds
        payload = self._get(
            "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/",
            params,
        )
        items = payload.get("appnews", {}).get("newsitems", [])
        return [dict(item) for item in items if isinstance(item, dict)]

    def search_store(
        self,
        term: str,
        *,
        count: int = 50,
        language: str = DEFAULT_LANGUAGE,
        country: str = DEFAULT_COUNTRY,
    ) -> list[dict]:
        """Discover a bounded set of Store candidates without profiling the full catalog."""

        if not term.strip():
            return []
        payload = self._get(
            "https://store.steampowered.com/api/storesearch/",
            {
                "term": term.strip(),
                "l": language,
                "cc": country,
                "count": max(1, min(int(count), 100)),
            },
        )
        items = payload.get("items") or []
        candidates: list[dict] = []
        seen: set[int] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                appid = int(item.get("id") or item.get("appid"))
            except (TypeError, ValueError):
                continue
            name = clean_text(item.get("name"))
            if appid <= 0 or not name or appid in seen:
                continue
            seen.add(appid)
            candidates.append(
                {
                    "appid": appid,
                    "name": name,
                    "source": "store.api.storesearch",
                    "discovery_term": term.strip(),
                }
            )
        return candidates

    def fetch_tag_candidates(
        self,
        tag: str,
        *,
        max_apps: int = 50,
        language: str = DEFAULT_LANGUAGE,
        country: str = DEFAULT_COUNTRY,
    ) -> list[dict]:
        """Discover ranked candidates from one bounded Steam Store tag page."""

        if not tag.strip():
            return []
        html_text = self._get_text(
            f"https://store.steampowered.com/tags/en/{quote(tag.strip(), safe='')}/",
            {"l": language, "cc": country},
        )
        candidates = parse_tag_candidates_html(html_text, max_apps=max_apps)
        for candidate in candidates:
            candidate["source"] = "store.html.tag_page"
            candidate["discovery_term"] = tag.strip()
        return candidates

    def fetch_tag_ids(self) -> dict[str, int]:
        """Load Steam Store's public popular-tag dictionary for tag-filtered discovery."""

        if self._tag_ids_cache is not None:
            return dict(self._tag_ids_cache)
        text = self._get_text(
            "https://store.steampowered.com/tagdata/populartags/english",
            {},
        )
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else payload.get("tags", [])
        tag_ids: dict[str, int] = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                tagid = int(item.get("tagid") or item.get("id"))
            except (TypeError, ValueError):
                continue
            name = str(item.get("name") or "").strip()
            if name and tagid > 0:
                tag_ids[name.casefold()] = tagid
        self._tag_ids_cache = tag_ids
        return dict(tag_ids)

    def search_store_by_tags(
        self,
        tags: Sequence[str],
        *,
        count: int = 50,
        language: str = DEFAULT_LANGUAGE,
        country: str = DEFAULT_COUNTRY,
    ) -> list[dict]:
        """Discover candidates from Steam search constrained by a tag-ID combination."""

        tag_names = [str(tag).strip() for tag in tags if str(tag).strip()]
        if not tag_names:
            return []
        dictionary = self.fetch_tag_ids()
        tag_ids = [dictionary.get(name.casefold()) for name in tag_names]
        resolved_ids = [tagid for tagid in tag_ids if tagid]
        # Partial resolution silently broadens an AND-style discovery query and
        # can fill the profile cache with games unrelated to the original need.
        if len(resolved_ids) != len(tag_names):
            return []
        payload = self._get(
            "https://store.steampowered.com/search/results/",
            {
                "query": "",
                "start": 0,
                "count": max(1, min(int(count), 100)),
                "dynamic_data": "",
                "sort_by": "_ASC",
                "tags": ",".join(str(tagid) for tagid in resolved_ids),
                "infinite": 1,
                "l": language,
                "cc": country,
            },
        )
        html_text = str(payload.get("results_html") or "")
        candidates = parse_tag_candidates_html(html_text, max_apps=count)
        for candidate in candidates:
            candidate["source"] = "store.search.tag_ids"
            candidate["discovery_term"] = " + ".join(tag_names)
            candidate["condition_hits"] = len(resolved_ids)
            candidate["matched_discovery_tags"] = tag_names
        return candidates

    def fetch_catalog(self, api_key: str, *, page_size: int = 50_000) -> list[dict]:
        if not api_key:
            raise ValueError("A Steam Web API key is required to synchronize the app catalog")
        apps: list[dict] = []
        last_appid = 0
        while True:
            payload = self._get(
                "https://api.steampowered.com/IStoreService/GetAppList/v1/",
                {
                    "key": api_key,
                    "include_games": True,
                    "include_dlc": False,
                    "include_software": False,
                    "include_videos": False,
                    "include_hardware": False,
                    "last_appid": last_appid,
                    "max_results": min(page_size, 50_000),
                    "format": "json",
                },
            )
            response = payload.get("response", {})
            batch = response.get("apps") or []
            apps.extend(item for item in batch if isinstance(item, dict) and item.get("name"))
            next_appid = int(response.get("last_appid") or (batch[-1].get("appid", 0) if batch else 0))
            have_more = bool(response.get("have_more_results"))
            if not batch or not have_more or next_appid <= last_appid:
                break
            last_appid = next_appid
        return apps


def normalize_store_info(app: dict) -> dict:
    about = clean_text(app.get("about_the_game") or app.get("detailed_description"))
    summary = clean_text(app.get("short_description"))
    genres = list(
        dict.fromkeys(
            item.get("description")
            for item in app.get("genres") or []
            if item.get("description")
        )
    )
    categories = list(
        dict.fromkeys(
            item.get("description")
            for item in app.get("categories") or []
            if item.get("description")
        )
    )
    price = normalize_price_overview(app)
    if len(about) < 100:
        about = "\n\n".join(
            part
            for part in (
                summary,
                f"Genres: {', '.join(genres)}" if genres else "",
                f"Store categories: {', '.join(categories)}" if categories else "",
            )
            if part
        )
    return {
        "app_type": app.get("type"),
        "name": app.get("name"),
        "steam_appid": app.get("steam_appid"),
        "release_date": (app.get("release_date") or {}).get("date"),
        "release_coming_soon": bool((app.get("release_date") or {}).get("coming_soon")),
        "header_image": app.get("header_image"),
        "capsule_image": app.get("capsule_image"),
        "developers": app.get("developers") or [],
        "publishers": app.get("publishers") or [],
        "genres": genres,
        "categories": categories,
        "steam_tags": app.get("steam_tags") or [],
        "popular_user_tags": app.get("popular_user_tags") or [],
        "popular_user_tags_source": app.get("popular_user_tags_source"),
        "popular_user_tags_collected_at": app.get("popular_user_tags_collected_at"),
        "popular_user_tags_language": app.get("popular_user_tags_language"),
        "popular_user_tags_error": app.get("popular_user_tags_error"),
        "popular_tags_source": app.get("popular_tags_source"),
        "popular_tags_language": app.get("popular_tags_language"),
        "popular_tags_error": app.get("popular_tags_error"),
        **price,
        "store_summary": summary,
        "about_text": about,
        "store_summary_source": "store.appdetails.short_description",
        "about_source": "store.appdetails.about_the_game_or_detailed_description",
    }


def normalize_price_overview(app: dict) -> dict:
    """Normalize Steam price fields without treating them as live forever."""

    overview = app.get("price_overview")
    is_free = bool(app.get("is_free"))
    if is_free:
        return {
            "is_free": True,
            "price_available": True,
            "price_currency": None,
            "price_initial": 0,
            "price_final": 0,
            "price_discount_percent": 100,
            "price_initial_formatted": "Free",
            "price_final_formatted": "Free",
            "price_source": "store.appdetails.is_free",
        }
    if not isinstance(overview, dict):
        return {
            "is_free": False,
            "price_available": False,
            "price_currency": None,
            "price_initial": None,
            "price_final": None,
            "price_discount_percent": None,
            "price_initial_formatted": None,
            "price_final_formatted": None,
            "price_source": "store.appdetails.price_overview",
        }
    return {
        "is_free": False,
        "price_available": True,
        "price_currency": overview.get("currency"),
        "price_initial": overview.get("initial"),
        "price_final": overview.get("final"),
        "price_discount_percent": overview.get("discount_percent"),
        "price_initial_formatted": overview.get("initial_formatted"),
        "price_final_formatted": overview.get("final_formatted"),
        "price_source": "store.appdetails.price_overview",
    }


def normalize_review(review: dict) -> dict:
    author = review.get("author") or {}
    return {
        "recommendationid": review.get("recommendationid"),
        "timestamp_created": review.get("timestamp_created"),
        "timestamp_updated": review.get("timestamp_updated"),
        "review_created_at": unix_date(review.get("timestamp_created")),
        "review_updated_at": unix_date(review.get("timestamp_updated")),
        "language": review.get("language"),
        "voted_up": review.get("voted_up"),
        "sentiment": "positive" if review.get("voted_up") is True else "negative",
        "review_text": clean_text(review.get("review")),
        "weighted_vote_score": review.get("weighted_vote_score"),
        "votes_up": review.get("votes_up"),
        "playtime_at_review": author.get("playtime_at_review"),
    }


PATCH_NEWS_TYPES = {"hotfix", "patch_note", "major_update", "content_update"}
LOW_VALUE_NEWS_TYPES = {"sale_promo", "community_event", "franchise_promo", "release_announcement", "unrelated"}


def classify_news_type(news: dict) -> str:
    title = str(news.get("title") or "").casefold()
    contents = str(news.get("contents") or "").casefold()
    feed = str(news.get("feedname") or "").casefold()
    text = f"{title} {contents} {feed}"
    if any(word in text for word in ("sale", "discount", "top sellers", "free weekend", "wishlist", "deal", "promotion", "할인", "세일", "무료 주말")):
        return "sale_promo"
    if any(word in text for word in ("coming soon", "coming september", "release date", "will launch", "launches on", "출시 예정", "출시일", "곧 출시")):
        return "release_announcement"
    if any(word in text for word in ("hotfix", "fix #", "quick fix", "핫픽스", "긴급 수정")):
        return "hotfix"
    if any(word in text for word in ("patch notes", "patch now live", "patch is live", "bug fix", "bug fixes", "fixes", "balance changes", "패치 노트", "버그 수정", "밸런스 조정")):
        return "patch_note"
    if "patch" in text and any(word in text for word in ("live", "fix", "bug", "balance")):
        return "patch_note"
    if any(word in text for word in ("major update", "big update", "new update", "update now live", "대규모 업데이트", "주요 업데이트")):
        return "major_update"
    if any(word in text for word in ("dlc", "expansion", "new content", "content update", "season", "확장팩", "신규 콘텐츠", "콘텐츠 업데이트", "시즌")):
        return "content_update"
    if any(word in text for word in ("community update", "community event", "contest", "livestream", "stream", "twitch")):
        return "community_event"
    if any(word in text for word in ("soundtrack", "artbook", "merch", "franchise", "pre-order", "preorder")):
        return "franchise_promo"
    return "unrelated"


def classify_news(news: dict) -> str:
    news_type = classify_news_type(news)
    if news_type in PATCH_NEWS_TYPES:
        return "valid_update_or_patch"
    if news_type == "sale_promo":
        return "store_or_sales_related"
    if news_type in {"community_event", "franchise_promo", "release_announcement"}:
        return "franchise_or_promotion_related"
    return "manual_review_needed"


def normalize_news(news: dict) -> dict:
    normalized = {
        "gid": news.get("gid"),
        "news_date": unix_date(news.get("date")),
        "title": clean_text(news.get("title")),
        "url": news.get("url"),
        "feedlabel": news.get("feedlabel"),
        "feedname": news.get("feedname"),
        "contents": clean_text(news.get("contents")),
    }
    normalized["news_type"] = classify_news_type(normalized)
    normalized["relevance_type"] = classify_news(normalized)
    return normalized


def build_markdown(
    game: SteamGame,
    store: dict,
    reviews: Sequence[dict],
    news: Sequence[dict],
    *,
    collected_at: str,
) -> str:
    name = str(store.get("name") or game.name)
    store_text = "\n\n".join(
        value
        for value in (
            str(store.get("store_summary") or "").strip(),
            str(store.get("about_text") or "").strip(),
        )
        if value
    )
    review_text = "\n\n".join(
        str(review.get("review_text") or "").strip()
        for review in reviews
        if str(review.get("review_text") or "").strip()
    )
    playstyle_text = "\n\n".join(value for value in (store_text, review_text) if value)
    playstyle = build_playstyle_metadata(
        playstyle_text,
        genres=store.get("genres"),
        categories=store.get("categories"),
        steam_tags=store.get("steam_tags"),
        popular_user_tags=store.get("popular_user_tags"),
        includes_reviews=bool(reviews),
        store_text=store_text,
        review_text=review_text,
    )
    lines = [
        f"# {name}",
        "",
        "## Metadata",
        "",
        f"- collection_schema_version: steam-md-v1",
        f"- collected_at: {collected_at}",
        f"- game_key: {game.resolved_key()}",
        f"- appid: {game.appid}",
        f"- name: {name}",
        f"- release_date: {store.get('release_date')}",
        f"- developers: {store.get('developers') or []}",
        f"- publishers: {store.get('publishers') or []}",
        f"- genres: {store.get('genres') or []}",
        f"- categories: {store.get('categories') or []}",
        f"- steam_tags: {store.get('steam_tags') or []}",
        f"- popular_user_tags: {store.get('popular_user_tags') or []}",
        f"- popular_user_tag_names: {[tag.get('name') for tag in store.get('popular_user_tags') or []]}",
        f"- popular_user_tag_ranks: {[tag.get('rank') for tag in store.get('popular_user_tags') or []]}",
        f"- popular_user_tags_source: {store.get('popular_user_tags_source')}",
        f"- popular_user_tags_language: {store.get('popular_user_tags_language')}",
        f"- popular_user_tags_collected_at: {store.get('popular_user_tags_collected_at')}",
        f"- popular_user_tags_error: {store.get('popular_user_tags_error')}",
        f"- popular_tags_source: {store.get('popular_tags_source')}",
        f"- popular_tags_language: {store.get('popular_tags_language')}",
        f"- popular_tags_collected_at: {collected_at if store.get('steam_tags') else None}",
        f"- popular_tags_error: {store.get('popular_tags_error')}",
        f"- is_free: {store.get('is_free')}",
        f"- price_available: {store.get('price_available')}",
        f"- price_currency: {store.get('price_currency')}",
        f"- price_initial: {store.get('price_initial')}",
        f"- price_final: {store.get('price_final')}",
        f"- price_discount_percent: {store.get('price_discount_percent')}",
        f"- price_initial_formatted: {store.get('price_initial_formatted')}",
        f"- price_final_formatted: {store.get('price_final_formatted')}",
        f"- price_source: {store.get('price_source')}",
        f"- price_collected_at: {collected_at}",
        f"- playstyle_profile_version: {playstyle['playstyle_profile_version']}",
        f"- playstyle_profile_source: {playstyle['playstyle_profile_source']}",
        f"- playstyle_profile_confidence: {playstyle['playstyle_profile_confidence']}",
        f"- playstyle_evidence_sources: {playstyle['playstyle_evidence_sources']}",
        f"- facet_evidence: {playstyle['facet_evidence']}",
        f"- steam_tags_normalized: {playstyle['steam_tags_normalized']}",
        f"- steam_genres_normalized: {playstyle['steam_genres_normalized']}",
        f"- steam_categories_normalized: {playstyle['steam_categories_normalized']}",
        f"- playstyle_terms_normalized: {playstyle['playstyle_terms_normalized']}",
        f"- combat_facets: {playstyle['combat_facets']}",
        f"- perspective_facets: {playstyle['perspective_facets']}",
        f"- dimension_facets: {playstyle['dimension_facets']}",
        f"- playstyle_facets: {playstyle['playstyle_facets']}",
        f"- store_summary_source: {store.get('store_summary_source')}",
        f"- about_source: {store.get('about_source')}",
        "- data_sources: ['store.appdetails', 'store.html.popular_tags', 'store.appreviews', 'ISteamNews.GetNewsForApp.v2']",
        "",
        "## Store Summary",
        "",
        str(store.get("store_summary") or "No store summary available."),
        "",
        "## About The Game",
        "",
        str(store.get("about_text") or "No about text available."),
        "",
        "## Recent Steam Reviews",
        "",
    ]
    if not reviews:
        lines.extend(["No review data available.", ""])
    for index, review in enumerate(reviews, start=1):
        lines.extend(
            [
                f"### Review {index}",
                f"- review_created_at: {review.get('review_created_at')}",
                f"- review_updated_at: {review.get('review_updated_at')}",
                f"- sentiment: {review.get('sentiment')}",
                f"- voted_up: {review.get('voted_up')}",
                f"- weighted_vote_score: {review.get('weighted_vote_score')}",
                f"- votes_up: {review.get('votes_up')}",
                f"- playtime_at_review: {review.get('playtime_at_review')}",
                "",
                str(review.get("review_text") or ""),
                "",
            ]
        )
    lines.extend(["## Steam News and Updates", ""])
    if not news:
        lines.extend(["No news data available.", ""])
    for index, item in enumerate(news, start=1):
        title = item.get("title") or f"News {index}"
        lines.extend(
            [
                f"### News {index}: {title}",
                f"- news_date: {item.get('news_date')}",
                f"- feedname: {item.get('feedname')}",
                f"- feedlabel: {item.get('feedlabel')}",
                f"- news_type: {item.get('news_type')}",
                f"- relevance_type: {item.get('relevance_type')}",
                f"- url: {item.get('url')}",
                "",
                str(item.get("contents") or ""),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def collect_game(
    client: SteamAPIClient,
    game: SteamGame,
    *,
    docs_dir: Path,
    raw_dir: Path,
    profiles_dir: Path | None = None,
    max_reviews: int = 50,
    news_count: int = 20,
    language: str = DEFAULT_LANGUAGE,
    country: str = DEFAULT_COUNTRY,
) -> CollectionResult:
    from steam_rag.game_recommendation.profile_builder import (
        build_recommendation_profile,
        ranked_popular_user_tags,
        save_recommendation_profile,
    )

    collected_at = utc_now()
    app = client.fetch_app_details(game.appid, language=language, country=country)
    try:
        popular_tags = client.fetch_popular_tags(
            game.appid,
            language=language,
            country=country,
            max_tags=20,
        )
        app["steam_tags"] = popular_tags
        app["popular_user_tags"] = ranked_popular_user_tags(popular_tags)
        app["popular_user_tags_source"] = "store_html.app_tag"
        app["popular_user_tags_collected_at"] = collected_at
        app["popular_user_tags_language"] = language
        app["popular_user_tags_error"] = ""
        app["popular_tags_source"] = "store.html.app_tag"
        app["popular_tags_language"] = language
        app["popular_tags_error"] = ""
    except Exception as exc:
        app["steam_tags"] = []
        app["popular_user_tags"] = []
        app["popular_user_tags_source"] = "store_html.app_tag"
        app["popular_user_tags_collected_at"] = collected_at
        app["popular_user_tags_language"] = language
        app["popular_user_tags_error"] = f"{type(exc).__name__}: {exc}"
        app["popular_tags_source"] = "store.html.app_tag"
        app["popular_tags_language"] = language
        app["popular_tags_error"] = f"{type(exc).__name__}: {exc}"
    store = normalize_store_info(app)
    reviews = [
        normalize_review(item)
        for item in client.fetch_reviews(
            game.appid,
            max_reviews=max_reviews,
            language=language,
        )
    ]
    news = [normalize_news(item) for item in client.fetch_news(game.appid, count=news_count)]
    resolved_name = str(store.get("name") or game.name)
    resolved_key = game.game_key or f"{slugify(resolved_name)}_{game.appid}"
    game = SteamGame(game.appid, resolved_name, resolved_key)
    markdown = build_markdown(
        game,
        store,
        reviews,
        news,
        collected_at=collected_at,
    )
    markdown_path = docs_dir / f"{game.resolved_key()}.md"

    # Write raw evidence first; replace the searchable Markdown only after every API call succeeds.
    _atomic_json(raw_dir / f"{game.resolved_key()}_store_info.json", store)
    _atomic_json(raw_dir / f"{game.resolved_key()}_reviews_normalized.json", reviews)
    _atomic_json(raw_dir / f"{game.resolved_key()}_news_normalized.json", news)
    profile_path = None
    if profiles_dir is not None:
        profile = build_recommendation_profile(
            game,
            store,
            reviews,
            collected_at=collected_at,
            language=language,
            country=country,
        )
        profile_path = profiles_dir / f"{game.resolved_key()}.json"
        save_recommendation_profile(profile_path, profile)
    _atomic_text(markdown_path, markdown)
    return CollectionResult(
        game,
        markdown_path,
        len(reviews),
        len(news),
        collected_at,
        profile_path,
    )


def save_catalog(path: Path, apps: Sequence[dict]) -> None:
    _atomic_json(path, {"schema_version": "steam-catalog-v1", "fetched_at": utc_now(), "apps": list(apps)})
