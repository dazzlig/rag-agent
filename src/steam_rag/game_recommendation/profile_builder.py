from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

from steam_rag.game_metadata.playstyle import build_playstyle_metadata, normalize_steam_tags

if TYPE_CHECKING:
    from steam_rag.steam_collection.steam_client import SteamAPIClient, SteamGame


PROFILE_SCHEMA_VERSION = "steam-recommendation-profile-v1"


def ranked_popular_user_tags(tags: Iterable[str], *, limit: int = 20) -> list[dict[str, object]]:
    """Preserve Steam Store display order as a one-based popularity rank."""

    ranked: list[dict[str, object]] = []
    seen: set[str] = set()
    for tag in tags:
        name = str(tag).strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        normalized = normalize_steam_tags([name])
        ranked.append(
            {
                "name": name,
                "normalized": normalized[0] if normalized else None,
                "rank": len(ranked) + 1,
            }
        )
        if len(ranked) >= limit:
            break
    return ranked


def build_recommendation_profile(
    game: "SteamGame",
    store: dict,
    reviews: Sequence[dict] = (),
    *,
    collected_at: str,
    language: str = "koreana",
    country: str = "KR",
) -> dict[str, object]:
    ranked_tags = list(store.get("popular_user_tags") or [])
    if not ranked_tags:
        ranked_tags = ranked_popular_user_tags(store.get("steam_tags") or [])

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
    playstyle = build_playstyle_metadata(
        "\n\n".join(value for value in (store_text, review_text) if value),
        genres=store.get("genres"),
        categories=store.get("categories"),
        steam_tags=[tag.get("name") for tag in ranked_tags],
        popular_user_tags=ranked_tags,
        includes_reviews=bool(reviews),
        store_text=store_text,
        review_text=review_text,
    )
    structured_sources = {
        "steam_genre",
        "steam_category",
        "steam_popular_user_tag",
    }
    verified_facets: dict[str, list[str]] = {}
    inferred_facets: dict[str, list[str]] = {}
    for field in (
        "combat_facets",
        "perspective_facets",
        "dimension_facets",
        "playstyle_facets",
    ):
        verified = sorted(
            {
                str(item["facet"])
                for item in playstyle["facet_evidence"]
                if item.get("facet_type") == field
                and item.get("source_type") in structured_sources
            }
        )
        verified_facets[field] = verified
        inferred_facets[field] = sorted(set(playstyle[field]) - set(verified))
    positive_count = sum(review.get("voted_up") is True for review in reviews)
    negative_count = sum(review.get("voted_up") is False for review in reviews)
    rated_count = positive_count + negative_count
    positive_ratio = round(positive_count / rated_count, 4) if rated_count else None
    completeness_parts = {
        "genres": 0.20 if store.get("genres") else 0.0,
        "categories": 0.15 if store.get("categories") else 0.0,
        "popular_user_tags": 0.30 if ranked_tags else 0.0,
        "verified_facets": 0.20 if any(verified_facets.values()) else 0.0,
        "store_text": 0.10 if store_text else 0.0,
        "price": 0.05 if store.get("price_available") else 0.0,
    }
    try:
        collected_datetime = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
        if collected_datetime.tzinfo is None:
            collected_datetime = collected_datetime.replace(tzinfo=timezone.utc)
    except ValueError:
        collected_datetime = datetime.now(timezone.utc)

    searchable_terms = sorted(
        {
            str(store.get("name") or game.name),
            *(str(value) for value in store.get("genres") or []),
            *(str(value) for value in store.get("categories") or []),
            *(str(tag.get("name") or "") for tag in ranked_tags),
            *(str(value) for value in playstyle["playstyle_terms_normalized"]),
            *(str(value) for field in ("combat_facets", "perspective_facets", "dimension_facets", "playstyle_facets") for value in verified_facets[field]),
        }
    )
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_tier": "core",
        "profile_completeness": round(sum(completeness_parts.values()), 2),
        "profile_completeness_breakdown": completeness_parts,
        "appid": game.appid,
        "game_key": game.resolved_key(),
        "app_type": store.get("app_type") or "game",
        "name": str(store.get("name") or game.name),
        "language": language,
        "country": country,
        "release_date": store.get("release_date"),
        "release_coming_soon": bool(store.get("release_coming_soon")),
        "header_image": store.get("header_image"),
        "capsule_image": store.get("capsule_image"),
        "developers": list(store.get("developers") or []),
        "publishers": list(store.get("publishers") or []),
        "genres": list(store.get("genres") or []),
        "categories": list(store.get("categories") or []),
        "popular_user_tags": ranked_tags,
        "popular_user_tags_source": store.get("popular_user_tags_source") or store.get("popular_tags_source"),
        "popular_user_tags_collected_at": store.get("popular_user_tags_collected_at") or collected_at,
        "popular_user_tags_language": store.get("popular_user_tags_language") or store.get("popular_tags_language") or language,
        "popular_user_tags_error": store.get("popular_user_tags_error") or store.get("popular_tags_error"),
        "steam_genres_normalized": playstyle["steam_genres_normalized"],
        "steam_categories_normalized": playstyle["steam_categories_normalized"],
        "steam_tags_normalized": playstyle["steam_tags_normalized"],
        "combat_facets": verified_facets["combat_facets"],
        "perspective_facets": verified_facets["perspective_facets"],
        "dimension_facets": verified_facets["dimension_facets"],
        "playstyle_facets": verified_facets["playstyle_facets"],
        "inferred_facets": inferred_facets,
        "facet_usage_policy": {
            "hard_filter": "official metadata and ranked Steam popular user tags",
            "soft_score_only": "store text and recent review inference",
        },
        "facet_evidence": playstyle["facet_evidence"],
        "playstyle_profile_version": playstyle["playstyle_profile_version"],
        "recent_review_summary": {
            "sample_size": len(reviews),
            "rated_count": rated_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_ratio": positive_ratio,
            "language": language,
        },
        "price": {
            "is_free": store.get("is_free"),
            "available": store.get("price_available"),
            "currency": store.get("price_currency"),
            "initial": store.get("price_initial"),
            "final": store.get("price_final"),
            "discount_percent": store.get("price_discount_percent"),
            "collected_at": collected_at,
        },
        "searchable_terms": [term for term in searchable_terms if term],
        "store_summary": store_text[:2000],
        "profile_updated_at": collected_at,
        "profile_expires_at": (collected_datetime + timedelta(days=30)).isoformat(),
        "source_freshness": {
            "genres_categories_ttl_days": 30,
            "popular_user_tags_ttl_days": 30,
            "store_text_ttl_days": 30,
            "price_ttl_hours": 6,
            "reviews_ttl_hours": 24,
            "news_ttl_hours": 24,
        },
    }


def save_recommendation_profile(path: Path, profile: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def collect_recommendation_profile(
    client: "SteamAPIClient",
    game: "SteamGame",
    *,
    profiles_dir: Path,
    language: str = "koreana",
    country: str = "KR",
) -> Path:
    """Collect profile-only evidence without reviews/news or embedding calls."""

    from steam_rag.steam_collection.steam_client import normalize_store_info, utc_now

    collected_at = utc_now()
    app = client.fetch_app_details(game.appid, language=language, country=country)
    try:
        tags = client.fetch_popular_tags(
            game.appid,
            language=language,
            country=country,
            max_tags=20,
        )
        app["steam_tags"] = tags
        app["popular_user_tags"] = ranked_popular_user_tags(tags)
        app["popular_user_tags_source"] = "store_html.app_tag"
        app["popular_user_tags_collected_at"] = collected_at
        app["popular_user_tags_language"] = language
        app["popular_user_tags_error"] = ""
    except Exception as exc:
        app["steam_tags"] = []
        app["popular_user_tags"] = []
        app["popular_user_tags_source"] = "store_html.app_tag"
        app["popular_user_tags_collected_at"] = collected_at
        app["popular_user_tags_language"] = language
        app["popular_user_tags_error"] = f"{type(exc).__name__}: {exc}"
    store = normalize_store_info(app)
    resolved_game = type(game)(
        game.appid,
        str(store.get("name") or game.name),
        game.game_key,
    )
    profile = build_recommendation_profile(
        resolved_game,
        store,
        collected_at=collected_at,
        language=language,
        country=country,
    )
    path = profiles_dir / f"{resolved_game.resolved_key()}.json"
    save_recommendation_profile(path, profile)
    return path
