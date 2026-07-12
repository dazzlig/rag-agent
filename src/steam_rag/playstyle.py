from __future__ import annotations

import ast
import re
from typing import Iterable, Mapping


PROFILE_VERSION = "playstyle-v1"


def _label_key(value: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", value.casefold()).strip()


TAG_ALIASES = {
    "action": "action",
    "action rpg": "action_rpg",
    "액션 rpg": "action_rpg",
    "rpg": "rpg",
    "role playing": "rpg",
    "action adventure": "action_adventure",
    "adventure": "adventure",
    "strategy": "strategy",
    "indie": "indie",
    "turn based combat": "turn_based_combat",
    "턴제 전투": "turn_based_combat",
    "turn based tactics": "turn_based_tactics",
    "real time combat": "real_time_combat",
    "souls like": "souls_like",
    "소울라이크": "souls_like",
    "metroidvania": "metroidvania",
    "메트로배니아": "metroidvania",
    "platformer": "platformer",
    "2d platformer": "platformer_2d",
    "2d": "2d",
    "2 5d": "2_5d",
    "3d": "3d",
    "3d platformer": "platformer_3d",
    "side scroller": "side_scroller",
    "third person": "third_person",
    "3인칭": "third_person",
    "first person": "first_person",
    "1인칭": "first_person",
    "top down": "top_down",
    "isometric": "isometric",
    "open world": "open_world",
    "story rich": "story_rich",
    "choices matter": "choices_matter",
    "exploration": "exploration",
    "survival": "survival",
    "crafting": "crafting",
    "co op": "co_op",
    "online co op": "online_co_op",
    "multi player": "multiplayer",
    "multiplayer": "multiplayer",
    "single player": "singleplayer",
    "singleplayer": "singleplayer",
    "pvp": "pvp",
    "online pvp": "online_pvp",
    "vr supported": "vr",
    "party based rpg": "party_based_rpg",
    "hunting": "hunting",
}


TAG_TO_FACETS = {
    "action_rpg": {"combat_facets": ["real_time"], "playstyle_facets": ["character_progression"]},
    "action_adventure": {"combat_facets": ["real_time"], "playstyle_facets": ["exploration"]},
    "turn_based_combat": {"combat_facets": ["turn_based"]},
    "turn_based_tactics": {"combat_facets": ["turn_based", "tactical"]},
    "real_time_combat": {"combat_facets": ["real_time"]},
    "souls_like": {"combat_facets": ["real_time", "boss_focused"], "playstyle_facets": ["souls_like"]},
    "metroidvania": {"playstyle_facets": ["metroidvania", "exploration"]},
    "platformer": {"playstyle_facets": ["platforming"]},
    "platformer_2d": {"dimension_facets": ["2d"], "playstyle_facets": ["platforming"]},
    "platformer_3d": {"dimension_facets": ["3d"], "playstyle_facets": ["platforming"]},
    "2d": {"dimension_facets": ["2d"]},
    "2_5d": {"dimension_facets": ["2_5d"]},
    "3d": {"dimension_facets": ["3d"]},
    "side_scroller": {"perspective_facets": ["side_view"]},
    "third_person": {"perspective_facets": ["third_person"]},
    "first_person": {"perspective_facets": ["first_person"]},
    "top_down": {"perspective_facets": ["top_down"]},
    "isometric": {"perspective_facets": ["isometric"]},
    "open_world": {"playstyle_facets": ["open_world", "exploration"]},
    "story_rich": {"playstyle_facets": ["story_rich"]},
    "choices_matter": {"playstyle_facets": ["choices_matter"]},
    "exploration": {"playstyle_facets": ["exploration"]},
    "survival": {"playstyle_facets": ["survival"]},
    "crafting": {"playstyle_facets": ["crafting"]},
    "co_op": {"playstyle_facets": ["co_op"]},
    "online_co_op": {"playstyle_facets": ["co_op", "online_multiplayer"]},
    "multiplayer": {"playstyle_facets": ["multiplayer"]},
    "pvp": {"playstyle_facets": ["pvp"]},
    "online_pvp": {"playstyle_facets": ["pvp", "online_multiplayer"]},
    "vr": {"dimension_facets": ["vr"]},
    "party_based_rpg": {"combat_facets": ["party_based"], "playstyle_facets": ["character_progression"]},
    "hunting": {"combat_facets": ["boss_focused"], "playstyle_facets": ["hunting"]},
}


FACET_PATTERNS = {
    "combat_facets": {
        "turn_based": (r"\bturn[- ]based\b", r"\binitiative\b", r"턴제", r"턴 기반"),
        "real_time": (r"\breal[- ]time\b", r"\baction combat\b", r"실시간", r"액션 전투"),
        "melee": (r"\bmelee\b", r"\bsword(?:s|play)?\b", r"근접 전투", r"검술"),
        "ranged": (r"\branged\b", r"\bgun(?:s|play)?\b", r"\bshoot(?:er|ing)?\b", r"원거리 전투", r"총기"),
        "tactical": (r"\btactical\b", r"\bstrategy\b", r"전술", r"전략"),
        "party_based": (r"\bparty[- ]based\b", r"\bgather your party\b", r"파티 기반"),
        "boss_focused": (r"\bboss(?:es| fight| battle)?\b", r"보스전", r"보스 전투"),
        "direct_control": (r"\bdirect[- ]control\b", r"직접 조작", r"액션 조작"),
        "command_based": (r"\bcommand[- ]based\b", r"명령식", r"커맨드 전투"),
        "auto_combat": (r"\bauto(?:matic)? combat\b", r"자동 전투"),
        "shooter": (r"\bshooter\b", r"\bfps\b", r"\btps\b", r"슈팅"),
        "stealth": (r"\bstealth\b", r"잠입"),
    },
    "perspective_facets": {
        "first_person": (r"\bfirst[- ]person\b", r"\bfps\b", r"1인칭"),
        "third_person": (r"\bthird[- ]person\b", r"3인칭", r"백뷰"),
        "side_view": (r"\bside[- ](?:view|scroll(?:er|ing)?)\b", r"횡스크롤", r"사이드뷰"),
        "top_down": (r"\btop[- ]down\b", r"탑다운", r"쿼터뷰"),
        "isometric": (r"\bisometric\b", r"아이소메트릭"),
    },
    "dimension_facets": {
        "2d": (r"(?<![a-z0-9])2d(?![a-z0-9])", r"2차원"),
        "2_5d": (r"(?<![a-z0-9])2\.5d(?![a-z0-9])", r"2\.5차원"),
        "3d": (r"(?<![a-z0-9])3d(?![a-z0-9])", r"3차원"),
        "vr": (r"\bvirtual reality\b", r"\bvr\b", r"가상현실"),
    },
    "playstyle_facets": {
        "exploration": (r"\bexplor(?:e|ation|ing)\b", r"탐험"),
        "open_world": (r"\bopen[- ]world\b", r"오픈월드"),
        "story_rich": (r"\bstory[- ]rich\b", r"\bnarrative\b", r"스토리 중심", r"서사 중심"),
        "choices_matter": (r"\bchoices? (?:shape|matter|define)\b", r"선택 중심", r"선택이 .*영향"),
        "co_op": (r"\bco[- ]op\b", r"협동"),
        "platforming": (r"\bplatform(?:er|ing)?\b", r"플랫포밍", r"플랫폼 액션"),
        "survival": (r"\bsurvival\b", r"생존"),
        "crafting": (r"\bcraft(?:ing)?\b", r"제작"),
        "hunting": (r"\bhunt(?:er|ing)?\b", r"사냥"),
        "metroidvania": (r"\bmetroidvania\b", r"메트로배니아"),
        "character_progression": (r"\bcharacter progression\b", r"\bclasses\b", r"캐릭터 성장"),
        "souls_like": (r"\bsouls[- ]like\b", r"소울라이크"),
    },
}


FACET_FIELDS = tuple(FACET_PATTERNS)


def coerce_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(item) for item in parsed]
            except (SyntaxError, ValueError):
                pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def normalize_steam_tags(tags: object) -> list[str]:
    """Normalize Steam tag/genre/category labels to stable snake_case values."""

    normalized: set[str] = set()
    for tag in coerce_list(tags):
        key = _label_key(tag)
        canonical = TAG_ALIASES.get(key)
        if canonical is None:
            canonical = re.sub(r"[^a-z0-9가-힣]+", "_", tag.casefold()).strip("_")
        if canonical:
            normalized.add(canonical)
    return sorted(normalized)


def infer_facets(text: str, normalized_tags: Iterable[str] = ()) -> dict[str, list[str]]:
    lowered = text.casefold()
    facets: dict[str, set[str]] = {field: set() for field in FACET_FIELDS}
    for field, values in FACET_PATTERNS.items():
        for facet, patterns in values.items():
            if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
                facets[field].add(facet)
    for tag in normalized_tags:
        for field, values in TAG_TO_FACETS.get(tag, {}).items():
            facets[field].update(values)
    return {field: sorted(values) for field, values in facets.items()}


def build_playstyle_metadata(
    text: str,
    *,
    genres: object = None,
    categories: object = None,
    steam_tags: object = None,
    includes_reviews: bool = False,
) -> dict[str, object]:
    genre_tags = normalize_steam_tags(genres)
    category_tags = normalize_steam_tags(categories)
    supplied_tags = normalize_steam_tags(steam_tags)
    all_tags = sorted(set(genre_tags + category_tags + supplied_tags))
    facets = infer_facets(text, all_tags)

    if supplied_tags and includes_reviews:
        source = "steam_popular_tags_store_text_and_reviews"
    elif supplied_tags:
        source = "steam_popular_tags_and_store_text"
    elif includes_reviews:
        source = "steam_store_text_and_reviews"
    else:
        source = "inferred_from_store_text"
    confidence: object = "medium_high" if supplied_tags else "medium"

    return {
        "playstyle_profile_version": PROFILE_VERSION,
        "playstyle_profile_source": source,
        "playstyle_profile_confidence": confidence,
        "steam_tags_normalized": supplied_tags,
        "steam_genres_normalized": genre_tags,
        "steam_categories_normalized": category_tags,
        "playstyle_terms_normalized": all_tags,
        "playstyle_evidence_sources": [
            source_name
            for source_name, available in (
                ("steam_store_popular_tags", bool(supplied_tags)),
                ("steam_genres", bool(genre_tags)),
                ("steam_categories", bool(category_tags)),
                ("steam_store_text", bool(text.strip())),
                ("steam_recent_reviews", includes_reviews),
            )
            if available
        ],
        **facets,
    }


def extract_query_facets(question: str) -> dict[str, list[str]]:
    label = _label_key(question)
    tags = {
        canonical
        for alias, canonical in TAG_ALIASES.items()
        if re.search(rf"(?:^| ){re.escape(alias)}(?: |$)", label)
    }
    return infer_facets(question, tags)


def facet_match_score(
    query_facets: Mapping[str, Iterable[str]], metadata: Mapping[str, object]
) -> tuple[float, list[str], list[str]]:
    """Return a soft facet score; missing metadata remains neutral."""

    matched: list[str] = []
    conflicts: list[str] = []
    score = 0.0
    mutually_exclusive = {
        "combat_facets": {"turn_based": "real_time", "real_time": "turn_based"},
        "dimension_facets": {"2d": "3d", "3d": "2d"},
    }
    for field in FACET_FIELDS:
        wanted = set(query_facets.get(field, []))
        available = set(coerce_list(metadata.get(field)))
        for facet in sorted(wanted & available):
            matched.append(f"{field}:{facet}")
            score += 0.2
        for facet in sorted(wanted):
            opposite = mutually_exclusive.get(field, {}).get(facet)
            if opposite and opposite in available:
                conflicts.append(f"{field}:{facet}!={opposite}")
                score -= 0.3
    return max(-0.8, min(1.0, score)), matched, conflicts
