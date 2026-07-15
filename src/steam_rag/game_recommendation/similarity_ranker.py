from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("&", " and ")
    return re.sub(r"[^a-z0-9가-힣]+", " ", text).strip()


_SEMANTIC_ALIASES_RAW = {
    "anime": "anime",
    "anime style": "anime",
    "anime styled": "anime",
    "animation": "anime",
    "애니": "anime",
    "애니풍": "anime",
    "애니메이션": "anime",
    "애니메이션풍": "anime",
    "free to play": "free_to_play",
    "free2play": "free_to_play",
    "f2p": "free_to_play",
    "무료 플레이": "free_to_play",
    "무료플레이": "free_to_play",
    "action": "action",
    "액션": "action",
    "action rpg": "action_rpg",
    "액션 rpg": "action_rpg",
    "rpg": "rpg",
    "role playing": "rpg",
    "롤플레잉": "rpg",
    "adventure": "adventure",
    "어드벤처": "adventure",
    "open world": "open_world",
    "오픈 월드": "open_world",
    "오픈월드": "open_world",
    "exploration": "exploration",
    "탐험": "exploration",
    "story rich": "story_rich",
    "narrative": "story_rich",
    "풍부한 스토리": "story_rich",
    "스토리 중심": "story_rich",
    "combat": "combat",
    "전투": "combat",
    "real time": "real_time",
    "real time combat": "real_time",
    "실시간": "real_time",
    "실시간 전투": "real_time",
    "turn based": "turn_based",
    "turn based combat": "turn_based",
    "턴제": "turn_based",
    "턴제 전투": "turn_based",
    "direct control": "direct_control",
    "직접 조작": "direct_control",
    "melee": "melee",
    "근접": "melee",
    "근접 전투": "melee",
    "ranged": "ranged",
    "원거리": "ranged",
    "shooter": "shooter",
    "슈팅": "shooter",
    "hack and slash": "hack_and_slash",
    "핵 앤 슬래시": "hack_and_slash",
    "souls like": "souls_like",
    "소울라이크": "souls_like",
    "third person": "third_person",
    "3인칭": "third_person",
    "first person": "first_person",
    "1인칭": "first_person",
    "top down": "top_down",
    "탑다운": "top_down",
    "isometric": "isometric",
    "아이소메트릭": "isometric",
    "2d": "2d",
    "3d": "3d",
    "single player": "singleplayer",
    "singleplayer": "singleplayer",
    "싱글 플레이어": "singleplayer",
    "싱글플레이어": "singleplayer",
    "co op": "co_op",
    "coop": "co_op",
    "협동": "co_op",
    "multiplayer": "multiplayer",
    "멀티플레이어": "multiplayer",
    "character focused": "character_focused",
    "캐릭터 중심": "character_focused",
    "character collection": "character_collection",
    "collect characters": "character_collection",
    "캐릭터 수집": "character_collection",
    "gacha": "gacha",
    "가챠": "gacha",
    "라이브 서비스": "live_service",
    "live service": "live_service",
    "survival": "survival",
    "생존": "survival",
    "crafting": "crafting",
    "제작": "crafting",
    "sandbox": "sandbox",
    "샌드박스": "sandbox",
    "pixel graphics": "pixel_graphics",
    "pixel art": "pixel_graphics",
    "픽셀 그래픽": "pixel_graphics",
    "픽셀 아트": "pixel_graphics",
    "party puzzle": "party_puzzle",
    "파티 퍼즐": "party_puzzle",
    "massively multiplayer": "massively_multiplayer",
    "mmorpg": "massively_multiplayer",
}

SEMANTIC_ALIASES = {_key(alias): value for alias, value in _SEMANTIC_ALIASES_RAW.items()}

FEATURE_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "action_rpg": ("action", "rpg", "real_time"),
    "hack_and_slash": ("action", "real_time", "melee", "direct_control"),
    "souls_like": ("action", "real_time", "boss_focused"),
    "shooter": ("action", "real_time", "ranged"),
    "open_world": ("exploration",),
    "gacha": ("character_collection", "live_service"),
}

HUMAN_LABELS = {
    "anime": "애니풍 비주얼",
    "free_to_play": "무료 플레이",
    "action": "액션 중심",
    "action_rpg": "액션 RPG",
    "rpg": "RPG 성장 구조",
    "open_world": "오픈 월드",
    "exploration": "탐험 중심 진행",
    "story_rich": "스토리 중심",
    "combat": "전투 중심",
    "real_time": "실시간 액션 전투",
    "turn_based": "턴제 전투",
    "direct_control": "직접 조작 전투",
    "melee": "근접 전투",
    "ranged": "원거리 전투",
    "shooter": "슈팅 전투",
    "third_person": "3인칭 시점",
    "first_person": "1인칭 시점",
    "top_down": "탑다운 시점",
    "isometric": "아이소메트릭 시점",
    "2d": "2D 화면",
    "3d": "3D 화면",
    "singleplayer": "싱글플레이",
    "co_op": "협동 플레이",
    "multiplayer": "멀티플레이",
    "character_focused": "캐릭터 중심 구성",
    "character_collection": "캐릭터 수집·육성",
    "live_service": "라이브 서비스",
    "survival": "생존 중심",
    "crafting": "제작 중심",
    "sandbox": "샌드박스 진행",
    "survival_sandbox": "생존·제작 샌드박스",
    "pixel_graphics": "픽셀 그래픽",
    "party_puzzle": "파티 퍼즐",
}


def canonicalize_semantic_tag(value: object) -> str:
    """Map Korean/English Steam labels to one comparison vocabulary."""

    key = _key(value)
    if not key:
        return ""
    return SEMANTIC_ALIASES.get(key, key.replace(" ", "_"))


@dataclass(frozen=True, slots=True)
class ReferenceGame:
    appid: int
    name: str
    matched_alias: str
    source: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "appid": self.appid,
            "name": self.name,
            "matched_alias": self.matched_alias,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class SimilaritySpec:
    seed: ReferenceGame
    must_have: tuple[str, ...]
    should_have: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    feature_weights: tuple[tuple[str, float], ...] = ()
    seed_features: tuple[str, ...] = ()

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.feature_weights)

    def to_dict(self) -> dict[str, Any]:
        search_terms = list(dict.fromkeys((*self.must_have, *self.should_have)))[:8]
        return {
            "seed": self.seed.to_dict(),
            "must_have": list(self.must_have),
            "should_have": list(self.should_have),
            "excluded": list(self.excluded),
            "feature_weights": dict(self.feature_weights),
            "seed_features": list(self.seed_features),
            "search_terms": [value.replace("_", " ") for value in search_terms],
        }


@dataclass(frozen=True, slots=True)
class SimilarityScore:
    appid: int
    name: str
    score: float
    passed_hard_gate: bool
    matched_features: tuple[str, ...]
    matched_aspects: tuple[str, ...]
    missing_must_have: tuple[str, ...]
    excluded_matches: tuple[str, ...]
    hard_gate_reasons: tuple[str, ...] = field(default_factory=tuple)
    profile: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "appid": self.appid,
            "name": self.name,
            "score": self.score,
            "passed_hard_gate": self.passed_hard_gate,
            "matched_features": list(self.matched_features),
            "matched_aspects": list(self.matched_aspects),
            "missing_must_have": list(self.missing_must_have),
            "excluded_matches": list(self.excluded_matches),
            "hard_gate_reasons": list(self.hard_gate_reasons),
        }


# This cache contains only well-known Korean service titles/abbreviations.  It
# resolves an alias to an existing local profile; it is not a hand-authored
# play-style profile and cannot create an unverified Steam game.
HIGH_CONFIDENCE_TITLE_ALIASES: dict[str, tuple[int, str]] = {
    "명조": (3513350, "Wuthering Waves"),
    "워더링 웨이브": (3513350, "Wuthering Waves"),
    "워더링 웨이브스": (3513350, "Wuthering Waves"),
}


def load_local_profiles(profiles_dir: Path) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in sorted(profiles_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("appid") and value.get("name"):
            profiles.append(value)
    return profiles


def _profile_values(profiles: Iterable[object]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for item in profiles:
        candidate = item[1] if isinstance(item, tuple) and len(item) == 2 else item
        if isinstance(candidate, Mapping) and candidate.get("appid") and candidate.get("name"):
            values.append(candidate)
    return values


def _title_aliases(profile: Mapping[str, Any]) -> set[str]:
    name = str(profile.get("name") or "").strip()
    aliases = {name}
    for field_name in ("title_aliases", "aliases", "localized_names"):
        value = profile.get(field_name)
        if isinstance(value, Mapping):
            aliases.update(str(item) for item in value.values())
        elif isinstance(value, (list, tuple, set)):
            aliases.update(str(item) for item in value)
    for separator in (":", " - ", " — "):
        if separator not in name:
            continue
        left, right = (part.strip() for part in name.split(separator, 1))
        if len(_key(left)) >= 4:
            aliases.add(left)
        if len(_key(right)) >= 4:
            aliases.add(right)
    return {_key(alias) for alias in aliases if len(_key(alias)) >= 2}


def _contains_alias(normalized_question: str, alias: str) -> bool:
    return bool(re.search(rf"(?:^| ){re.escape(alias)}(?: |$)", normalized_question))


def resolve_reference_game(
    question: str,
    profiles: Iterable[object],
    *,
    alias_cache: Mapping[str, tuple[int, str] | int] | None = None,
) -> ReferenceGame | None:
    """Resolve a seed title only when it has a matching local Steam profile."""

    local_profiles = _profile_values(profiles)
    by_appid = {int(profile["appid"]): profile for profile in local_profiles}
    explicit = re.search(r"(?:appid\s*[:=]?\s*|/app/)(\d+)", question, flags=re.IGNORECASE)
    if explicit:
        appid = int(explicit.group(1))
        profile = by_appid.get(appid)
        if profile is not None:
            return ReferenceGame(appid, str(profile["name"]), str(appid), "explicit_appid", 1.0)

    normalized_question = _key(question)
    combined_cache = dict(HIGH_CONFIDENCE_TITLE_ALIASES)
    if alias_cache:
        combined_cache.update({_key(alias): target for alias, target in alias_cache.items()})
    cache_matches: list[tuple[int, str, Mapping[str, Any]]] = []
    for raw_alias, target in combined_cache.items():
        alias = _key(raw_alias)
        appid = int(target[0] if isinstance(target, tuple) else target)
        profile = by_appid.get(appid)
        if profile is not None and _contains_alias(normalized_question, alias):
            cache_matches.append((len(alias), alias, profile))
    if cache_matches:
        _, alias, profile = max(cache_matches, key=lambda item: item[0])
        return ReferenceGame(
            int(profile["appid"]), str(profile["name"]), alias, "high_confidence_alias", 0.99
        )

    title_matches: list[tuple[int, bool, str, Mapping[str, Any]]] = []
    for profile in local_profiles:
        canonical = _key(profile.get("name"))
        for alias in _title_aliases(profile):
            if not _contains_alias(normalized_question, alias):
                continue
            title_matches.append((len(alias), alias == canonical, alias, profile))
    if not title_matches:
        return None
    longest = max(item[0] for item in title_matches)
    best = [item for item in title_matches if item[0] == longest]
    appids = {int(item[3]["appid"]) for item in best}
    if len(appids) != 1:
        return None
    _, exact, alias, profile = max(best, key=lambda item: item[1])
    return ReferenceGame(
        int(profile["appid"]),
        str(profile["name"]),
        alias,
        "canonical_title" if exact else "generated_title_alias",
        0.98 if exact else 0.94,
    )


def _rank_weight(rank: object) -> float:
    try:
        value = max(1, int(rank))
    except (TypeError, ValueError):
        value = 20
    return 1.0 + 1.4 / math.log2(value + 1)


def _add_feature(target: dict[str, float], value: object, weight: float) -> None:
    feature = canonicalize_semantic_tag(value)
    if not feature:
        return
    target[feature] = max(target.get(feature, 0.0), weight)
    for expanded in FEATURE_EXPANSIONS.get(feature, ()):
        target[expanded] = max(target.get(expanded, 0.0), weight * 0.9)


def _profile_feature_weights(profile: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for item in profile.get("popular_user_tags") or ():
        if not isinstance(item, Mapping):
            continue
        weight = _rank_weight(item.get("rank"))
        _add_feature(features, item.get("normalized"), weight)
        _add_feature(features, item.get("name"), weight)
    for field_name, weight in (
        ("steam_tags_normalized", 1.4),
        ("steam_genres_normalized", 1.7),
        ("steam_categories_normalized", 1.4),
        ("combat_facets", 1.9),
        ("perspective_facets", 1.7),
        ("dimension_facets", 1.5),
        ("playstyle_facets", 1.8),
    ):
        for value in profile.get(field_name) or ():
            _add_feature(features, value, weight)
    inferred = profile.get("inferred_facets") or {}
    if isinstance(inferred, Mapping):
        for values in inferred.values():
            if isinstance(values, (list, tuple, set)):
                for value in values:
                    _add_feature(features, value, 0.7)
    price = profile.get("price") or {}
    if isinstance(price, Mapping) and price.get("is_free") is True:
        _add_feature(features, "free_to_play", 1.3)

    text = _key(profile.get("store_summary"))
    text_patterns = {
        "character_collection": (r"character collection", r"collect characters", r"캐릭터 수집"),
        "character_focused": (r"character focused", r"캐릭터 중심"),
        "live_service": (r"live service", r"라이브 서비스"),
        "open_world": (r"open world", r"오픈 월드", r"오픈월드"),
        "real_time": (r"real time combat", r"실시간 전투"),
    }
    for feature, patterns in text_patterns.items():
        if any(re.search(pattern, text) for pattern in patterns):
            _add_feature(features, feature, 0.65)

    if {"anime", "rpg"} <= features.keys() and re.search(
        r"characters?|companions?|operatives?|캐릭터|동료|파트너|공명자", text
    ):
        _add_feature(features, "character_focused", 0.75)
    if re.search(
        r"collect (?:and )?(?:unlock )?characters?|unlock .*characters?|"
        r"캐릭터 (?:수집|획득|해금)|동료 (?:수집|획득|영입)",
        text,
    ):
        _add_feature(features, "character_collection", 0.75)

    if "survival" in features and ({"sandbox", "crafting"} & features.keys()):
        features["survival_sandbox"] = max(features["survival"], 1.0)
    if "party_puzzle" in features or ({"co_op", "puzzle"} <= features.keys()):
        features["party_puzzle"] = max(features.get("party_puzzle", 0.0), 1.0)
    return features


def extract_profile_features(profile: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(_profile_feature_weights(profile))


def build_similarity_spec(
    seed_profile: Mapping[str, Any],
    *,
    reference: ReferenceGame | None = None,
) -> SimilaritySpec:
    """Create a deterministic, evidence-derived similarity contract."""

    if reference is None:
        reference = ReferenceGame(
            int(seed_profile["appid"]),
            str(seed_profile["name"]),
            str(seed_profile["name"]),
            "seed_profile",
            1.0,
        )
    weighted = _profile_feature_weights(seed_profile)
    features = set(weighted)

    must: list[str] = []
    if "anime" in features:
        must.append("anime")
    for feature in ("rpg", "strategy", "simulation", "survival"):
        if feature in features:
            must.append(feature)
            break
    for feature in ("turn_based", "real_time", "shooter", "tactical"):
        if feature in features:
            must.append(feature)
            break
    if len(must) < 2:
        for feature in ("action", "open_world", "co_op", "third_person", "2d", "3d"):
            if feature in features and feature not in must:
                must.append(feature)
            if len(must) >= 2:
                break

    should_priority = (
        "open_world",
        "action",
        "third_person",
        "first_person",
        "3d",
        "2d",
        "exploration",
        "story_rich",
        "direct_control",
        "melee",
        "ranged",
        "character_focused",
        "character_collection",
        "free_to_play",
        "live_service",
        "singleplayer",
        "co_op",
        "multiplayer",
    )
    should = [feature for feature in should_priority if feature in features and feature not in must]

    excluded: list[str] = []
    if "real_time" in features and "turn_based" not in features:
        excluded.append("turn_based")
    if "third_person" in features and "3d" in features and "pixel_graphics" not in features:
        excluded.append("pixel_graphics")
    if {"open_world", "rpg", "real_time"} <= features and "survival_sandbox" not in features:
        excluded.append("survival_sandbox")
    if "party_puzzle" not in features and {"rpg", "real_time"} <= features:
        excluded.append("party_puzzle")

    selected = [*must, *should]
    feature_weights = []
    for feature in selected:
        base = 3.0 if feature in must else 1.0
        feature_weights.append((feature, round(max(base, weighted.get(feature, 0.0)), 4)))
    return SimilaritySpec(
        seed=reference,
        must_have=tuple(must),
        should_have=tuple(should),
        excluded=tuple(excluded),
        feature_weights=tuple(feature_weights),
        seed_features=tuple(sorted(features)),
    )


def adapt_similarity_spec_to_question(spec: SimilaritySpec, question: str) -> SimilaritySpec:
    """Adjust emphasis from explicit user wording without redefining the seed.

    In Korean game discourse, ``서브컬처 게임`` primarily asks for an
    anime/character-centred product family.  Combat and world structure still
    affect rank, but turn-based games must not be rejected solely because the
    reference title happens to use real-time combat.
    """

    must = list(spec.must_have)
    should = list(spec.should_have)
    excluded = list(spec.excluded)
    weights = spec.weights
    if "서브컬처" in _key(question).replace(" ", ""):
        must = [feature for feature in ("anime", "rpg") if feature in spec.seed_features]
        if not must:
            must = list(spec.must_have[:2])
        concept_features = (
            "character_focused",
            "character_collection",
            "open_world",
            "real_time",
            "action",
            "third_person",
            "3d",
            "exploration",
            "story_rich",
            "free_to_play",
            "live_service",
        )
        should = list(
            dict.fromkeys(
                feature
                for feature in (*concept_features, *spec.should_have, *spec.must_have)
                if feature not in must
            )
        )
        excluded = [feature for feature in spec.excluded if feature != "turn_based"]
        weights.update(
            {
                "anime": 4.5,
                "rpg": 3.5,
                "character_focused": 3.0,
                "character_collection": 3.0,
                "open_world": 4.0,
                "real_time": 2.4,
                "free_to_play": 3.5,
                "live_service": 1.8,
                "singleplayer": 0.7,
                "co_op": 0.7,
                "multiplayer": 0.7,
            }
        )

    # Follow-up turns are applied as a delta over the persisted seed contract.
    # This prevents a short phrase such as "그중 협동만" from discarding the
    # original game's genre/play-style requirements.
    terms = {
        "turn_based": r"턴제|turn[ -]?based",
        "real_time": r"실시간|real[ -]?time",
        "co_op": r"협동|코옵|co[ -]?op",
        "multiplayer": r"멀티(?:플레이)?|multiplayer",
        "singleplayer": r"싱글(?:플레이)?|single[ -]?player",
        "story_rich": r"스토리|서사|story",
        "open_world": r"오픈\s*월드|open[ -]?world",
        "free_to_play": r"무료|free[ -]?to[ -]?play",
        "survival": r"생존|서바이벌|survival",
        "rpg": r"\brpg\b|롤플레잉",
    }
    lowered = question.casefold()
    for feature, term in terms.items():
        if not re.search(term, lowered, flags=re.IGNORECASE):
            continue
        negative = bool(
            re.search(
                rf"(?:{term}).{{0,12}}(?:빼고|제외|말고|싫|아닌|없(?:는|이))|"
                rf"(?:빼고|제외|말고).{{0,12}}(?:{term})",
                lowered,
                flags=re.IGNORECASE,
            )
        )
        neutral = bool(
            re.search(rf"(?:{term}).{{0,10}}상관\s*없", lowered, flags=re.IGNORECASE)
        )
        required = bool(
            re.search(
                rf"(?:{term}).{{0,15}}(?:만|필수|원해|원하는|가능한|지원(?:하|해)|이어야)|"
                rf"(?:반드시|꼭).{{0,12}}(?:{term})",
                lowered,
                flags=re.IGNORECASE,
            )
        )
        must = [item for item in must if item != feature]
        should = [item for item in should if item != feature]
        excluded = [item for item in excluded if item != feature]
        if neutral:
            weights.pop(feature, None)
        elif negative:
            excluded.append(feature)
        elif required:
            must.append(feature)
            weights[feature] = max(4.0, weights.get(feature, 0.0))
        else:
            should.append(feature)
            weights[feature] = max(2.0, weights.get(feature, 0.0))

    must = list(dict.fromkeys(must))
    should = [feature for feature in dict.fromkeys(should) if feature not in must]
    excluded = [
        feature
        for feature in dict.fromkeys(excluded)
        if feature not in must and feature not in should
    ]
    return SimilaritySpec(
        seed=spec.seed,
        must_have=tuple(must),
        should_have=tuple(should),
        excluded=tuple(excluded),
        feature_weights=tuple((feature, weights.get(feature, 1.0)) for feature in (*must, *should)),
        seed_features=spec.seed_features,
    )


def describe_similarity_spec(spec: SimilaritySpec) -> str:
    focus = humanize_aspects((*spec.must_have, *spec.should_have[:5]))
    return (
        f"기준 게임 **{spec.seed.name}**와 요청에 포함된 "
        f"{', '.join(focus)}을 중심으로 Steam 후보를 비교했습니다."
    )


def humanize_aspects(features: Iterable[str]) -> tuple[str, ...]:
    return tuple(HUMAN_LABELS.get(feature, feature.replace("_", " ")) for feature in features)


def score_profile_similarity(
    profile: Mapping[str, Any],
    spec: SimilaritySpec,
) -> SimilarityScore:
    features = extract_profile_features(profile)
    selected = (*spec.must_have, *spec.should_have)
    matched = tuple(feature for feature in selected if feature in features)
    missing = tuple(feature for feature in spec.must_have if feature not in features)
    excluded = tuple(feature for feature in spec.excluded if feature in features)
    try:
        appid = int(profile.get("appid"))
    except (TypeError, ValueError):
        appid = 0
    is_seed = appid == spec.seed.appid
    reasons = [f"필수 유사점 부족: {', '.join(humanize_aspects(missing))}" for _ in [0] if missing]
    if excluded:
        reasons.append(f"비유사 특성 포함: {', '.join(humanize_aspects(excluded))}")
    if is_seed:
        reasons.append("기준 게임 자체")
    passed = not missing and not excluded and not is_seed and appid > 0

    weights = spec.weights
    denominator = sum(weights.get(feature, 1.0) for feature in selected) or 1.0
    numerator = sum(weights.get(feature, 1.0) for feature in matched)
    raw_score = 100.0 * numerator / denominator
    raw_score -= 22.0 * len(missing) + 35.0 * len(excluded)
    return SimilarityScore(
        appid=appid,
        name=str(profile.get("name") or profile.get("game_key") or appid),
        score=round(max(0.0, min(100.0, raw_score)), 2),
        passed_hard_gate=passed,
        matched_features=matched,
        matched_aspects=humanize_aspects(matched),
        missing_must_have=missing,
        excluded_matches=excluded,
        hard_gate_reasons=tuple(reasons),
        profile=profile,
    )


def rank_similar_profiles(
    profiles: Sequence[object],
    spec: SimilaritySpec,
    *,
    limit: int = 10,
    include_failed: bool = False,
) -> list[SimilarityScore]:
    scores = [score_profile_similarity(profile, spec) for profile in _profile_values(profiles)]
    if not include_failed:
        scores = [score for score in scores if score.passed_hard_gate]
    scores.sort(key=lambda item: (-int(item.passed_hard_gate), -item.score, item.name.casefold(), item.appid))
    return scores[: max(0, int(limit))]
