from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence

from .index import VectorIndex
from .models import Document, SearchResult
from .playstyle import extract_query_facets, facet_match_score


GAME_ALIASES = {
    "baldurs_gate_3": ["baldur's gate 3", "baldurs gate 3", "baldur's gate", "발더스 게이트 3", "발더스 게이트", "발더스"],
    "cyberpunk_2077": ["cyberpunk 2077", "cyberpunk", "사이버펑크 2077", "사이버펑크"],
    "hollow_knight": ["hollow knight", "할로우 나이트", "할로우"],
    "monster_hunter_world": ["monster hunter: world", "monster hunter world", "monster hunter", "몬스터 헌터 월드", "몬스터헌터 월드", "몬스터 헌터", "몬헌"],
    "no_mans_sky": ["no man's sky", "no mans sky", "nomanssky", "노 맨즈 스카이", "노맨즈 스카이", "노맨즈"],
}

SECTION_POLICY = {
    "review": {"primary": ["review"], "secondary": ["store_summary", "about"]},
    "news": {"primary": ["news"], "secondary": ["metadata"]},
    "price": {"primary": ["metadata", "store_summary"], "secondary": ["news"]},
    "after_update": {"primary": ["review", "news"], "secondary": ["store_summary"]},
    "gameplay": {"primary": ["about", "store_summary"], "secondary": ["metadata", "review"]},
    "general": {"primary": ["store_summary", "about", "metadata"], "secondary": ["review"]},
}

PATCH_NEWS_TYPES = {"hotfix", "patch_note", "major_update", "content_update"}
LOW_VALUE_NEWS_TYPES = {"sale_promo", "community_event", "franchise_promo", "unrelated"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9가-힣']+", text.casefold())


def detect_game_key(question: str) -> str | None:
    lowered = question.casefold()
    aliases = sorted(
        ((alias.casefold(), key) for key, values in GAME_ALIASES.items() for alias in values),
        key=lambda value: len(value[0]),
        reverse=True,
    )
    return next((key for alias, key in aliases if alias in lowered), None)


def detect_intent(question: str) -> str:
    lowered = question.casefold()
    price = any(word in lowered for word in ("가격", "할인", "세일", "구매", "정가", "price", "discount", "sale", "deal", "buy"))
    review = any(word in lowered for word in ("리뷰", "반응", "평가", "민심", "유저", "review", "sentiment", "reaction"))
    news = any(word in lowered for word in ("업데이트", "패치", "핫픽스", "뉴스", "update", "patch", "hotfix", "news"))
    after = any(word in lowered for word in ("이후", "후에", "뒤에", "after", "since"))
    if price:
        return "price"
    if review and news and after:
        return "after_update"
    if review:
        return "review"
    if news:
        return "news"
    if any(word in lowered for word in (
        "플레이", "전투", "조작", "시점", "장르", "특징", "루프", "턴제", "실시간",
        "1인칭", "3인칭", "횡스크롤", "탑다운", "2d", "2.5d", "3d", "소울라이크", "메트로배니아",
        "gameplay", "combat", "camera", "genre", "perspective", "turn-based", "real-time",
        "first person", "third person", "side-scroller", "top-down", "souls-like", "metroidvania",
    )):
        return "gameplay"
    return "general"


def augment_query(question: str, intent: str) -> str:
    suffix = {
        "review": "recent user reviews player reaction positive negative sentiment",
        "news": "latest update patch notes hotfix release announcement",
        "price": "current stored price discount sale currency final initial price overview",
        "after_update": "reviews after latest update patch player sentiment reaction",
        "gameplay": "gameplay combat exploration progression controls camera perspective",
    }.get(intent, "")
    if intent == "gameplay":
        facets = extract_query_facets(question)
        facet_terms = [facet.replace("_", " ") for values in facets.values() for facet in values]
        suffix = f"{suffix} {' '.join(facet_terms)}".strip()
    return f"{question} {suffix}".strip()


def parse_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    for pattern in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(str(value).strip(), pattern).date()
        except ValueError:
            continue
    return None


def recency_score(source_date: object, reference_date: date, half_life_days: int) -> float:
    parsed = parse_date(source_date)
    if parsed is None:
        return 0.0
    age = max((reference_date - parsed).days, 0)
    return math.exp(-age / half_life_days)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("query and document embedding dimensions do not match")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _dedup_key(document: Document) -> tuple[object, ...]:
    metadata = document.metadata
    if metadata.get("section") in {"review", "news"}:
        return metadata.get("game_key"), metadata.get("section"), metadata.get("item_title")
    return (metadata.get("chunk_id") or id(document),)


def _bm25_scores(query: str, documents: Sequence[Document]) -> list[float]:
    tokens = [tokenize(document.page_content) for document in documents]
    if not tokens:
        return []
    query_terms = tokenize(query)
    document_frequency: Counter[str] = Counter()
    for row in tokens:
        document_frequency.update(set(row))
    average_length = sum(map(len, tokens)) / len(tokens) or 1.0
    scores: list[float] = []
    for row in tokens:
        frequencies = Counter(row)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (len(tokens) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(row) / average_length)
            score += inverse_frequency * frequency * 2.5 / denominator
        scores.append(score)
    return scores


@dataclass(slots=True)
class _Candidate:
    document_index: int
    role: str
    local_rank: int
    dense_rank: int | None
    bm25_rank: int | None
    rrf_score: float


class HybridTimeAwareRetriever:
    """Section-aware BM25+dense RRF with week-8 absolute/relative recency."""

    def __init__(self, index: VectorIndex, *, reference_date: date | None = None) -> None:
        self.index = index
        self.reference_date = reference_date or date.today()
        self._latest_patches = self._find_latest_patches()

    def _find_latest_patches(self) -> dict[str, tuple[date, str]]:
        latest: dict[str, tuple[date, str]] = {}
        for document in self.index.documents:
            metadata = document.metadata
            if metadata.get("section") != "news":
                continue
            text = f"{metadata.get('item_title', '')} {document.page_content}".casefold()
            relevance = metadata.get("relevance_type")
            news_type = metadata.get("news_type")
            if news_type not in PATCH_NEWS_TYPES and relevance != "valid_update_or_patch" and not any(
                word in text for word in ("patch", "hotfix", "update", "now live", "업데이트", "패치")
            ):
                continue
            parsed = parse_date(metadata.get("source_date"))
            game_key = str(metadata.get("game_key", ""))
            if parsed and game_key and (game_key not in latest or parsed > latest[game_key][0]):
                latest[game_key] = (parsed, str(metadata.get("item_title", "")))
        return latest

    def _detect_game_key(self, question: str) -> str | None:
        known = detect_game_key(question)
        if known:
            return known
        normalized_question = re.sub(r"[^a-z0-9가-힣]+", " ", question.casefold()).strip()
        candidates: list[tuple[int, str]] = []
        for document in self.index.documents:
            metadata = document.metadata
            name = str(metadata.get("game_name") or "").strip()
            game_key = str(metadata.get("game_key") or "").strip()
            if not name or not game_key:
                continue
            normalized_name = re.sub(r"[^a-z0-9가-힣]+", " ", name.casefold()).strip()
            if normalized_name and re.search(
                rf"(?:^| ){re.escape(normalized_name)}(?: |$)", normalized_question
            ):
                candidates.append((len(normalized_name), game_key))
        return max(candidates, default=(0, ""))[1] or None

    def _rank_section(
        self,
        query: str,
        query_embedding: Sequence[float],
        game_key: str | None,
        section: str,
        per_section_k: int,
    ) -> list[tuple[int, int | None, int | None, float]]:
        indices = [
            index
            for index, document in enumerate(self.index.documents)
            if document.metadata.get("section") == section
            and (not game_key or document.metadata.get("game_key") == game_key)
        ]
        if not indices:
            return []
        dense = sorted(
            ((index, _cosine(query_embedding, self.index.embeddings[index])) for index in indices),
            key=lambda value: value[1],
            reverse=True,
        )[:per_section_k]
        bm25_values = _bm25_scores(query, [self.index.documents[index] for index in indices])
        bm25 = sorted(zip(indices, bm25_values), key=lambda value: value[1], reverse=True)[:per_section_k]
        fused: dict[tuple[object, ...], dict[str, object]] = {}
        for source, ranked, weight in (("dense", dense, 0.5), ("bm25", bm25, 0.5)):
            for rank, (document_index, _score) in enumerate(ranked, start=1):
                key = _dedup_key(self.index.documents[document_index])
                row = fused.setdefault(key, {"index": document_index, "rrf": 0.0, "dense": None, "bm25": None})
                row["rrf"] = float(row["rrf"]) + weight / (60 + rank)
                row[source] = rank
        ordered = sorted(fused.values(), key=lambda value: float(value["rrf"]), reverse=True)
        return [
            (int(row["index"]), row["dense"], row["bm25"], float(row["rrf"]))
            for row in ordered
        ]

    def retrieve(
        self,
        question: str,
        query_embedding: Sequence[float],
        *,
        k: int = 5,
        per_section_k: int = 12,
    ) -> list[SearchResult]:
        if k <= 0 or per_section_k <= 0:
            raise ValueError("k and per_section_k must be positive")
        intent = detect_intent(question)
        game_key = self._detect_game_key(question)
        policy = SECTION_POLICY[intent]
        query = augment_query(question, intent)
        half_life = 30 if any(word in question.casefold() for word in ("최근", "latest", "recent")) else 90
        candidates: dict[tuple[object, ...], _Candidate] = {}
        search_plan = [("primary", section) for section in policy["primary"]] + [
            ("secondary", section) for section in policy["secondary"]
        ]
        for role, section in search_plan:
            rows = self._rank_section(query, query_embedding, game_key, section, per_section_k)
            for local_rank, (document_index, dense_rank, bm25_rank, rrf) in enumerate(rows, start=1):
                key = _dedup_key(self.index.documents[document_index])
                candidate = _Candidate(document_index, role, local_rank, dense_rank, bm25_rank, rrf)
                previous = candidates.get(key)
                if previous is None or candidate.rrf_score > previous.rrf_score:
                    candidates[key] = candidate

        relative_scores = self._relative_recency(candidates.values(), half_life, intent)
        query_facets = extract_query_facets(question) if intent == "gameplay" else {}
        results: list[SearchResult] = []
        for candidate in candidates.values():
            document = self.index.documents[candidate.document_index]
            metadata = document.metadata
            source_date = metadata.get("source_date")
            dated_section = metadata.get("section") in {"news", "review"}
            absolute = (
                recency_score(source_date, self.reference_date, half_life)
                if intent in {"news", "review", "after_update", "price"} and dated_section
                else 0.0
            )
            relative = relative_scores.get(candidate.document_index, 0.0)
            lexical_bonus = self._content_bonus(document, question, intent)
            facet_score, matched_facets, conflicting_facets = (
                facet_match_score(query_facets, metadata) if query_facets else (0.0, [], [])
            )
            content_bonus = lexical_bonus + facet_score
            base = (1.0 if candidate.role == "primary" else 0.45) + 4.0 * candidate.rrf_score + 0.08 / candidate.local_rank
            if intent == "news":
                score = base + 0.45 * absolute + 1.35 * relative + content_bonus
            elif intent == "price":
                score = base + 0.15 * absolute + content_bonus
            elif intent in {"review", "after_update"}:
                score = base + 0.5 * absolute + 0.75 * relative + content_bonus
            else:
                score = base + content_bonus
            patch = self._latest_patches.get(str(metadata.get("game_key", "")))
            results.append(
                SearchResult(
                    document=document,
                    score=score,
                    dense_rank=candidate.dense_rank,
                    bm25_rank=candidate.bm25_rank,
                    rrf_score=candidate.rrf_score,
                    recency_score=absolute,
                    relative_recency_score=relative,
                    content_bonus=content_bonus,
                    facet_score=facet_score,
                    matched_facets=matched_facets,
                    conflicting_facets=conflicting_facets,
                    role=candidate.role,
                    intent=intent,
                    latest_patch_date=patch[0].isoformat() if patch else None,
                    latest_patch_title=patch[1] if patch else None,
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        for rank, result in enumerate(results[:k], start=1):
            result.rank = rank
        return results[:k]

    def _relative_recency(
        self, candidates: Iterable[_Candidate], half_life: int, intent: str
    ) -> dict[int, float]:
        if intent not in {"news", "review", "after_update"}:
            return {}
        grouped: dict[str, list[tuple[int, date]]] = defaultdict(list)
        for candidate in candidates:
            document = self.index.documents[candidate.document_index]
            if document.metadata.get("section") not in {"news", "review"}:
                continue
            parsed = parse_date(document.metadata.get("source_date"))
            if parsed:
                grouped[str(document.metadata.get("game_key", ""))].append((candidate.document_index, parsed))
        scores: dict[int, float] = {}
        for rows in grouped.values():
            latest = max(parsed for _, parsed in rows)
            for document_index, parsed in rows:
                scores[document_index] = math.exp(-max((latest - parsed).days, 0) / half_life)
        return scores

    def _content_bonus(self, document: Document, question: str, intent: str) -> float:
        metadata = document.metadata
        text = f"{metadata.get('item_title', '')} {document.page_content}".casefold()
        if intent == "gameplay":
            positive = ("gameplay", "combat", "turn-based", "real-time", "classes", "exploration", "co-op", "strategy", "choices")
            negative = ("streamer", "twitch", "nudity", "underwear")
            return max(-0.8, min(0.8, 0.08 * sum(word in text for word in positive) - 0.35 * sum(word in text for word in negative)))
        if intent == "news":
            news_type = str(metadata.get("news_type") or "")
            bonus = 0.2 if metadata.get("relevance_type") == "valid_update_or_patch" else 0.0
            if news_type in PATCH_NEWS_TYPES:
                bonus += 0.35
            if news_type in LOW_VALUE_NEWS_TYPES:
                bonus -= 0.55
            bonus += 0.35 * any(word in text for word in ("hotfix", "fix #", "patch notes", "now live"))
            bonus -= 0.25 * any(word in text for word in ("sale", "discount", "wishlist", "community update"))
            return max(-0.6, min(0.8, bonus))
        if intent == "price":
            bonus = 0.0
            if metadata.get("section") == "metadata" and metadata.get("price_available") is not None:
                bonus += 0.65
            if str(metadata.get("news_type") or "") == "sale_promo":
                bonus += 0.25
            if any(word in text for word in ("sale", "discount", "price", "deal", "할인", "가격", "세일")):
                bonus += 0.15
            return max(-0.3, min(0.9, bonus))
        if intent in {"review", "after_update"} and metadata.get("section") == "review":
            length_bonus = 0.2 if len(document.page_content) >= 500 else 0.1 if len(document.page_content) >= 250 else -0.15 if len(document.page_content) < 120 else 0.0
            try:
                quality = min(float(metadata.get("weighted_vote_score", 0)), 1.0) * 0.1
            except (TypeError, ValueError):
                quality = 0.0
            patch = self._latest_patches.get(str(metadata.get("game_key", "")))
            temporal = 0.0
            source = parse_date(metadata.get("source_date"))
            if intent == "after_update" and patch and source:
                temporal = 0.35 if source >= patch[0] else -0.5
            return length_bonus + quality + temporal
        return 0.0
