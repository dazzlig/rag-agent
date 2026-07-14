from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from steam_rag.steam_collection.steam_client import (
    DEFAULT_LANGUAGE,
    PATCH_NEWS_TYPES,
    SteamAPIClient,
    normalize_news,
    normalize_review,
    utc_now,
)

ANALYSIS_SCHEMA_VERSION = "time-aware-analysis-v1"

FEATURE_PATTERNS = {
    "performance": (r"performance", r"optimization", r"fps", r"stutter", r"성능", r"최적화", r"프레임", r"끊김"),
    "stability": (r"crash", r"freeze", r"stability", r"크래시", r"튕김", r"멈춤", r"안정성"),
    "bugs": (r"\bbug", r"glitch", r"fix", r"버그", r"오류", r"수정"),
    "combat_balance": (r"combat", r"balance", r"weapon", r"damage", r"전투", r"밸런스", r"무기", r"피해"),
    "new_content": (r"new content", r"content update", r"dlc", r"expansion", r"quest", r"신규 콘텐츠", r"확장팩", r"퀘스트"),
    "multiplayer": (r"multiplayer", r"co-?op", r"network", r"server", r"멀티플레이", r"협동", r"네트워크", r"서버"),
    "controls": (r"control", r"controller", r"input", r"조작", r"컨트롤러", r"입력"),
    "ui_ux": (r"\bui\b", r"interface", r"menu", r"quality of life", r"인터페이스", r"메뉴", r"편의성"),
    "story": (r"story", r"narrative", r"character", r"스토리", r"서사", r"캐릭터"),
    "difficulty": (r"difficulty", r"hard", r"easy", r"난이도", r"어렵", r"쉬움"),
}


@dataclass(frozen=True, slots=True)
class PatchEvent:
    date: str
    title: str
    event_type: str
    importance: str
    affected_features: list[str]
    url: str | None = None
    gid: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewWindowStats:
    start_date: str
    end_date: str
    sample_size: int
    positive_count: int
    negative_count: int
    positive_ratio: float | None
    strengths: list[dict[str, Any]] = field(default_factory=list)
    weaknesses: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TimeAwareAnalysis:
    schema_version: str
    generated_at: str
    appid: int
    game_name: str
    language: str
    patch_event: PatchEvent
    before: ReviewWindowStats
    after: ReviewWindowStats
    positive_ratio_delta: float | None
    positive_ratio_delta_pp: float | None
    direction: str
    confidence_label: str
    confidence_score: float
    z_score: float | None
    topic_deltas: list[dict[str, Any]]
    pages_fetched: int
    pagination_reached_start: bool
    pagination_truncated: bool
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimeAnalysisRun:
    analysis: TimeAwareAnalysis
    json_path: Path
    markdown_path: Path
    index_path: Path


def structure_patch_events(news_items: Sequence[dict]) -> list[PatchEvent]:
    events: list[PatchEvent] = []
    importance_by_type = {
        "major_update": "high",
        "content_update": "high",
        "patch_note": "medium",
        "hotfix": "low",
    }
    for raw in news_items:
        item = raw if raw.get("news_type") else normalize_news(raw)
        event_type = str(item.get("news_type") or "")
        event_date = str(item.get("news_date") or "")
        if event_type not in PATCH_NEWS_TYPES or not _parse_date(event_date):
            continue
        text = f"{item.get('title', '')}\n{item.get('contents', '')}"
        title = str(item.get("title") or "Untitled update")
        title_has_event_signal = bool(
            re.search(
                r"patch|hotfix|update|version\s*\d|v\d+(?:\.\d+)*|dlc|expansion|"
                r"패치|핫픽스|업데이트|버전\s*\d|확장팩",
                title,
                flags=re.IGNORECASE,
            )
        )
        live_signal = bool(
            re.search(r"now live|is live|available now|배포|적용|업데이트 완료", text, flags=re.IGNORECASE)
        )
        if event_type in {"major_update", "content_update"} and not (
            title_has_event_signal or live_signal
        ):
            continue
        events.append(
            PatchEvent(
                date=event_date,
                title=title,
                event_type=event_type,
                importance=importance_by_type[event_type],
                affected_features=_extract_features(text),
                url=str(item.get("url")) if item.get("url") else None,
                gid=str(item.get("gid")) if item.get("gid") else None,
            )
        )
    events.sort(key=lambda event: event.date, reverse=True)
    return events


def select_patch_event(
    events: Sequence[PatchEvent], *, focus_features: Sequence[str] = ()
) -> PatchEvent:
    if not events:
        raise LookupError("분석할 패치/업데이트 이벤트를 찾지 못했습니다.")
    focus = set(focus_features)
    importance_score = {"low": 1, "medium": 2, "high": 3}

    def score(event: PatchEvent) -> tuple[int, int, str]:
        feature_match = len(focus & set(event.affected_features)) if focus else 0
        return feature_match, importance_score[event.importance], event.date
    if focus:
        focused = [event for event in events if focus & set(event.affected_features)]
        if focused:
            return max(focused, key=score)
    latest_date = max(_parse_date(event.date) for event in events)
    recent_cutoff = latest_date - timedelta(days=180)
    meaningful_recent = [
        event
        for event in events
        if event.importance in {"medium", "high"}
        and (_parse_date(event.date) or date.min) >= recent_cutoff
    ]
    if meaningful_recent:
        return max(meaningful_recent, key=lambda event: (importance_score[event.importance], event.date))
    return max(events, key=lambda event: event.date)


def analyze_patch_reviews(
    client: SteamAPIClient,
    *,
    appid: int,
    game_name: str,
    patch_event: PatchEvent,
    before_days: int = 30,
    after_days: int = 30,
    max_reviews: int = 5_000,
    max_pages: int = 100,
    language: str = DEFAULT_LANGUAGE,
    today: date | None = None,
) -> TimeAwareAnalysis:
    if before_days <= 0 or after_days <= 0:
        raise ValueError("before_days and after_days must be positive")
    patch_date = _parse_date(patch_event.date)
    if patch_date is None:
        raise ValueError(f"Invalid patch date: {patch_event.date}")
    reference_date = today or datetime.now(timezone.utc).date()
    if patch_date > reference_date:
        raise ValueError("patch date cannot be in the future")
    before_start = patch_date - timedelta(days=before_days)
    before_end = patch_date - timedelta(days=1)
    after_start = patch_date
    after_end = min(patch_date + timedelta(days=after_days - 1), reference_date)

    page_result = client.fetch_reviews_by_date_range(
        appid,
        start_date=before_start,
        end_date=after_end,
        max_reviews=max_reviews,
        max_pages=max_pages,
        language=language,
    )
    normalized = [normalize_review(review) for review in page_result.reviews]
    before_reviews = [
        review
        for review in normalized
        if (created := _parse_date(review.get("review_created_at")))
        and before_start <= created <= before_end
    ]
    after_reviews = [
        review
        for review in normalized
        if (created := _parse_date(review.get("review_created_at")))
        and after_start <= created <= after_end
    ]
    before_stats = _window_stats(before_reviews, before_start, before_end)
    after_stats = _window_stats(after_reviews, after_start, after_end)
    delta, delta_pp, direction, confidence_label, confidence_score, z_score = _change_metrics(
        before_stats,
        after_stats,
    )
    limitations: list[str] = []
    if page_result.truncated:
        limitations.append("리뷰 pagination이 설정된 최대 리뷰 수 또는 최대 페이지 수에서 중단됨")
    if before_stats.sample_size < 10 or after_stats.sample_size < 10:
        limitations.append("패치 전후 중 한 구간의 리뷰 표본이 10개 미만임")
    if after_end < patch_date + timedelta(days=after_days - 1):
        limitations.append("요청한 패치 후 관찰 기간이 아직 모두 지나지 않음")
    limitations.append("Steam 리뷰 언어 필터와 공개 리뷰 표본에 기반한 관찰 분석이며 인과관계를 확정하지 않음")
    return TimeAwareAnalysis(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        generated_at=utc_now(),
        appid=appid,
        game_name=game_name,
        language=language,
        patch_event=patch_event,
        before=before_stats,
        after=after_stats,
        positive_ratio_delta=delta,
        positive_ratio_delta_pp=delta_pp,
        direction=direction,
        confidence_label=confidence_label,
        confidence_score=confidence_score,
        z_score=z_score,
        topic_deltas=_topic_deltas(before_reviews, after_reviews),
        pages_fetched=page_result.pages_fetched,
        pagination_reached_start=page_result.reached_start,
        pagination_truncated=page_result.truncated,
        limitations=limitations,
    )


def save_time_analysis(path: Path, analysis: TimeAwareAnalysis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_time_analysis_and_index(
    *,
    client: SteamAPIClient,
    embedder: Any,
    game: Any,
    catalog_path: Path,
    docs_dir: Path,
    raw_dir: Path,
    profiles_dir: Path,
    index_path: Path,
    output_dir: Path,
    before_days: int = 30,
    after_days: int = 30,
    max_reviews: int = 5_000,
    max_pages: int = 100,
    news_count: int = 100,
    patch_date: str = "",
    focus_features: Sequence[str] = (),
    language: str = DEFAULT_LANGUAGE,
) -> TimeAnalysisRun:
    """Ensure the game corpus exists, analyze a patch, then re-index the enriched Markdown."""

    from steam_rag.rag_search.vector_store import VectorIndex, upsert_game_documents
    from steam_rag.steam_collection.corpus_manager import OnDemandCorpusManager
    from steam_rag.steam_collection.markdown_documents import chunk_documents, parse_markdown
    from steam_rag.steam_collection.steam_client import normalize_news, slugify

    update = OnDemandCorpusManager(
        client=client,
        catalog_path=catalog_path,
        docs_dir=docs_dir,
        raw_dir=raw_dir,
        profiles_dir=profiles_dir,
        index_path=index_path,
    ).ensure_game(game, embedder)
    normalized_news = [
        normalize_news(item)
        for item in client.fetch_news(update.game.appid, count=news_count)
    ]
    events = structure_patch_events(normalized_news)
    if patch_date:
        patch_event = next((event for event in events if event.date == patch_date), None)
        if patch_event is None:
            raise LookupError(f"No patch event found on {patch_date}")
    else:
        patch_event = select_patch_event(events, focus_features=focus_features)
    analysis = analyze_patch_reviews(
        client,
        appid=update.game.appid,
        game_name=update.game.name,
        patch_event=patch_event,
        before_days=before_days,
        after_days=after_days,
        max_reviews=max_reviews,
        max_pages=max_pages,
        language=language,
    )
    json_path = output_dir / (
        f"{slugify(update.game.name)}_{update.game.appid}_{patch_event.date}.json"
    )
    save_time_analysis(json_path, analysis)
    upsert_time_analysis_markdown(update.markdown_path, analysis)
    documents = chunk_documents(parse_markdown(update.markdown_path))
    current_index = VectorIndex.load(index_path)
    upsert_game_documents(
        current_index,
        documents,
        embedder,
        appid=update.game.appid,
    ).save(index_path)
    return TimeAnalysisRun(analysis, json_path, update.markdown_path, index_path)


def build_time_analysis_markdown(analysis: TimeAwareAnalysis) -> str:
    before = analysis.before
    after = analysis.after
    patch = analysis.patch_event
    lines = [
        "## Patch Impact Analysis",
        "",
        f"- analysis_schema_version: {analysis.schema_version}",
        f"- analysis_generated_at: {analysis.generated_at}",
        f"- appid: {analysis.appid}",
        f"- game_name: {analysis.game_name}",
        f"- language: {analysis.language}",
        f"- patch_date: {patch.date}",
        f"- patch_title: {patch.title}",
        f"- patch_event_type: {patch.event_type}",
        f"- patch_importance: {patch.importance}",
        f"- patch_affected_features: {patch.affected_features}",
        f"- url: {patch.url}",
        f"- before_start_date: {before.start_date}",
        f"- before_end_date: {before.end_date}",
        f"- before_sample_size: {before.sample_size}",
        f"- before_positive_ratio: {before.positive_ratio}",
        f"- after_start_date: {after.start_date}",
        f"- after_end_date: {after.end_date}",
        f"- after_sample_size: {after.sample_size}",
        f"- after_positive_ratio: {after.positive_ratio}",
        f"- positive_ratio_delta_pp: {analysis.positive_ratio_delta_pp}",
        f"- change_direction: {analysis.direction}",
        f"- confidence_label: {analysis.confidence_label}",
        f"- confidence_score: {analysis.confidence_score}",
        f"- pagination_pages_fetched: {analysis.pages_fetched}",
        f"- pagination_reached_start: {analysis.pagination_reached_start}",
        f"- pagination_truncated: {analysis.pagination_truncated}",
        "",
        f"패치 전 {before.sample_size}개 리뷰의 긍정률은 {_percent(before.positive_ratio)}였고, "
        f"패치 후 {after.sample_size}개 리뷰의 긍정률은 {_percent(after.positive_ratio)}였습니다. "
        f"변화는 {_signed_pp(analysis.positive_ratio_delta_pp)}이며 방향은 `{analysis.direction}`, "
        f"신뢰도는 `{analysis.confidence_label}`입니다.",
        "",
        "### Before strengths",
        _topic_markdown(before.strengths),
        "",
        "### Before weaknesses",
        _topic_markdown(before.weaknesses),
        "",
        "### After strengths",
        _topic_markdown(after.strengths),
        "",
        "### After weaknesses",
        _topic_markdown(after.weaknesses),
        "",
        "### Topic changes",
        _topic_markdown(analysis.topic_deltas),
        "",
        "### Limitations",
        *(f"- {item}" for item in analysis.limitations),
        "",
    ]
    return "\n".join(lines)


def upsert_time_analysis_markdown(path: Path, analysis: TimeAwareAnalysis) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = build_time_analysis_markdown(analysis).rstrip()
    pattern = re.compile(
        r"(?ms)^## Patch Impact Analysis\s*$.*?(?=^##\s|\Z)"
    )
    if pattern.search(text):
        updated = pattern.sub(section + "\n", text).rstrip() + "\n"
    else:
        updated = text.rstrip() + "\n\n" + section + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(path)


def _window_stats(
    reviews: Sequence[dict], start_date: date, end_date: date
) -> ReviewWindowStats:
    positives = [review for review in reviews if review.get("voted_up") is True]
    negatives = [review for review in reviews if review.get("voted_up") is False]
    rated_count = len(positives) + len(negatives)
    ratio = round(len(positives) / rated_count, 4) if rated_count else None
    return ReviewWindowStats(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        sample_size=rated_count,
        positive_count=len(positives),
        negative_count=len(negatives),
        positive_ratio=ratio,
        strengths=_topic_counts(positives),
        weaknesses=_topic_counts(negatives),
    )


def _change_metrics(
    before: ReviewWindowStats,
    after: ReviewWindowStats,
) -> tuple[float | None, float | None, str, str, float, float | None]:
    if before.positive_ratio is None or after.positive_ratio is None:
        return None, None, "insufficient_evidence", "insufficient", 0.0, None
    delta = after.positive_ratio - before.positive_ratio
    n1, n2 = before.sample_size, after.sample_size
    p1, p2 = before.positive_ratio, after.positive_ratio
    standard_error = math.sqrt(
        p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
    ) if n1 and n2 else 0.0
    z_score = delta / standard_error if standard_error else 0.0
    statistical_confidence = math.erf(abs(z_score) / math.sqrt(2))
    sample_factor = min(1.0, min(n1, n2) / 50)
    confidence_score = round(statistical_confidence * sample_factor, 4)
    if min(n1, n2) < 10:
        label = "insufficient"
    elif confidence_score >= 0.95 and min(n1, n2) >= 50:
        label = "high"
    elif confidence_score >= 0.75 and min(n1, n2) >= 20:
        label = "medium"
    else:
        label = "low"
    if label == "insufficient":
        direction = "insufficient_evidence"
    elif delta >= 0.05:
        direction = "improved"
    elif delta <= -0.05:
        direction = "declined"
    else:
        direction = "stable"
    return round(delta, 4), round(delta * 100, 2), direction, label, confidence_score, round(z_score, 4)


def _topic_counts(reviews: Sequence[dict], *, limit: int = 5) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for review in reviews:
        counts.update(_extract_features(str(review.get("review_text") or "")))
    return [
        {"topic": topic, "mentions": count}
        for topic, count in counts.most_common(limit)
    ]


def _topic_deltas(before: Sequence[dict], after: Sequence[dict]) -> list[dict[str, Any]]:
    before_counts = Counter(
        topic
        for review in before
        for topic in _extract_features(str(review.get("review_text") or ""))
    )
    after_counts = Counter(
        topic
        for review in after
        for topic in _extract_features(str(review.get("review_text") or ""))
    )
    before_total = len(before) or 1
    after_total = len(after) or 1
    rows = []
    for topic in sorted(set(before_counts) | set(after_counts)):
        before_rate = before_counts[topic] / before_total
        after_rate = after_counts[topic] / after_total
        rows.append(
            {
                "topic": topic,
                "before_mentions": before_counts[topic],
                "after_mentions": after_counts[topic],
                "mention_rate_delta_pp": round((after_rate - before_rate) * 100, 2),
            }
        )
    rows.sort(key=lambda row: abs(float(row["mention_rate_delta_pp"])), reverse=True)
    return rows[:8]


def _extract_features(text: str) -> list[str]:
    lowered = text.casefold()
    return sorted(
        feature
        for feature, patterns in FEATURE_PATTERNS.items()
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)
    )


def _parse_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError):
        return None


def _percent(value: float | None) -> str:
    return "표본 없음" if value is None else f"{value * 100:.1f}%"


def _signed_pp(value: float | None) -> str:
    return "계산 불가" if value is None else f"{value:+.2f}%p"


def _topic_markdown(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "- 충분한 주제 언급 없음"
    return "\n".join(
        "- " + ", ".join(f"{key}={value}" for key, value in row.items())
        for row in rows
    )
