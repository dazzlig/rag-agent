from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .models import Document
from .playstyle import build_playstyle_metadata, coerce_list
from .steam import classify_news_type


SECTION_NAMES = {
    "metadata": "metadata",
    "store summary": "store_summary",
    "about the game": "about",
    "recent steam reviews": "review",
    "steam news": "news",
    "recent news": "news",
}


def _split_by_heading(text: str, level: int) -> list[tuple[str, str]]:
    marker = "#" * level + " "
    sections: list[tuple[str, str]] = []
    title = "preamble"
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(marker) and not line.startswith(marker + "#"):
            if lines:
                sections.append((title, "\n".join(lines).strip()))
            title = line[len(marker) :].strip()
            lines = [line]
        else:
            lines.append(line)
    if lines:
        sections.append((title, "\n".join(lines).strip()))
    return sections


def normalize_section(title: str) -> str:
    lowered = title.casefold().strip()
    for key, value in SECTION_NAMES.items():
        if key in lowered:
            return value
    if any(word in lowered for word in ("review", "리뷰")):
        return "review"
    if any(word in lowered for word in ("news", "update", "patch", "뉴스", "업데이트", "패치")):
        return "news"
    return "unknown"


def parse_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*-\s*([^:]+):\s*(.*)$", line)
        if match:
            metadata[match.group(1).strip()] = match.group(2).strip()
    return metadata


def _source_date(metadata: dict[str, str], section: str) -> str | None:
    if section == "review":
        return metadata.get("review_created_at") or metadata.get("review_updated_at")
    if section == "news":
        return metadata.get("news_date")
    if section == "metadata":
        return metadata.get("release_date")
    return None


def _item_index(title: str) -> int | None:
    match = re.search(r"\b(?:Review|News)\s+(\d+)", title, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_markdown(
    path: Path,
    *,
    min_item_chars: int = 30,
) -> list[Document]:
    """Parse one week-8 Steam Markdown file into section/item documents."""

    text = path.read_text(encoding="utf-8")
    sections = _split_by_heading(text, 2)
    playstyle_text = "\n\n".join(
        content
        for title, content in sections
        if normalize_section(title) in {"store_summary", "about", "review"}
    )
    common: dict[str, Any] = {"source_file": path.name, "game_key": path.stem}

    for title, content in sections:
        if normalize_section(title) == "metadata":
            raw = parse_metadata(content)
            genres = coerce_list(raw.get("genres"))
            categories = coerce_list(raw.get("categories"))
            common.update(
                {
                    "game_key": raw.get("game_key", path.stem),
                    "appid": raw.get("appid"),
                    "game_name": raw.get("name"),
                    "release_date": raw.get("release_date"),
                    "genres": genres,
                    "categories": categories,
                    "steam_tags": coerce_list(raw.get("steam_tags")),
                    "popular_tags_source": raw.get("popular_tags_source"),
                    "popular_tags_language": raw.get("popular_tags_language"),
                    "popular_tags_collected_at": raw.get("popular_tags_collected_at"),
                    "popular_tags_error": raw.get("popular_tags_error"),
                    "about_source": raw.get("about_source"),
                    "is_free": raw.get("is_free"),
                    "price_available": raw.get("price_available"),
                    "price_currency": raw.get("price_currency"),
                    "price_initial": raw.get("price_initial"),
                    "price_final": raw.get("price_final"),
                    "price_discount_percent": raw.get("price_discount_percent"),
                    "price_initial_formatted": raw.get("price_initial_formatted"),
                    "price_final_formatted": raw.get("price_final_formatted"),
                    "price_source": raw.get("price_source"),
                    "price_collected_at": raw.get("price_collected_at"),
                }
            )
            playstyle_metadata = build_playstyle_metadata(
                playstyle_text,
                genres=genres,
                categories=categories,
                steam_tags=raw.get("steam_tags") or raw.get("tags"),
                includes_reviews=any(
                    normalize_section(section_title) == "review" and "### Review" in section_content
                    for section_title, section_content in sections
                ),
            )
            common.update(playstyle_metadata)
            break

    documents: list[Document] = []
    for title, content in sections:
        section = normalize_section(title)
        if title == "preamble" or section == "unknown":
            continue
        items = _split_by_heading(content, 3) if section in {"review", "news"} else [(title, content)]
        for item_title, item_content in items:
            if section in {"review", "news"} and item_title == "preamble":
                continue
            if len(item_content.strip()) < min_item_chars:
                continue
            raw = parse_metadata(item_content)
            news_type = raw.get("news_type")
            if section == "news" and not news_type:
                news_type = classify_news_type({"title": item_title, "contents": item_content})
            metadata = {
                **common,
                "section": section,
                "section_title": title,
                "item_title": item_title,
                "item_index": _item_index(item_title),
                "source_date": _source_date(raw, section),
                "review_created_at": raw.get("review_created_at"),
                "review_updated_at": raw.get("review_updated_at"),
                "news_date": raw.get("news_date"),
                "sentiment": raw.get("sentiment"),
                "voted_up": raw.get("voted_up"),
                "weighted_vote_score": raw.get("weighted_vote_score"),
                "playtime_at_review": raw.get("playtime_at_review"),
                "news_type": news_type,
                "relevance_type": raw.get("relevance_type"),
                "url": raw.get("url"),
            }
            documents.append(
                Document(item_content, {key: value for key, value in metadata.items() if value is not None})
            )
    return documents


def _split_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    if len(text) <= chunk_size:
        yield text
        return
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        yield text[start:end].strip()
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)


def chunk_documents(
    documents: Iterable[Document], *, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be in [0, chunk_size)")
    chunks: list[Document] = []
    for document in documents:
        for local_index, content in enumerate(_split_text(document.page_content, chunk_size, chunk_overlap)):
            metadata = dict(document.metadata)
            item = metadata.get("item_index", "section")
            chunk_id = f"{metadata.get('game_key')}_{metadata.get('section')}_{item}_{local_index}"
            metadata.update({"chunk_id": chunk_id, "chunk_index": local_index})
            chunks.append(Document(content, metadata))
    return chunks


def load_documents(
    docs_dir: Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No Markdown files found in {docs_dir}")
    base_documents = [
        document
        for path in paths
        for document in parse_markdown(path)
    ]
    return chunk_documents(base_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
