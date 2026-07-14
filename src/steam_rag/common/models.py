from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A retrievable text chunk and its source metadata."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"page_content": self.page_content, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Document":
        return cls(
            page_content=str(value["page_content"]),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(slots=True)
class SearchResult:
    document: Document
    score: float
    rank: int = 0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    recency_score: float = 0.0
    relative_recency_score: float = 0.0
    content_bonus: float = 0.0
    facet_score: float = 0.0
    rerank_score: float | None = None
    matched_facets: list[str] = field(default_factory=list)
    conflicting_facets: list[str] = field(default_factory=list)
    role: str = ""
    intent: str = "general"
    latest_patch_date: str | None = None
    latest_patch_title: str | None = None

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        value = {
            "rank": self.rank,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rrf_score": self.rrf_score,
            "recency_score": self.recency_score,
            "relative_recency_score": self.relative_recency_score,
            "content_bonus": self.content_bonus,
            "facet_score": self.facet_score,
            "rerank_score": self.rerank_score,
            "matched_facets": self.matched_facets,
            "conflicting_facets": self.conflicting_facets,
            "role": self.role,
            "intent": self.intent,
            "latest_patch_date": self.latest_patch_date,
            "latest_patch_title": self.latest_patch_title,
            "metadata": self.document.metadata,
        }
        if include_content:
            value["page_content"] = self.document.page_content
        return value


@dataclass(slots=True)
class RAGAnswer:
    question: str
    answer: str
    sources: list[SearchResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "question": self.question,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.metadata:
            value["metadata"] = self.metadata
        return value
