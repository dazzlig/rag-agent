from __future__ import annotations

from typing import Protocol, Sequence

from .models import SearchResult


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class AnswerGenerator(Protocol):
    def generate(self, question: str, results: Sequence[SearchResult]) -> str: ...
