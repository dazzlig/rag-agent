from __future__ import annotations

from typing import Protocol, Sequence

from .models import SearchResult


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker(Protocol):
    """Second-stage ranker for retrieved candidate chunks."""

    model_name: str

    def rerank(self, question: str, results: Sequence[SearchResult], *, top_n: int) -> list[SearchResult]: ...


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder reranker.

    The retriever still does broad hybrid retrieval first. This class then
    scores each (question, chunk) pair with a cross-encoder/BGE reranker and
    returns the strongest evidence chunks.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, *, max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model: object | None = None

    def rerank(self, question: str, results: Sequence[SearchResult], *, top_n: int) -> list[SearchResult]:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if not results:
            return []

        model = self._load_model()
        pairs = [(question, self._document_text(result)) for result in results]
        scores = model.predict(pairs)  # type: ignore[attr-defined]
        scored = []
        for result, score in zip(results, scores, strict=False):
            result.rerank_score = float(score)
            scored.append(result)

        scored.sort(key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"), reverse=True)
        reranked = scored[:top_n]
        for rank, result in enumerate(reranked, start=1):
            result.rank = rank
        return reranked

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "cross-encoder reranker를 사용하려면 sentence-transformers가 필요합니다. "
                "`poetry install` 또는 `pip install sentence-transformers` 후 다시 실행하세요."
            ) from exc
        self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def _document_text(self, result: SearchResult) -> str:
        metadata = result.document.metadata
        prefix = " ".join(
            str(value)
            for value in (
                metadata.get("game_name") or metadata.get("game_key"),
                metadata.get("section"),
                metadata.get("item_title"),
                metadata.get("source_date"),
            )
            if value
        )
        return f"{prefix}\n{result.document.page_content}".strip()
