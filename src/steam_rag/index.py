from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .interfaces import Embedder
from .models import Document


INDEX_VERSION = 1


@dataclass(slots=True)
class VectorIndex:
    documents: list[Document]
    embeddings: list[list[float]]
    embedding_model: str

    def __post_init__(self) -> None:
        if len(self.documents) != len(self.embeddings):
            raise ValueError("documents and embeddings must have the same length")
        dimensions = {len(vector) for vector in self.embeddings}
        if len(dimensions) > 1:
            raise ValueError("all embeddings must have the same dimension")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": self.embedding_model,
            "documents": [document.to_dict() for document in self.documents],
            "embeddings": self.embeddings,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != INDEX_VERSION:
            raise ValueError(f"Unsupported index version: {payload.get('version')}")
        return cls(
            documents=[Document.from_dict(value) for value in payload["documents"]],
            embeddings=[[float(item) for item in vector] for vector in payload["embeddings"]],
            embedding_model=str(payload["embedding_model"]),
        )


def build_index(
    documents: Iterable[Document], embedder: Embedder, *, batch_size: int = 64
) -> VectorIndex:
    docs = list(documents)
    if not docs:
        raise ValueError("cannot build an index without documents")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    vectors: list[list[float]] = []
    for start in range(0, len(docs), batch_size):
        texts = [document.page_content for document in docs[start : start + batch_size]]
        batch = embedder.embed_documents(texts)
        if len(batch) != len(texts):
            raise ValueError("embedder returned an unexpected number of vectors")
        vectors.extend(batch)
    return VectorIndex(docs, vectors, embedder.model_name)


def upsert_game_documents(
    index: VectorIndex,
    documents: Iterable[Document],
    embedder: Embedder,
    *,
    appid: int,
    batch_size: int = 64,
) -> VectorIndex:
    """Replace one game's chunks without re-embedding the rest of the corpus."""

    if index.embedding_model != embedder.model_name:
        raise ValueError(
            f"Index uses {index.embedding_model!r}, but embedder uses {embedder.model_name!r}"
        )
    incoming = list(documents)
    if not incoming:
        raise ValueError("cannot upsert a game without documents")
    kept = [
        (document, vector)
        for document, vector in zip(index.documents, index.embeddings)
        if str(document.metadata.get("appid")) != str(appid)
    ]
    incoming_index = build_index(incoming, embedder, batch_size=batch_size)
    return VectorIndex(
        documents=[document for document, _ in kept] + incoming_index.documents,
        embeddings=[vector for _, vector in kept] + incoming_index.embeddings,
        embedding_model=index.embedding_model,
    )
