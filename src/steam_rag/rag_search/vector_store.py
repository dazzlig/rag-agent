from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from steam_rag.common.interfaces import Embedder
from steam_rag.common.models import Document


INDEX_VERSION = 1
CHROMA_COLLECTION = "steam_rag_chunks"


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
        if _uses_chroma(path):
            self._save_chroma(path)
            return
        self._save_json(path)

    def _save_json(self, path: Path) -> None:
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
        last_error: PermissionError | None = None
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.1 * (2**attempt))
        raise PermissionError(f"Could not replace locked vector index: {path}") from last_error

    def _save_chroma(self, path: Path) -> None:
        chromadb = _import_chromadb()
        if path.exists() and path.is_file():
            raise ValueError(f"Chroma index path must be a directory, got file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path))
        try:
            try:
                client.delete_collection(CHROMA_COLLECTION)
            except Exception:
                pass
            collection = client.create_collection(
                CHROMA_COLLECTION,
                metadata={
                    "version": INDEX_VERSION,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "embedding_model": self.embedding_model,
                },
            )
            if not self.documents:
                return
            for start in range(0, len(self.documents), 512):
                rows = list(enumerate(self.documents[start : start + 512], start=start))
                collection.add(
                    ids=[_chroma_id(order, document) for order, document in rows],
                    documents=[document.page_content for _, document in rows],
                    embeddings=self.embeddings[start : start + 512],
                    metadatas=[_chroma_metadata(order, document) for order, document in rows],
                )
        finally:
            _clear_chroma_cache()

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        if _uses_chroma(path):
            return cls._load_chroma(path)
        return cls._load_json(path)

    @classmethod
    def _load_json(cls, path: Path) -> "VectorIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != INDEX_VERSION:
            raise ValueError(f"Unsupported index version: {payload.get('version')}")
        return cls(
            documents=[Document.from_dict(value) for value in payload["documents"]],
            embeddings=[[float(item) for item in vector] for vector in payload["embeddings"]],
            embedding_model=str(payload["embedding_model"]),
        )

    @classmethod
    def _load_chroma(cls, path: Path) -> "VectorIndex":
        if not path.exists():
            raise FileNotFoundError(f"Chroma index not found: {path}")
        chromadb = _import_chromadb()
        client = chromadb.PersistentClient(path=str(path))
        try:
            collection = client.get_collection(CHROMA_COLLECTION)
            payload = collection.get(include=["documents", "metadatas", "embeddings"])
            embedding_model = str((collection.metadata or {}).get("embedding_model") or "")
        finally:
            _clear_chroma_cache()
        raw_documents = payload.get("documents") or []
        raw_metadatas = payload.get("metadatas") or []
        raw_embeddings = payload.get("embeddings")
        if raw_embeddings is None:
            raw_embeddings = []
        rows: list[tuple[int, Document, list[float]]] = []
        for fallback_order, (content, metadata, embedding) in enumerate(
            zip(raw_documents, raw_metadatas, raw_embeddings)
        ):
            metadata = dict(metadata or {})
            order = _safe_int(metadata.get("order"), fallback_order)
            document_metadata = _decode_document_metadata(metadata)
            rows.append(
                (
                    order,
                    Document(str(content or ""), document_metadata),
                    [float(value) for value in embedding],
                )
            )
        rows.sort(key=lambda row: row[0])
        return cls(
            documents=[document for _, document, _ in rows],
            embeddings=[embedding for _, _, embedding in rows],
            embedding_model=embedding_model,
        )


def delete_index(path: Path) -> None:
    if _uses_chroma(path):
        if path.exists():
            shutil.rmtree(path)
        return
    if path.exists():
        path.unlink()


def _uses_chroma(path: Path) -> bool:
    return path.suffix.lower() != ".json"


def _import_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Chroma vector store requires the `chromadb` package. "
            "Install project dependencies before using a directory index path."
        ) from exc
    return chromadb


def _clear_chroma_cache() -> None:
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def _chroma_id(order: int, document: Document) -> str:
    metadata = document.metadata
    chunk_id = str(metadata.get("chunk_id") or "chunk")
    appid = str(metadata.get("appid") or metadata.get("game_key") or "unknown")
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in f"{appid}_{chunk_id}")
    return f"{order:08d}_{safe[:180]}"


def _chroma_metadata(order: int, document: Document) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "order": order,
        "metadata_json": json.dumps(document.metadata, ensure_ascii=False),
    }
    for key in ("appid", "game_key", "game_name", "section", "item_title", "source_date", "chunk_id"):
        value = document.metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return metadata


def _decode_document_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    encoded = metadata.get("metadata_json")
    if isinstance(encoded, str):
        try:
            value = json.loads(encoded)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"order", "metadata_json"} and not key.startswith("chroma:")
    }


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
