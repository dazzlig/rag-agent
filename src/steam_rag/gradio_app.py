from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from .corpus import OnDemandCorpusManager, explicit_appid_from_question
from .index import VectorIndex, build_index
from .markdown_corpus import load_documents, parse_metadata
from .models import RAGAnswer, SearchResult
from .openai_adapter import OpenAIAnswerGenerator, OpenAIEmbedder, load_env_file
from .pipeline import RAGPipeline
from .rerank import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from .steam import SteamAPIClient


DEFAULT_DOCS = Path("data/docs_timeaware_playstyle")
DEFAULT_INDEX = Path("data/vectorstore/steam_rag_timeaware_playstyle.json")
DEFAULT_RAW = Path("data/raw/on_demand")
DEFAULT_CATALOG = Path("data/steam_catalog.json")
DEFAULT_EVAL = Path("data/eval")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_ANSWER_MODEL = "gpt-5-mini"
DEFAULT_EVAL_MODEL = "gpt-4o-mini"
RAGAS_METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
RAGAS_TEXT_COLUMNS = (
    "user_input",
    "question",
    "response",
    "answer",
    "retrieved_contexts",
    "contexts",
    "reference",
    "ground_truth",
)
RAGAS_TEXT_MAX_CHARS = 140


CSS = """
.gradio-container {
  max-width: 100% !important;
  min-height: 100vh !important;
  background: #f7f5f0 !important;
  color: #242424;
  font-family: "Malgun Gothic", "맑은 고딕", "Noto Sans KR", "Apple SD Gothic Neo", system-ui, sans-serif !important;
}
body, main, .main, gradio-app, .app, .wrap, .contain {
  background: #f7f5f0 !important;
  font-family: "Malgun Gothic", "맑은 고딕", "Noto Sans KR", "Apple SD Gothic Neo", system-ui, sans-serif !important;
}
*, label, button, textarea, input, .prose, .markdown, .checkbox, .form, .wrap {
  font-family: "Malgun Gothic", "맑은 고딕", "Noto Sans KR", "Apple SD Gothic Neo", system-ui, sans-serif !important;
}
#app-shell {
  max-width: 1480px;
  margin: 0 auto;
  padding: 18px 18px 28px 18px;
}
#app-title { padding: 4px 4px 14px 4px; }
#app-title h1 { margin-bottom: 0; color: #242424; letter-spacing: -0.03em; font-size: 28px; }
#left-panel, #main-panel, #dash-panel {
  border: 1px solid #e6e0d8;
  box-shadow: none;
}
#left-panel {
  background: #ebe7df;
  color: #242424;
  border-radius: 16px;
  padding: 14px;
}
#left-panel .block, #left-panel textarea, #left-panel input, #left-panel .table-wrap {
  background: #f7f5f0 !important;
  color: #242424 !important;
  border-color: #ded8cf !important;
}
#main-panel {
  background: #ffffff;
  border-radius: 16px;
  padding: 16px;
}
#dash-panel {
  background: #ffffff;
  border-radius: 16px;
  padding: 16px;
}
.block, .form, .panel, .tabs, .tabitem, .accordion {
  background: transparent !important;
}
.gr-button-primary {
  background: #2f2f2f !important;
  border-color: #2f2f2f !important;
}
.gr-button-secondary {
  background: #f7f5f0 !important;
  border-color: #ded8cf !important;
  color: #242424 !important;
}
.chatbot {
  background: #fbfaf7 !important;
  border-color: #e6e0d8 !important;
}
.dataframe, table {
  background: #ffffff !important;
}
.dataframe th {
  background: #f1eee8 !important;
}
#ragas-result table thead {
  position: sticky;
  top: 0;
  z-index: 5;
}
#ragas-result table th {
  white-space: nowrap !important;
}
#ragas-result table td {
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap !important;
}
.source-card {
  border: 1px solid #e6e0d8;
  border-radius: 12px;
  background: #fbfaf7;
  padding: 12px 14px;
  margin: 10px 0;
}
.source-card code { color: #333333; }
.source-card pre {
  background: #f1eee8;
  border-radius: 10px;
  padding: 10px;
  white-space: pre-wrap;
}
"""


@dataclass(frozen=True)
class AppPaths:
    docs_dir: Path = DEFAULT_DOCS
    index_path: Path = DEFAULT_INDEX
    raw_dir: Path = DEFAULT_RAW
    catalog_path: Path = DEFAULT_CATALOG
    eval_dir: Path = DEFAULT_EVAL


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _ui_error(action: str, exc: Exception) -> str:
    message = str(exc).strip() or "알 수 없는 오류"
    return f"**{action} 실패**\n\n{type(exc).__name__}: {message}"


def _read_metadata_from_markdown(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    marker = "## Metadata"
    if marker not in text:
        return {}
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    return parse_metadata(section)


def list_markdown_files(docs_dir: Path) -> list[dict[str, Any]]:
    if not docs_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(docs_dir.glob("*.md")):
        metadata = _read_metadata_from_markdown(path)
        rows.append(
            {
                "file": path.name,
                "game": metadata.get("name") or path.stem,
                "appid": metadata.get("appid", ""),
                "modified": path.stat().st_mtime,
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    return rows


def docs_table_rows(docs_dir: Path) -> list[list[Any]]:
    return [
        [item["file"], item["game"], item["appid"], item["size_kb"]]
        for item in list_markdown_files(docs_dir)
    ]


def markdown_preview(docs_dir: Path, filename: str | None) -> str:
    if not filename:
        return "선택된 Markdown 파일이 없습니다."
    path = docs_dir / filename
    if not path.exists():
        return f"`{filename}` 파일을 찾을 수 없습니다."
    metadata = _read_metadata_from_markdown(path)
    return (
        f"### {metadata.get('name') or path.stem}\n\n"
        f"- AppID: `{metadata.get('appid', '-')}`\n"
        f"- 파일: `{path.name}`\n"
        f"- 크기: `{path.stat().st_size / 1024:.1f} KB`"
    )


def index_status(index_path: Path, docs_dir: Path) -> tuple[str, dict[str, Any]]:
    docs_count = len(list(docs_dir.glob("*.md"))) if docs_dir.exists() else 0
    if not index_path.exists():
        payload = {"exists": False, "docs_count": docs_count}
        return (
            "### 인덱스 상태\n\n"
            f"- 저장된 MD 파일: `{docs_count}`개\n"
            f"- 인덱스: `{index_path}` 없음\n"
            "- 먼저 **인덱스 빌드/재빌드**를 실행하세요.",
            payload,
        )
    index = VectorIndex.load(index_path)
    sections: dict[str, int] = {}
    games: dict[str, int] = {}
    for document in index.documents:
        section = str(document.metadata.get("section", "unknown"))
        game = str(document.metadata.get("game_name") or document.metadata.get("game_key") or "unknown")
        sections[section] = sections.get(section, 0) + 1
        games[game] = games.get(game, 0) + 1
    payload = {
        "exists": True,
        "docs_count": docs_count,
        "chunks": len(index.documents),
        "embedding_model": index.embedding_model,
        "sections": sections,
        "games": games,
    }
    top_games = sorted(games.items(), key=lambda item: item[1], reverse=True)[:8]
    top_games_text = "\n".join(f"- {name}: `{count}` chunks" for name, count in top_games)
    section_text = "\n".join(f"- {name}: `{count}`" for name, count in sorted(sections.items()))
    return (
        "### 인덱스 상태\n\n"
        f"- 저장된 MD 파일: `{docs_count}`개\n"
        f"- 청크 수: `{len(index.documents)}`개\n"
        f"- 임베딩 모델: `{index.embedding_model}`\n\n"
        "#### 섹션별 청크\n"
        f"{section_text or '- 없음'}\n\n"
        "#### 게임별 상위 청크\n"
        f"{top_games_text or '- 없음'}",
        payload,
    )


def _load_pipeline(
    index_path: Path,
    embedding_model: str,
    *,
    answer_model: str | None = None,
    reranker_model: str = "",
    rerank_candidates: int = 24,
) -> RAGPipeline:
    embedder = OpenAIEmbedder(embedding_model)
    generator = OpenAIAnswerGenerator(answer_model) if answer_model else None
    reranker = CrossEncoderReranker(reranker_model) if reranker_model.strip() else None
    return RAGPipeline.from_path(
        index_path,
        embedder,
        generator,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
    )


def _ensure_question_if_needed(
    question: str,
    *,
    auto_collect: bool,
    paths: AppPaths,
    embedding_model: str,
    max_age_hours: float,
) -> str:
    explicit_appid = explicit_appid_from_question(question)
    should_collect = auto_collect or explicit_appid is not None
    if not should_collect:
        return ""
    embedder = OpenAIEmbedder(embedding_model)
    try:
        update = OnDemandCorpusManager(
            client=SteamAPIClient(),
            catalog_path=paths.catalog_path,
            docs_dir=paths.docs_dir,
            raw_dir=paths.raw_dir,
            index_path=paths.index_path,
            max_age=timedelta(hours=max_age_hours),
        ).ensure_question(question, embedder)
    except (LookupError, FileNotFoundError, RuntimeError, ValueError) as exc:
        if explicit_appid is not None:
            raise RuntimeError(
                f"appid `{explicit_appid}` 문서 생성/인덱싱에 실패했습니다. 기존 문서로 답변하지 않습니다. 원인: {exc}"
            ) from exc
        return f"자동 수집 건너뜀: {exc}"
    if update.collected and update.indexed:
        status = "새 MD 생성 후 벡터스토어 인덱싱"
    elif update.collected:
        status = "새 MD 생성"
    elif update.indexed:
        status = "기존 MD를 벡터스토어에 인덱싱"
    else:
        status = "기존 MD/벡터스토어 재사용"
    return (
        f"자동 수집: `{update.game.name}` / appid `{update.game.appid}` "
        f"/ 상태=`{status}` / 수집=`{update.collected}` / 인덱싱=`{update.indexed}` "
        f"/ 사유=`{update.reason}` / MD=`{update.markdown_path}`"
    )


def build_index_ui(
    docs_dir: str,
    index_path: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[str, str]:
    target = _as_path(index_path)
    docs_path = _as_path(docs_dir)
    try:
        documents = load_documents(
            docs_path,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
        )
        embedder = OpenAIEmbedder(embedding_model)
        index = build_index(documents, embedder, batch_size=int(batch_size))
        index.save(target)
    except Exception as exc:
        status, _ = index_status(target, docs_path)
        return _ui_error("인덱스 빌드", exc), status
    status, _ = index_status(target, docs_path)
    return f"빌드 완료: `{target}` / 청크 `{len(documents)}`개 / 모델 `{embedding_model}`", status


def _source_rows(results: Sequence[SearchResult]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for result in results:
        metadata = result.document.metadata
        rows.append(
            [
                result.rank,
                metadata.get("game_name") or metadata.get("game_key"),
                metadata.get("section"),
                metadata.get("item_title"),
                metadata.get("source_date") or "",
                round(_safe_float(result.score), 4),
                "" if result.rerank_score is None else round(_safe_float(result.rerank_score), 4),
                round(_safe_float(result.rrf_score), 4),
                round(_safe_float(result.recency_score), 4),
                round(_safe_float(result.relative_recency_score), 4),
                round(_safe_float(result.facet_score), 4),
            ]
        )
    return rows


def _source_cards(results: Sequence[SearchResult]) -> str:
    if not results:
        return "검색 결과가 없습니다."
    cards: list[str] = []
    for result in results:
        metadata = result.document.metadata
        content = result.document.page_content.strip().replace("```", "'''")
        if len(content) > 900:
            content = content[:900].rstrip() + "..."
        cards.append(
            "<div class='source-card'>"
            f"<h4>#{result.rank} {metadata.get('game_name') or metadata.get('game_key')} · {metadata.get('section')}</h4>"
            f"<p><code>{metadata.get('item_title') or '-'}</code> · date: <code>{metadata.get('source_date') or '-'}</code></p>"
            f"<p>score={result.score:.4f} · rrf={result.rrf_score:.4f} · recency={result.recency_score:.3f} "
            f"· relative={result.relative_recency_score:.3f} · facet={result.facet_score:.3f}</p>"
            f"<p>matched facets: <code>{', '.join(result.matched_facets) or '-'}</code></p>"
            f"<pre>{content}</pre>"
            "</div>"
        )
    return "\n".join(cards)


def _latest_state(question: str, answer: str, sources: Sequence[SearchResult]) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "contexts": [source.document.page_content for source in sources],
        "sources": [source.to_dict() for source in sources],
    }


def search_ui(
    question: str,
    docs_dir: str,
    index_path: str,
    raw_dir: str,
    catalog_path: str,
    embedding_model: str,
    top_k: int,
    auto_collect: bool,
    max_age_hours: float,
    retrieval_mode: str = "기본 RAG",
    answer_model: str = DEFAULT_ANSWER_MODEL,
    agentic_steps: int = 3,
    use_reranker: bool = False,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    rerank_candidates: int = 24,
) -> tuple[str, list[list[Any]], str, str, dict[str, Any], str]:
    if not question.strip():
        return "질문을 입력하세요.", [], "", "", {}, ""
    paths = AppPaths(
        docs_dir=_as_path(docs_dir),
        index_path=_as_path(index_path),
        raw_dir=_as_path(raw_dir),
        catalog_path=_as_path(catalog_path),
    )
    try:
        collect_note = _ensure_question_if_needed(
            question,
            auto_collect=auto_collect,
            paths=paths,
            embedding_model=embedding_model,
            max_age_hours=float(max_age_hours),
        )
        use_agentic = retrieval_mode.startswith("Agentic")
        pipeline = _load_pipeline(
            paths.index_path,
            embedding_model,
            answer_model=answer_model if use_agentic else None,
            reranker_model=reranker_model if use_reranker else "",
            rerank_candidates=int(rerank_candidates),
        )
        if use_agentic:
            results, metadata = pipeline.search_agentic(
                question,
                k=int(top_k),
                max_steps=int(agentic_steps),
                use_hyde=True,
            )
        else:
            metadata = {"strategy": "basic_rag", "reranker": reranker_model if use_reranker else ""}
            results = pipeline.search(question, k=int(top_k))
    except Exception as exc:
        return _ui_error("검색", exc), [], "", "", {}, ""
    raw_payload = {"metadata": metadata, "sources": [result.to_dict() for result in results]}
    raw = json.dumps(raw_payload, ensure_ascii=False, indent=2)
    summary = f"검색 완료: `{len(results)}`개 근거 반환 / 방식 `{metadata.get('strategy')}`"
    if metadata.get("steps"):
        step_lines = [
            f"- {step.get('step')}. {step.get('goal')} · 결과 {step.get('results')}개 · 충분성 `{step.get('sufficient')}`"
            for step in metadata.get("steps", [])
            if isinstance(step, dict)
        ]
        summary += "\n\n#### Agentic 실행 로그\n" + "\n".join(step_lines)
    if collect_note:
        summary += f"\n\n{collect_note}"
    state = _latest_state(question, "", results)
    state["metadata"] = metadata
    return summary, _source_rows(results), _source_cards(results), raw, state, ""


def ask_ui(
    question: str,
    docs_dir: str,
    index_path: str,
    raw_dir: str,
    catalog_path: str,
    embedding_model: str,
    answer_model: str,
    top_k: int,
    auto_collect: bool,
    max_age_hours: float,
    history: list[dict[str, str]] | None,
    retrieval_mode: str = "기본 RAG",
    agentic_steps: int = 3,
    use_reranker: bool = False,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    rerank_candidates: int = 24,
) -> tuple[list[dict[str, str]], str, list[list[Any]], str, str, dict[str, Any]]:
    if not question.strip():
        return list(history or []), "질문을 입력하세요.", [], "", "", {}
    paths = AppPaths(
        docs_dir=_as_path(docs_dir),
        index_path=_as_path(index_path),
        raw_dir=_as_path(raw_dir),
        catalog_path=_as_path(catalog_path),
    )
    try:
        collect_note = _ensure_question_if_needed(
            question,
            auto_collect=auto_collect,
            paths=paths,
            embedding_model=embedding_model,
            max_age_hours=float(max_age_hours),
        )
        pipeline = _load_pipeline(
            paths.index_path,
            embedding_model,
            answer_model=answer_model,
            reranker_model=reranker_model if use_reranker else "",
            rerank_candidates=int(rerank_candidates),
        )
        if retrieval_mode.startswith("Agentic"):
            rag_answer: RAGAnswer = pipeline.ask_agentic(
                question,
                k=int(top_k),
                max_steps=int(agentic_steps),
                use_hyde=True,
            )
        else:
            rag_answer = pipeline.ask(question, k=int(top_k))
    except Exception as exc:
        return list(history or []), _ui_error("답변 생성", exc), [], "", "", {}
    answer = rag_answer.answer
    if collect_note:
        answer = f"{answer}\n\n---\n{collect_note}"
    messages = list(history or [])
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    raw = json.dumps(rag_answer.to_dict(), ensure_ascii=False, indent=2)
    state = _latest_state(question, rag_answer.answer, rag_answer.sources)
    state["metadata"] = rag_answer.metadata
    return (
        messages,
        answer,
        _source_rows(rag_answer.sources),
        _source_cards(rag_answer.sources),
        raw,
        state,
    )


def _selected_metrics(metric_names: Iterable[str], has_reference: bool) -> list[Any]:
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError("ragas metrics를 import할 수 없습니다. ragas 설치 상태를 확인하세요.") from exc

    registry = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    selected: list[Any] = []
    for name in metric_names:
        if name in {"context_precision", "context_recall"} and not has_reference:
            continue
        metric = registry.get(name)
        if metric is not None:
            selected.append(metric)
    return selected or [faithfulness, answer_relevancy]


def _configure_ragas_metrics(metrics: Sequence[Any], eval_model: str, embedding_model: str) -> tuple[Any, Any]:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    raw_llm = ChatOpenAI(model=eval_model, temperature=0)
    raw_embeddings = OpenAIEmbeddings(model=embedding_model)
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        llm = LangchainLLMWrapper(raw_llm)
        embeddings = LangchainEmbeddingsWrapper(raw_embeddings)
    except Exception:
        llm = raw_llm
        embeddings = raw_embeddings
    for metric in metrics:
        if hasattr(metric, "llm"):
            metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
    return llm, embeddings


def _ragas_rows_to_dataset(rows: Sequence[dict[str, Any]]) -> Any:
    from datasets import Dataset

    normalized = []
    for row in rows:
        reference = str(row.get("reference") or row.get("ground_truth") or "").strip()
        normalized.append(
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row["contexts"],
                "reference": reference,
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": reference,
            }
        )
    return Dataset.from_list(normalized)


def _truncate_ragas_cell(value: Any, max_chars: int = RAGAS_TEXT_MAX_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        text = " | ".join(str(item) for item in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _result_to_rows(result: Any) -> Any:
    import pandas as pd

    if hasattr(result, "to_pandas"):
        frame = result.to_pandas().copy()
    elif isinstance(result, dict):
        frame = pd.DataFrame([result])
    else:
        try:
            frame = pd.DataFrame([dict(result)])
        except Exception:
            frame = pd.DataFrame([{"result": str(result)}])

    metric_columns = [name for name in RAGAS_METRIC_COLUMNS if name in frame.columns]
    text_columns: list[str] = []
    for aliases in (
        ("user_input", "question"),
        ("response", "answer"),
        ("retrieved_contexts", "contexts"),
        ("reference", "ground_truth"),
    ):
        selected = next((name for name in aliases if name in frame.columns), None)
        if selected:
            text_columns.append(selected)
    remaining_columns = [
        name
        for name in frame.columns
        if name not in metric_columns and name not in RAGAS_TEXT_COLUMNS
    ]
    frame = frame[metric_columns + text_columns + remaining_columns]

    for column in metric_columns:
        frame[column] = frame[column].map(
            lambda value: round(_safe_float(value), 4) if value not in (None, "") else ""
        )
    for column in text_columns:
        frame[column] = frame[column].map(_truncate_ragas_cell)
    return frame.fillna("").rename(
        columns={
            "user_input": "질문",
            "question": "질문",
            "response": "응답",
            "answer": "응답",
            "retrieved_contexts": "검색 컨텍스트",
            "contexts": "검색 컨텍스트",
            "reference": "기준 답변",
            "ground_truth": "기준 답변",
        }
    )


def run_ragas_rows(
    rows: Sequence[dict[str, Any]],
    metric_names: Sequence[str],
    eval_model: str,
    embedding_model: str,
) -> Any:
    if not rows:
        raise ValueError("RAGAS 평가할 row가 없습니다.")
    has_reference = any(str(row.get("reference") or row.get("ground_truth") or "").strip() for row in rows)
    metrics = _selected_metrics(metric_names, has_reference)
    llm, embeddings = _configure_ragas_metrics(metrics, eval_model, embedding_model)
    from ragas import evaluate

    dataset = _ragas_rows_to_dataset(rows)
    try:
        result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    except TypeError:
        result = evaluate(dataset, metrics=metrics)
    return _result_to_rows(result)


def ragas_current_ui(
    latest: dict[str, Any] | None,
    reference: str,
    metric_names: Sequence[str],
    eval_model: str,
    embedding_model: str,
) -> tuple[Any, str]:
    if not latest or not latest.get("answer"):
        return [], "먼저 RAG 답변을 생성하세요."
    row = {
        "question": latest["question"],
        "answer": latest["answer"],
        "contexts": latest["contexts"],
        "reference": reference.strip(),
    }
    try:
        result = run_ragas_rows([row], metric_names, eval_model, embedding_model)
    except Exception as exc:
        return [], _ui_error("RAGAS 평가", exc)
    return result, "현재 답변 RAGAS 평가 완료"


def _dataframe_records(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        records: list[dict[str, str]] = []
        for row in value.fillna("").to_dict(orient="records"):
            normalized = {str(key): "" if cell is None else str(cell) for key, cell in row.items()}
            if "질문" in normalized:
                normalized["question"] = normalized["질문"]
            if "기준 답변" in normalized:
                normalized["reference"] = normalized["기준 답변"]
            records.append(normalized)
        return records
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(row, dict) for row in value):
            records = []
            for row in value:
                normalized = {str(key): str(cell) for key, cell in row.items()}
                if "질문" in normalized:
                    normalized["question"] = normalized["질문"]
                if "기준 답변" in normalized:
                    normalized["reference"] = normalized["기준 답변"]
                records.append(normalized)
            return records
        headers = ["question", "reference"]
        return [
            {headers[index]: str(cell) for index, cell in enumerate(row[: len(headers)])}
            for row in value
            if row
        ]
    return []


def ragas_batch_ui(
    questions_frame: Any,
    docs_dir: str,
    index_path: str,
    embedding_model: str,
    answer_model: str,
    top_k: int,
    metric_names: Sequence[str],
    eval_model: str,
) -> tuple[Any, str]:
    records = [row for row in _dataframe_records(questions_frame) if row.get("question", "").strip()]
    if not records:
        return [], "배치 질문을 1개 이상 입력하세요."
    try:
        pipeline = _load_pipeline(_as_path(index_path), embedding_model, answer_model=answer_model)
        rows: list[dict[str, Any]] = []
        for record in records:
            question = record["question"].strip()
            rag_answer = pipeline.ask(question, k=int(top_k))
            rows.append(
                {
                    "question": question,
                    "answer": rag_answer.answer,
                    "contexts": [source.document.page_content for source in rag_answer.sources],
                    "reference": record.get("reference", "").strip(),
                }
            )
        result_rows = run_ragas_rows(rows, metric_names, eval_model, embedding_model)
    except Exception as exc:
        return [], _ui_error("배치 RAGAS 평가", exc)
    DEFAULT_EVAL.mkdir(parents=True, exist_ok=True)
    output_path = DEFAULT_EVAL / "gradio_ragas_last_input.json"
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_rows, f"배치 RAGAS 평가 완료. 입력 스냅샷 저장: `{output_path}`"


def refresh_docs_ui(docs_dir: str) -> tuple[list[list[Any]], str]:
    path = _as_path(docs_dir)
    rows = docs_table_rows(path)
    return rows, f"MD 파일 {len(rows)}개"


def refresh_status_ui(index_path: str, docs_dir: str) -> str:
    status, _ = index_status(_as_path(index_path), _as_path(docs_dir))
    return status


def create_app(default_paths: AppPaths = AppPaths()) -> Any:
    import gradio as gr

    initial_docs = docs_table_rows(default_paths.docs_dir)

    with gr.Blocks(title="Steam RAG Lab", fill_width=True, elem_id="app-shell") as demo:
        latest_state = gr.State({})
        embedding_model = gr.State(DEFAULT_EMBEDDING_MODEL)
        gr.Markdown(
            "# Steam RAG Lab",
            elem_id="app-title",
        )

        with gr.Row():
            with gr.Column(scale=2, min_width=280, elem_id="left-panel"):
                gr.Markdown("## 문서 보관함")
                docs_count = gr.Markdown(f"MD 파일 {len(initial_docs)}개")
                refresh_docs = gr.Button("MD 파일 새로고침", variant="secondary")
                md_table = gr.Dataframe(
                    headers=["파일", "게임", "AppID", "KB"],
                    value=initial_docs,
                    interactive=False,
                    label="저장된 MD 파일",
                    wrap=True,
                )

                gr.Markdown("## 인덱스")
                build_button = gr.Button("인덱스 빌드/재빌드", variant="primary")
                build_log = gr.Markdown()

                with gr.Accordion("경로 설정", open=False):
                    docs_dir = gr.Textbox(str(default_paths.docs_dir), label="MD 문서 폴더")
                    index_path = gr.Textbox(str(default_paths.index_path), label="인덱스 파일 경로")
                    raw_dir = gr.Textbox(str(default_paths.raw_dir), label="자동 수집 원본 저장 폴더")
                    catalog_path = gr.Textbox(str(default_paths.catalog_path), label="Steam 카탈로그 파일")

            with gr.Column(scale=6, min_width=560, elem_id="main-panel"):
                gr.Markdown("## RAG 테스트")
                chatbot = gr.Chatbot(label="대화", height=420)
                question = gr.Textbox(
                    label="질문",
                    placeholder="예: 3D 3인칭 실시간 전투 RPG 중 최근 평가가 좋은 게임은?",
                    lines=3,
                )
                with gr.Row():
                    answer_model = gr.Textbox(DEFAULT_ANSWER_MODEL, label="답변 모델", scale=2)
                    top_k = gr.Slider(1, 12, value=5, step=1, label="검색 개수 Top-K", scale=1)
                with gr.Row():
                    retrieval_mode = gr.Radio(
                        ["기본 RAG", "Agentic RAG + HyDE"],
                        value="기본 RAG",
                        label="검색 방식",
                        scale=2,
                    )
                    agentic_steps = gr.Slider(
                        1,
                        5,
                        value=3,
                        step=1,
                        label="Agentic 최대 검색 단계",
                        scale=1,
                    )
                with gr.Row():
                    use_reranker = gr.Checkbox(False, label="BGE Re-ranker 사용", scale=1)
                    reranker_model = gr.Textbox(DEFAULT_RERANKER_MODEL, label="Re-ranker 모델", scale=2)
                    rerank_candidates = gr.Slider(5, 50, value=24, step=1, label="Re-rank 후보 수", scale=1)
                with gr.Accordion("자동 수집", open=False):
                    auto_collect = gr.Checkbox(False, label="질문 게임 MD 없으면 Steam에서 자동 생성/인덱싱")
                    gr.Markdown("`appid:` 또는 Steam 상점 URL이 질문에 있으면 이 옵션이 꺼져 있어도 문서 생성/인덱싱을 먼저 시도합니다.")
                    max_age_hours = gr.Number(24, label="자동 수집 갱신 기준(시간)")
                with gr.Row():
                    search_button = gr.Button("근거만 검색", variant="secondary")
                    ask_button = gr.Button("RAG 답변 생성", variant="primary")
                    clear_button = gr.Button("대화 지우기")
                answer_markdown = gr.Markdown("답변이 여기에 표시됩니다.")
                with gr.Accordion("검색 근거", open=False):
                    source_table = gr.Dataframe(
                        headers=["순위", "게임", "섹션", "제목", "날짜", "점수", "Rerank", "RRF", "최신성", "상대 최신성", "Facet"],
                        interactive=False,
                        label="검색 점수",
                        wrap=True,
                    )
                    source_cards = gr.HTML()
                with gr.Accordion("원본 JSON", open=False):
                    raw_json = gr.Code(language="json")

            with gr.Column(scale=3, min_width=360, elem_id="dash-panel"):
                gr.Markdown("## 대시보드")
                status_markdown = gr.Markdown(index_status(default_paths.index_path, default_paths.docs_dir)[0])
                refresh_status = gr.Button("상태 새로고침", variant="secondary")

                gr.Markdown("## RAGAS 평가")
                reference = gr.Textbox(
                    label="기준 답변",
                    placeholder="정답 기준이 있으면 입력하세요. 없으면 faithfulness, answer_relevancy 중심으로 평가합니다.",
                    lines=4,
                )
                metric_names = gr.CheckboxGroup(
                    ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
                    value=["faithfulness", "answer_relevancy"],
                    label="평가 지표",
                )
                eval_model = gr.Textbox(DEFAULT_EVAL_MODEL, label="RAGAS 평가 모델")
                ragas_current_button = gr.Button("현재 답변 평가", variant="primary")
                ragas_log = gr.Markdown()
                ragas_result = gr.Dataframe(
                    label="RAGAS 결과",
                    interactive=False,
                    wrap=False,
                    max_chars=RAGAS_TEXT_MAX_CHARS,
                    max_height=320,
                    show_row_numbers=False,
                    pinned_columns=4,
                    elem_id="ragas-result",
                )

                with gr.Accordion("배치 RAGAS 질문", open=False):
                    batch_questions = gr.Dataframe(
                        headers=["질문", "기준 답변"],
                        value=[
                            ["Hollow Knight의 전투와 탐험 특징은?", ""],
                            ["Cyberpunk 2077은 업데이트 이후 평가가 어떤가?", ""],
                        ],
                        row_count=2,
                        column_count=2,
                        interactive=True,
                    )
                    ragas_batch_button = gr.Button("배치 RAGAS 실행", variant="secondary")

        refresh_docs.click(refresh_docs_ui, inputs=[docs_dir], outputs=[md_table, docs_count])
        refresh_status.click(refresh_status_ui, inputs=[index_path, docs_dir], outputs=[status_markdown])
        build_button.click(
            build_index_ui,
            inputs=[docs_dir, index_path, embedding_model],
            outputs=[build_log, status_markdown],
        )
        search_button.click(
            search_ui,
            inputs=[
                question,
                docs_dir,
                index_path,
                raw_dir,
                catalog_path,
                embedding_model,
                top_k,
                auto_collect,
                max_age_hours,
                retrieval_mode,
                answer_model,
                agentic_steps,
                use_reranker,
                reranker_model,
                rerank_candidates,
            ],
            outputs=[answer_markdown, source_table, source_cards, raw_json, latest_state, ragas_log],
        )
        ask_button.click(
            ask_ui,
            inputs=[
                question,
                docs_dir,
                index_path,
                raw_dir,
                catalog_path,
                embedding_model,
                answer_model,
                top_k,
                auto_collect,
                max_age_hours,
                chatbot,
                retrieval_mode,
                agentic_steps,
                use_reranker,
                reranker_model,
                rerank_candidates,
            ],
            outputs=[chatbot, answer_markdown, source_table, source_cards, raw_json, latest_state],
        )
        question.submit(
            ask_ui,
            inputs=[
                question,
                docs_dir,
                index_path,
                raw_dir,
                catalog_path,
                embedding_model,
                answer_model,
                top_k,
                auto_collect,
                max_age_hours,
                chatbot,
                retrieval_mode,
                agentic_steps,
                use_reranker,
                reranker_model,
                rerank_candidates,
            ],
            outputs=[chatbot, answer_markdown, source_table, source_cards, raw_json, latest_state],
        )
        clear_button.click(
            lambda: ([], "답변이 여기에 표시됩니다.", [], "", "", {}),
            outputs=[chatbot, answer_markdown, source_table, source_cards, raw_json, latest_state],
        )
        ragas_current_button.click(
            ragas_current_ui,
            inputs=[latest_state, reference, metric_names, eval_model, embedding_model],
            outputs=[ragas_result, ragas_log],
        )
        ragas_batch_button.click(
            ragas_batch_ui,
            inputs=[batch_questions, docs_dir, index_path, embedding_model, answer_model, top_k, metric_names, eval_model],
            outputs=[ragas_result, ragas_log],
        )

    return demo


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Steam RAG Gradio prototype")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_env_file(args.env_file)
    app = create_app(
        AppPaths(
            docs_dir=args.docs,
            index_path=args.index,
            raw_dir=args.raw,
            catalog_path=args.catalog,
            eval_dir=args.eval_dir,
        )
    )
    import gradio as gr

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        theme=gr.themes.Soft(primary_hue="neutral", neutral_hue="stone"),
        css=CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
