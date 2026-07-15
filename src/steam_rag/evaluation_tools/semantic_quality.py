from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from steam_rag.common.telemetry import TelemetryCollector, telemetry_session, tracked_openai_call
from steam_rag.external_apis.openai_client import OpenAIEmbedder


QUALITY_DIMENSIONS = (
    "target_entity_correctness",
    "requirement_satisfaction",
    "evidence_grounding",
    "temporal_correctness",
    "conversation_continuity",
    "readability",
)
RAGAS_METRICS = ("faithfulness", "answer_relevancy")
DEFAULT_RAGAS_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"
DEFAULT_RAGAS_MAX_COMPLETION_TOKENS = 2400
DEFAULT_JUDGE_MAX_COMPLETION_TOKENS = 2400


class QualityJudgeDecision(BaseModel):
    """Structured semantic-quality decision for one service turn."""

    verdict: Literal["pass", "partial", "fail"]
    target_entity_correctness: int | None = Field(default=None, ge=0, le=4)
    requirement_satisfaction: int | None = Field(default=None, ge=0, le=4)
    evidence_grounding: int | None = Field(default=None, ge=0, le=4)
    temporal_correctness: int | None = Field(default=None, ge=0, le=4)
    conversation_continuity: int | None = Field(default=None, ge=0, le=4)
    readability: int | None = Field(default=None, ge=0, le=4)
    critical_errors: list[str] = Field(default_factory=list, max_length=5)
    rationale: str = Field(max_length=700)
    confidence: float = Field(ge=0.0, le=1.0)


class QualityJudge(Protocol):
    model: str

    def judge(self, row: dict[str, Any]) -> QualityJudgeDecision: ...


class RagasEvaluator(Protocol):
    model: str
    embedding_model: str

    def evaluate(self, row: dict[str, Any]) -> dict[str, float | None]: ...


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
        rows.append(value)
    return rows


def _row_id(row: dict[str, Any]) -> str:
    case_id = str(row.get("case_id") or "").strip()
    turn_id = str(row.get("turn_id") or "").strip()
    if not case_id or not turn_id:
        raise ValueError("Each evaluation row requires case_id and turn_id")
    return f"{case_id}/{turn_id}"


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_contexts(row: dict[str, Any], *, max_context_chars: int = 2400) -> list[str]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    raw_sources = (
        payload.get("evidence_contexts")
        if isinstance(payload.get("evidence_contexts"), list)
        else payload.get("sources")
    ) if isinstance(payload, dict) else []
    contexts: list[str] = []
    for source in raw_sources if isinstance(raw_sources, list) else []:
        if not isinstance(source, dict):
            continue
        content = str(
            source.get("content")
            or source.get("page_content")
            or source.get("snippet")
            or ""
        ).strip()
        if not content:
            continue
        header = (
            f"game={source.get('game') or source.get('game_name') or ''}; "
            f"appid={source.get('appid') or ''}; section={source.get('section') or ''}; "
            f"source_type={source.get('source_type') or ''}; date={source.get('date') or ''}"
        )
        contexts.append(f"{header}\n{content[:max_context_chars]}")
    return contexts[:8]


def build_ragas_inputs(details: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select answered turns that include persisted evidence contexts."""

    selected: list[dict[str, Any]] = []
    for row in details:
        answer = str(row.get("answer") or "").strip()
        contexts = _source_contexts(row)
        if str(row.get("error") or "").strip() or not answer or not contexts:
            continue
        selected.append(
            {
                "id": _row_id(row),
                "case_id": str(row.get("case_id") or ""),
                "turn_id": str(row.get("turn_id") or ""),
                "category": str(row.get("category") or ""),
                "mode": str(row.get("mode") or ""),
                "question": str(row.get("question") or ""),
                "answer": answer,
                "contexts": contexts,
                "reference": "",
                "deterministic_contract_pass": _nested_value(row, "metrics", "contract_pass"),
            }
        )
    return selected


def build_judge_inputs(details: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build complete judge inputs while preserving prior turns for continuity scoring."""

    history_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    selected: list[dict[str, Any]] = []
    for row in details:
        case_id = str(row.get("case_id") or "")
        answer = str(row.get("answer") or "").strip()
        error = str(row.get("error") or "").strip()
        automatic_failure = error or ("empty_answer" if not answer else "")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        selected.append(
            {
                "id": _row_id(row),
                "case_id": case_id,
                "turn_id": str(row.get("turn_id") or ""),
                "turn_number": int(row.get("turn_number") or 0),
                "category": str(row.get("category") or ""),
                "title": str(row.get("title") or ""),
                "mode": str(row.get("mode") or ""),
                "question": str(row.get("question") or ""),
                "answer": answer,
                "expected": row.get("expected") if isinstance(row.get("expected"), dict) else {},
                "forbidden": row.get("forbidden") if isinstance(row.get("forbidden"), dict) else {},
                "observed_appids": list(row.get("observed_appids") or []),
                "deterministic_metrics": (
                    row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                ),
                "conversation_history": list(history_by_case[case_id][-6:]),
                "contexts": _source_contexts(row),
                "candidate_evidence": _compact_candidate_evidence(payload),
                "evidence_coverage": (
                    payload.get("evidence_coverage")
                    if isinstance(payload.get("evidence_coverage"), dict)
                    else {}
                ),
                "requires_llm": not bool(automatic_failure),
                "automatic_failure": automatic_failure,
            }
        )
        history_by_case[case_id].append(
            {"question": str(row.get("question") or ""), "answer": answer[:1800]}
        )
    return selected


def prepare_semantic_evaluation(
    details_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    details = load_jsonl(details_path)
    ids = [_row_id(row) for row in details]
    if len(ids) != len(set(ids)):
        raise ValueError("Conversation details contain duplicate case/turn ids")
    ragas_rows = build_ragas_inputs(details)
    judge_rows = build_judge_inputs(details)
    ragas_ids = {row["id"] for row in ragas_rows}
    source_metadata_only = [
        _row_id(row)
        for row in details
        if not str(row.get("error") or "").strip()
        and str(row.get("answer") or "").strip()
        and _has_source_metadata(row)
        and _row_id(row) not in ragas_ids
    ]
    ragas_path = output_dir / "ragas-input.jsonl"
    judge_path = output_dir / "judge-input.jsonl"
    selection_path = output_dir / "selection-summary.json"
    _write_jsonl(ragas_path, ragas_rows)
    _write_jsonl(judge_path, judge_rows)
    summary = {
        "source_details": str(details_path),
        "turn_count": len(details),
        "ragas": {
            "selected_count": len(ragas_rows),
            "ids": [row["id"] for row in ragas_rows],
            "selection_rule": "no error + non-empty answer + one or more persisted source contexts",
            "metrics": ["faithfulness", "answer_relevancy"],
            "source_metadata_only_excluded_count": len(source_metadata_only),
            "source_metadata_only_excluded_ids": source_metadata_only,
            "exclusion_reason": (
                "A source title or URL without persisted evidence text is not a RAGAS context"
            ),
        },
        "llm_judge": {
            "input_count": len(judge_rows),
            "requires_llm_count": sum(bool(row["requires_llm"]) for row in judge_rows),
            "automatic_failure_count": sum(not bool(row["requires_llm"]) for row in judge_rows),
            "automatic_failure_ids": [
                row["id"] for row in judge_rows if not bool(row["requires_llm"])
            ],
            "dimensions": list(QUALITY_DIMENSIONS),
        },
        "outputs": {
            "ragas_input": str(ragas_path),
            "judge_input": str(judge_path),
            "selection_summary": str(selection_path),
        },
        "external_calls": False,
    }
    _write_json(selection_path, summary)
    return summary


class OpenAIQualityJudge:
    """Evidence-aware judge with a bounded structured response."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, client: Any | None = None) -> None:
        self.model = model
        if client is not None:
            self._client = client
            return
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
        from openai import OpenAI

        self._client = OpenAI()

    def judge(self, row: dict[str, Any]) -> QualityJudgeDecision:
        input_payload = {
            key: row.get(key)
            for key in (
                "id",
                "category",
                "mode",
                "question",
                "answer",
                "expected",
                "forbidden",
                "observed_appids",
                "conversation_history",
                "contexts",
                "candidate_evidence",
                "evidence_coverage",
            )
        }
        request_options: dict[str, Any] = {
            "max_completion_tokens": DEFAULT_JUDGE_MAX_COMPLETION_TOKENS,
        }
        if _uses_fixed_temperature_reasoning_model(self.model):
            # GPT-5 can otherwise spend the entire completion budget on hidden
            # reasoning and return no JSON for Structured Outputs.
            request_options.update(reasoning_effort="minimal", verbosity="low")
        completion = tracked_openai_call(
            model=self.model,
            operation="chat",
            call=lambda: self._client.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Steam 추천·분석 RAG의 품질 평가자다. 답변을 새로 작성하지 말고 제공된 "
                            "질문, 대화 이력, expected/forbidden 계약, AppID, 검색 근거만 평가한다. "
                            "각 차원은 0=완전 실패, 1=심각한 결함, 2=부분 충족, 3=대체로 충족, "
                            "4=완전 충족이다. 적용할 수 없는 차원은 null이다. 한글명·영문명·공식 "
                            "별칭이 다르더라도 AppID와 의미가 같으면 감점하지 않는다. 반대로 잘못된 "
                            "게임을 고른 뒤 그 게임 근거와 답변이 일치해도 target_entity와 overall은 "
                            "실패다. 추천은 모든 주요 후보가 장르·인원·가격·출시·제외 조건을 실제로 "
                            "충족하는지 본다. 비교는 두 대상을 모두 다뤄야 한다. 업데이트 변화는 전후 "
                            "날짜·긍정률·표본 수가 없으면 개선됐다고 단정할 수 없다. 인용 번호가 있어도 "
                            "근거가 주장을 지지하지 않으면 evidence 점수를 낮춘다. 후속 질문은 이전 대상과 "
                            "교정 조건을 유지해야 한다. 첫 결론, 중복, 과도한 내부 용어, 읽기 쉬운 길이를 "
                            "readability에서 평가한다. 치명적 대상 오류, 조건 역전, 근거 없는 최신성 단정, "
                            "답변 거부는 critical_errors에 짧게 적는다. verdict pass는 치명적 오류가 없고 "
                            "핵심 차원이 모두 3점 이상일 때만 사용한다. rationale은 한국어 4문장 이내다."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(input_payload, ensure_ascii=False),
                    },
                ],
                response_format=QualityJudgeDecision,
                **request_options,
            ),
        )
        parsed = completion.choices[0].message.parsed
        if not isinstance(parsed, QualityJudgeDecision):
            raise RuntimeError("LLM Judge returned no parsed decision")
        return parsed


class _RagasTelemetryCallback(BaseCallbackHandler):
    """Collect LangChain chat usage into the project's telemetry schema."""

    def __init__(self, collector: TelemetryCollector, model: str) -> None:
        self.collector = collector
        self.model = model
        self.started: dict[str, float] = {}

    def on_llm_start(self, _serialized: Any, _prompts: Any, *, run_id: Any, **_kwargs: Any) -> None:
        self.started[str(run_id)] = time.perf_counter()

    def on_chat_model_start(
        self,
        _serialized: Any,
        _messages: Any,
        *,
        run_id: Any,
        **_kwargs: Any,
    ) -> None:
        self.started[str(run_id)] = time.perf_counter()

    def on_llm_end(self, response: Any, *, run_id: Any, **_kwargs: Any) -> None:
        key = str(run_id)
        usage = _langchain_usage(response)
        self.collector.record_openai(
            model=self.model,
            operation="chat",
            response={"usage": usage},
            latency_ms=(time.perf_counter() - self.started.pop(key, time.perf_counter())) * 1000,
        )

    def on_llm_error(self, _error: BaseException, *, run_id: Any, **_kwargs: Any) -> None:
        key = str(run_id)
        self.collector.record_openai(
            model=self.model,
            operation="chat",
            error=True,
            latency_ms=(time.perf_counter() - self.started.pop(key, time.perf_counter())) * 1000,
        )


class OpenAIRagasEvaluator:
    """Run reference-free RAGAS metrics for one persisted turn at a time."""

    def __init__(
        self,
        model: str = DEFAULT_RAGAS_MODEL,
        embedding_model: str = "text-embedding-3-small",
        max_completion_tokens: int = DEFAULT_RAGAS_MAX_COMPLETION_TOKENS,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.max_completion_tokens = max(1, int(max_completion_tokens))

    def evaluate(self, row: dict[str, Any]) -> dict[str, float | None]:
        from datasets import Dataset
        from langchain_core.embeddings import Embeddings
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, faithfulness

        collector = _active_collector()
        callback = _RagasTelemetryCallback(collector, self.model)

        class TrackedEmbeddings(Embeddings):
            def __init__(self, model: str) -> None:
                self.embedder = OpenAIEmbedder(model)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.embedder.embed_documents(texts)

            def embed_query(self, text: str) -> list[float]:
                return self.embedder.embed_query(text)

        raw_llm = ChatOpenAI(
            model=self.model,
            callbacks=[callback],
            max_completion_tokens=self.max_completion_tokens,
        )
        raw_embeddings = TrackedEmbeddings(self.embedding_model)
        llm = LangchainLLMWrapper(
            raw_llm,
            # RAGAS 0.4.x overrides temperature with 0.01/0.3. GPT-5 only
            # accepts its default temperature, so those overrides must not be
            # forwarded to ChatOpenAI.
            bypass_temperature=_uses_fixed_temperature_reasoning_model(self.model),
        )
        embeddings = LangchainEmbeddingsWrapper(raw_embeddings)
        metrics = [faithfulness, answer_relevancy]
        for metric in metrics:
            if hasattr(metric, "llm"):
                metric.llm = llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = embeddings
        dataset = Dataset.from_list(
            [
                {
                    "user_input": row["question"],
                    "response": row["answer"],
                    "retrieved_contexts": list(row["contexts"]),
                    "reference": "",
                    "question": row["question"],
                    "answer": row["answer"],
                    "contexts": list(row["contexts"]),
                    "ground_truth": "",
                }
            ]
        )
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=True,
            show_progress=False,
            batch_size=1,
        )
        frame = result.to_pandas()
        first = frame.iloc[0].to_dict()
        metrics_result = {
            "faithfulness": _optional_float(first.get("faithfulness")),
            "answer_relevancy": _optional_float(first.get("answer_relevancy")),
        }
        if not _has_valid_ragas_metrics(metrics_result):
            raise RuntimeError(
                "RAGAS returned no valid faithfulness/answer_relevancy values"
            )
        return metrics_result


def run_llm_judge(
    rows: Sequence[dict[str, Any]],
    *,
    judge: QualityJudge,
    output_path: Path,
    summary_path: Path,
    limit: int = 0,
    resume: bool = True,
) -> dict[str, Any]:
    existing = load_jsonl(output_path) if resume and output_path.exists() else []
    by_id = {str(row.get("id")): row for row in existing}
    completed = {
        row_id
        for row_id, row in by_id.items()
        if str(row.get("status")) in {"complete", "automatic_failure"}
    }
    new_calls = 0
    for row in rows:
        row_id = str(row["id"])
        if row_id in completed:
            continue
        if not bool(row.get("requires_llm")):
            by_id[row_id] = _automatic_judge_failure(row, judge.model)
            _write_ordered_results(output_path, rows, by_id)
            continue
        if limit > 0 and new_calls >= limit:
            continue
        new_calls += 1
        started = time.perf_counter()
        with telemetry_session() as collector:
            try:
                decision = judge.judge(row)
                result = _decision_result(row, decision, judge.model)
                result["status"] = "complete"
            except Exception as exc:
                result = _evaluation_error_result(row, judge.model, exc)
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            result["telemetry"] = collector.snapshot()
        by_id[row_id] = result
        _write_ordered_results(output_path, rows, by_id)
    results = _ordered_results(rows, by_id)
    summary = summarize_judge_results(results, selected_count=len(rows), model=judge.model)
    _write_json(summary_path, summary)
    return summary


def run_ragas_evaluation(
    rows: Sequence[dict[str, Any]],
    *,
    evaluator: RagasEvaluator,
    output_path: Path,
    summary_path: Path,
    limit: int = 0,
    resume: bool = True,
) -> dict[str, Any]:
    existing = load_jsonl(output_path) if resume and output_path.exists() else []
    by_id = {str(row.get("id")): row for row in existing}
    completed = {
        row_id
        for row_id, row in by_id.items()
        if _is_valid_ragas_result(
            row,
            model=evaluator.model,
            embedding_model=evaluator.embedding_model,
        )
    }
    new_calls = 0
    for row in rows:
        row_id = str(row["id"])
        if row_id in completed:
            continue
        if limit > 0 and new_calls >= limit:
            continue
        new_calls += 1
        started = time.perf_counter()
        with telemetry_session() as collector:
            try:
                metrics = evaluator.evaluate(row)
                if not _has_valid_ragas_metrics(metrics):
                    raise RuntimeError(
                        "RAGAS returned no valid faithfulness/answer_relevancy values"
                    )
                result = {
                    "id": row_id,
                    "case_id": row.get("case_id"),
                    "turn_id": row.get("turn_id"),
                    "category": row.get("category"),
                    "model": evaluator.model,
                    "embedding_model": evaluator.embedding_model,
                    "status": "complete",
                    "metrics": metrics,
                }
            except Exception as exc:
                result = {
                    "id": row_id,
                    "case_id": row.get("case_id"),
                    "turn_id": row.get("turn_id"),
                    "category": row.get("category"),
                    "status": "error",
                    "evaluation_error": f"{type(exc).__name__}: {exc}",
                    "metrics": {},
                }
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            result["telemetry"] = collector.snapshot()
        by_id[row_id] = result
        _write_ordered_results(output_path, rows, by_id)
    results = _ordered_results(rows, by_id)
    summary = summarize_ragas_results(
        results,
        selected_count=len(rows),
        model=evaluator.model,
        embedding_model=evaluator.embedding_model,
    )
    _write_json(summary_path, summary)
    return summary


def summarize_judge_results(
    results: Sequence[dict[str, Any]],
    *,
    selected_count: int,
    model: str,
) -> dict[str, Any]:
    finished = [row for row in results if row.get("status") in {"complete", "automatic_failure"}]
    verdicts = {name: sum(row.get("verdict") == name for row in finished) for name in ("pass", "partial", "fail")}
    dimensions = {
        name: _mean(
            [
                _optional_float(row.get("scores", {}).get(name))
                for row in finished
                if isinstance(row.get("scores"), dict)
            ]
        )
        for name in QUALITY_DIMENSIONS
    }
    return {
        "metric_family": "llm_judge_semantic_quality",
        "model": model,
        "selected_count": selected_count,
        "completed_count": len(finished),
        "pending_count": max(0, selected_count - len(finished)),
        "evaluation_error_count": sum(row.get("status") == "error" for row in results),
        "automatic_failure_count": sum(row.get("status") == "automatic_failure" for row in results),
        "verdicts": verdicts,
        "pass_rate": round(verdicts["pass"] / len(finished), 6) if finished else None,
        "overall_quality": _mean([_optional_float(row.get("overall_quality")) for row in finished]),
        "dimension_scores_0_to_4": dimensions,
        "operations": _aggregate_result_telemetry(results),
    }


def summarize_ragas_results(
    results: Sequence[dict[str, Any]],
    *,
    selected_count: int,
    model: str,
    embedding_model: str,
) -> dict[str, Any]:
    finished = [
        row
        for row in results
        if _is_valid_ragas_result(
            row,
            model=model,
            embedding_model=embedding_model,
        )
    ]
    invalid = [
        row
        for row in results
        if row.get("status") == "error"
        or (row.get("status") == "complete" and not _is_valid_ragas_result(row))
    ]
    return {
        "metric_family": "ragas_reference_free",
        "model": model,
        "embedding_model": embedding_model,
        "selected_count": selected_count,
        "completed_count": len(finished),
        "pending_count": max(0, selected_count - len(finished)),
        "evaluation_error_count": len(invalid),
        "metrics": {
            name: _mean(
                [
                    _optional_float(row.get("metrics", {}).get(name))
                    for row in finished
                    if isinstance(row.get("metrics"), dict)
                ]
            )
            for name in RAGAS_METRICS
        },
        "operations": _aggregate_result_telemetry(results),
    }


def _decision_result(
    row: dict[str, Any],
    decision: QualityJudgeDecision,
    model: str,
) -> dict[str, Any]:
    payload = decision.model_dump()
    scores = {name: payload.pop(name) for name in QUALITY_DIMENSIONS}
    applicable = [float(value) for value in scores.values() if value is not None]
    return {
        "id": row["id"],
        "case_id": row.get("case_id"),
        "turn_id": row.get("turn_id"),
        "category": row.get("category"),
        "model": model,
        "verdict": payload["verdict"],
        "overall_quality": round(sum(applicable) / (len(applicable) * 4), 6) if applicable else 0.0,
        "scores": scores,
        "critical_errors": payload["critical_errors"],
        "rationale": payload["rationale"],
        "confidence": payload["confidence"],
        "deterministic_contract_pass": _nested_value(
            row, "deterministic_metrics", "contract_pass"
        ),
    }


def _automatic_judge_failure(row: dict[str, Any], model: str) -> dict[str, Any]:
    reason = str(row.get("automatic_failure") or "no_answer")
    return {
        "id": row["id"],
        "case_id": row.get("case_id"),
        "turn_id": row.get("turn_id"),
        "category": row.get("category"),
        "model": model,
        "status": "automatic_failure",
        "verdict": "fail",
        "overall_quality": 0.0,
        "scores": {name: None for name in QUALITY_DIMENSIONS},
        "critical_errors": [reason[:500]],
        "rationale": "실행 오류 또는 빈 답변이므로 추가 LLM 호출 없이 자동 실패 처리했다.",
        "confidence": 1.0,
        "latency_ms": 0.0,
        "telemetry": {},
        "deterministic_contract_pass": _nested_value(
            row, "deterministic_metrics", "contract_pass"
        ),
    }


def _evaluation_error_result(row: dict[str, Any], model: str, exc: Exception) -> dict[str, Any]:
    return {
        "id": row["id"],
        "case_id": row.get("case_id"),
        "turn_id": row.get("turn_id"),
        "category": row.get("category"),
        "model": model,
        "status": "error",
        "evaluation_error": f"{type(exc).__name__}: {exc}",
    }


def _ordered_results(
    input_rows: Sequence[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [by_id[str(row["id"])] for row in input_rows if str(row["id"]) in by_id]


def _write_ordered_results(
    output_path: Path,
    input_rows: Sequence[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> None:
    _write_jsonl(output_path, _ordered_results(input_rows, by_id))


def _aggregate_result_telemetry(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    openai_keys = (
        "call_count",
        "chat_call_count",
        "embedding_call_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "estimated_cost_usd",
        "error_count",
        "latency_ms",
    )
    openai = {key: 0 for key in openai_keys}
    external_calls = 0
    estimated_cost = 0.0
    for result in results:
        telemetry = result.get("telemetry") if isinstance(result.get("telemetry"), dict) else {}
        external_calls += int(telemetry.get("external_call_count") or 0)
        estimated_cost += float(telemetry.get("estimated_cost_usd") or 0.0)
        source = telemetry.get("openai") if isinstance(telemetry.get("openai"), dict) else {}
        for key in openai_keys:
            openai[key] += source.get(key) or 0
    openai["estimated_cost_usd"] = round(float(openai["estimated_cost_usd"]), 8)
    openai["latency_ms"] = round(float(openai["latency_ms"]), 3)
    return {
        "external_call_count": external_calls,
        "estimated_cost_usd": round(estimated_cost, 8),
        "openai": openai,
    }


def _langchain_usage(response: Any) -> dict[str, int]:
    output = getattr(response, "llm_output", None)
    if isinstance(output, dict):
        usage = output.get("token_usage") or output.get("usage")
        if isinstance(usage, dict):
            return usage
    try:
        message = response.generations[0][0].message
    except (AttributeError, IndexError, TypeError):
        return {}
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }


def _active_collector() -> TelemetryCollector:
    from steam_rag.common.telemetry import current_telemetry

    collector = current_telemetry()
    if collector is None:
        raise RuntimeError("RAGAS evaluation requires a telemetry_session")
    return collector


def _nested_value(row: dict[str, Any], parent: str, name: str) -> Any:
    value = row.get(parent)
    return value.get(name) if isinstance(value, dict) else None


def _has_source_metadata(row: dict[str, Any]) -> bool:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    sources = payload.get("sources") if isinstance(payload, dict) else []
    return bool(sources) if isinstance(sources, list) else False


def _compact_candidate_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    games = payload.get("games") if isinstance(payload, dict) else []
    compact: list[dict[str, Any]] = []
    for game in games if isinstance(games, list) else []:
        if not isinstance(game, dict):
            continue
        compact.append(
            {
                "appid": game.get("appid"),
                "name": game.get("name"),
                "score": game.get("score"),
                "genres": list(game.get("genres") or [])[:8],
                "popular_tags": list(game.get("popular_tags") or [])[:12],
                "matched_tags": list(game.get("matched_tags") or [])[:12],
                "matched_facets": list(game.get("matched_facets") or [])[:12],
                "matched_aspects": list(game.get("matched_aspects") or [])[:12],
                "positive_ratio": game.get("positive_ratio"),
                "sample_size": game.get("sample_size"),
                "is_free": game.get("is_free"),
                "discount_percent": game.get("discount_percent"),
                "price": game.get("price"),
                "release_date": game.get("release_date"),
                "release_coming_soon": game.get("release_coming_soon"),
                "store_summary": str(game.get("store_summary") or "")[:600],
            }
        )
    return compact[:5]


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _has_valid_ragas_metrics(metrics: Any) -> bool:
    if not isinstance(metrics, dict):
        return False
    return all(_optional_float(metrics.get(name)) is not None for name in RAGAS_METRICS)


def _is_valid_ragas_result(
    row: dict[str, Any],
    *,
    model: str | None = None,
    embedding_model: str | None = None,
) -> bool:
    if row.get("status") != "complete" or not _has_valid_ragas_metrics(row.get("metrics")):
        return False
    if model is not None and row.get("model") != model:
        return False
    if embedding_model is not None and row.get("embedding_model") != embedding_model:
        return False
    return True


def _uses_fixed_temperature_reasoning_model(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _mean(values: Sequence[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(sum(usable) / len(usable), 6) if usable else None
