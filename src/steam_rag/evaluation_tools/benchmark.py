from __future__ import annotations

import csv
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from steam_rag.agents.agentic_rag import AgenticRAGConfig, AgenticRAGCoordinator
from steam_rag.common.interfaces import AnswerGenerator, Embedder
from steam_rag.common.models import SearchResult
from steam_rag.common.telemetry import telemetry_session
from steam_rag.rag_search.hybrid_retriever import HybridTimeAwareRetriever, _cosine, augment_query
from steam_rag.rag_search.reranker import Reranker
from steam_rag.rag_search.search_spec import SearchSpec, evaluate_evidence_coverage
from steam_rag.rag_search.vector_store import VectorIndex


STAGE4_STRATEGIES = ("agentic", "agentic_hyde")
SUPPORTED_STRATEGIES = ("basic", "hybrid", "reranker", *STAGE4_STRATEGIES)
CONVERSATION_GOLDEN_SET_SIZE = 20


@dataclass(frozen=True, slots=True)
class GoldenCase:
    case_id: str
    category: str
    question: str
    expected_intent: str
    expected_game_keys: tuple[str, ...] = ()
    expected_sections: tuple[str, ...] = ()
    expected_evidence_keywords: tuple[str, ...] = ()
    expected_answer_keywords: tuple[str, ...] = ()
    expected_appids: tuple[int, ...] = ()
    expected_patch_date: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoldenCase":
        return cls(
            case_id=str(value["id"]),
            category=str(value["category"]),
            question=str(value["question"]),
            expected_intent=str(value.get("expected_intent") or "general"),
            expected_game_keys=tuple(map(str, value.get("expected_game_keys") or ())),
            expected_sections=tuple(map(str, value.get("expected_sections") or ())),
            expected_evidence_keywords=tuple(map(str, value.get("expected_evidence_keywords") or ())),
            expected_answer_keywords=tuple(map(str, value.get("expected_answer_keywords") or ())),
            expected_appids=tuple(int(item) for item in value.get("expected_appids") or ()),
            expected_patch_date=str(value.get("expected_patch_date") or ""),
            notes=str(value.get("notes") or ""),
        )


@dataclass(frozen=True, slots=True)
class ConversationTurnExpectation:
    """Deterministic service contract for one conversation turn."""

    mode: str
    required_keywords: tuple[str, ...] = ()
    appids: tuple[int, ...] = ()
    answer_required: bool = True
    context_used: bool | None = None
    followup_relation: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConversationTurnExpectation":
        mode = str(value.get("mode") or "").strip()
        if mode not in {"research", "recommendation"}:
            raise ValueError("expected.mode must be 'research' or 'recommendation'")
        answer_required = value.get("answer_required", True)
        if not isinstance(answer_required, bool):
            raise ValueError("expected.answer_required must be boolean")
        return cls(
            mode=mode,
            required_keywords=_string_tuple(value.get("required_keywords")),
            appids=_appid_tuple(value.get("appids")),
            answer_required=answer_required,
            context_used=_optional_bool(value, "context_used"),
            followup_relation=str(value.get("followup_relation") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class ConversationTurnForbidden:
    """Content that must not leak into the answer or selected game set."""

    keywords: tuple[str, ...] = ()
    appids: tuple[int, ...] = ()
    modes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConversationTurnForbidden":
        modes = _string_tuple(value.get("modes"))
        unknown_modes = set(modes) - {"research", "recommendation"}
        if unknown_modes:
            raise ValueError(f"forbidden.modes contains unsupported values: {sorted(unknown_modes)}")
        return cls(
            keywords=_string_tuple(value.get("keywords")),
            appids=_appid_tuple(value.get("appids")),
            modes=modes,
        )


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    turn_id: str
    question: str
    expected: ConversationTurnExpectation
    forbidden: ConversationTurnForbidden
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConversationTurn":
        turn_id = str(value.get("id") or "").strip()
        question = str(value.get("question") or "").strip()
        if not turn_id:
            raise ValueError("conversation turn id is required")
        if not question:
            raise ValueError(f"conversation turn {turn_id!r} question is required")
        if "expected" not in value or not isinstance(value["expected"], dict):
            raise ValueError(f"conversation turn {turn_id!r} requires an expected contract")
        if "forbidden" not in value or not isinstance(value["forbidden"], dict):
            raise ValueError(f"conversation turn {turn_id!r} requires a forbidden contract")
        return cls(
            turn_id=turn_id,
            question=question,
            expected=ConversationTurnExpectation.from_dict(value["expected"]),
            forbidden=ConversationTurnForbidden.from_dict(value["forbidden"]),
            notes=str(value.get("notes") or ""),
        )


@dataclass(frozen=True, slots=True)
class ConversationCase:
    case_id: str
    category: str
    title: str
    turns: tuple[ConversationTurn, ...]
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConversationCase":
        case_id = str(value.get("id") or "").strip()
        category = str(value.get("category") or "").strip()
        title = str(value.get("title") or "").strip()
        raw_turns = value.get("turns")
        if not case_id or not category or not title:
            raise ValueError("conversation case id, category, and title are required")
        if not isinstance(raw_turns, list):
            raise ValueError(f"conversation case {case_id!r} turns must be a list")
        turns = tuple(ConversationTurn.from_dict(turn) for turn in raw_turns)
        if not 2 <= len(turns) <= 4:
            raise ValueError(f"conversation case {case_id!r} must contain 2-4 turns")
        turn_ids = [turn.turn_id for turn in turns]
        if len(set(turn_ids)) != len(turn_ids):
            raise ValueError(f"conversation case {case_id!r} turn ids must be unique")
        return cls(case_id, category, title, turns, str(value.get("notes") or ""))


class ConversationRuntime(Protocol):
    def ask(
        self,
        question: str,
        *,
        top_k: int = 6,
        history: list[dict[str, str]] | None = None,
        context_games: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class ConversationTurnRecord:
    case_id: str
    category: str
    title: str
    turn_id: str
    turn_number: int
    question: str
    answer: str
    mode: str
    expected: dict[str, Any]
    forbidden: dict[str, Any]
    observed_appids: list[int]
    latency_ms: float
    error: str
    metrics: dict[str, float | None]
    telemetry: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "title": self.title,
            "turn_id": self.turn_id,
            "turn_number": self.turn_number,
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "expected": self.expected,
            "forbidden": self.forbidden,
            "observed_appids": self.observed_appids,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metrics": self.metrics,
            "telemetry": self.telemetry,
            "payload": self.payload,
        }


@dataclass(slots=True)
class BenchmarkRecord:
    case_id: str
    category: str
    strategy: str
    question: str
    answer: str
    latency_ms: float
    result_count: int
    search_spec: dict[str, Any]
    evidence_coverage: dict[str, Any]
    metrics: dict[str, dict[str, float | None]]
    results: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "strategy": self.strategy,
            "question": self.question,
            "answer": self.answer,
            "latency_ms": self.latency_ms,
            "result_count": self.result_count,
            "search_spec": self.search_spec,
            "evidence_coverage": self.evidence_coverage,
            "metrics": self.metrics,
            "results": self.results,
            "trace": self.trace,
        }


def load_golden_set(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Golden Set JSONL at line {line_number}: {exc}") from exc
        cases.append(GoldenCase.from_dict(value))
    if not 40 <= len(cases) <= 60:
        raise ValueError(f"Stage 4 Golden Set must contain 40-60 cases, found {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Golden Set case ids must be unique")
    return cases


def load_conversation_golden_set(path: Path) -> list[ConversationCase]:
    """Load and validate the versioned, service-level conversation Golden Set."""

    cases: list[ConversationCase] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid conversation Golden Set JSONL at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"Conversation Golden Set line {line_number} must be an object")
        try:
            cases.append(ConversationCase.from_dict(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid conversation case at line {line_number}: {exc}") from exc
    if len(cases) != CONVERSATION_GOLDEN_SET_SIZE:
        raise ValueError(
            f"Conversation Golden Set must contain exactly {CONVERSATION_GOLDEN_SET_SIZE} cases, "
            f"found {len(cases)}"
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Conversation Golden Set case ids must be unique")
    return cases


class ServiceConversationBenchmarkRunner:
    """Run the consumer service across complete, stateful conversation scenarios."""

    def __init__(
        self,
        runtime: ConversationRuntime,
        *,
        top_k: int = 6,
        history_limit: int = 8,
    ) -> None:
        self.runtime = runtime
        self.top_k = max(1, min(int(top_k), 10))
        self.history_limit = max(1, int(history_limit))

    def run(self, cases: Sequence[ConversationCase]) -> list[ConversationTurnRecord]:
        records: list[ConversationTurnRecord] = []
        for case in cases:
            history: list[dict[str, str]] = []
            context_games: list[dict[str, Any]] = []
            for turn_number, turn in enumerate(case.turns, start=1):
                started = time.perf_counter()
                payload: dict[str, Any] = {}
                error = ""
                with telemetry_session() as telemetry_collector:
                    try:
                        raw_payload = self.runtime.ask(
                            turn.question,
                            top_k=self.top_k,
                            history=list(history[-self.history_limit :]),
                            context_games=list(context_games),
                        )
                        if not isinstance(raw_payload, dict):
                            raise TypeError("runtime.ask() must return a dict payload")
                        payload = raw_payload
                    except Exception as exc:  # Each failed turn remains observable in the report.
                        error = f"{type(exc).__name__}: {exc}"
                    turn_telemetry = telemetry_collector.snapshot()
                latency_ms = (time.perf_counter() - started) * 1000
                answer = str(payload.get("answer") or "").strip()
                observed_appids = sorted(_payload_appids(payload))
                metrics = score_conversation_turn(turn, payload, error=error)
                records.append(
                    ConversationTurnRecord(
                        case_id=case.case_id,
                        category=case.category,
                        title=case.title,
                        turn_id=turn.turn_id,
                        turn_number=turn_number,
                        question=turn.question,
                        answer=answer,
                        mode=str(payload.get("mode") or ""),
                        expected=_expectation_dict(turn.expected),
                        forbidden=_forbidden_dict(turn.forbidden),
                        observed_appids=observed_appids,
                        latency_ms=round(latency_ms, 3),
                        error=error,
                        metrics=metrics,
                        telemetry=turn_telemetry,
                        payload=payload,
                    )
                )
                history.append({"role": "user", "content": turn.question})
                context_games = _merge_context_games(context_games, payload.get("games"))
        return records


def score_conversation_turn(
    turn: ConversationTurn,
    payload: dict[str, Any],
    *,
    error: str = "",
) -> dict[str, float | None]:
    """Score explicit contracts only; no LLM or embedding call is made here."""

    answer = str(payload.get("answer") or "").strip()
    answer_text = answer.casefold()
    mode = str(payload.get("mode") or "")
    observed_appids = _payload_appids(payload)
    expectation = turn.expected
    forbidden = turn.forbidden

    answer_presence = 1.0 if bool(answer) == expectation.answer_required else 0.0
    required_keyword_recall = _keyword_recall(expectation.required_keywords, answer_text)
    forbidden_keyword_leakage = _keyword_match_ratio(forbidden.keywords, answer_text)
    expected_appid_recall = _set_recall(set(expectation.appids), observed_appids)
    forbidden_appid_leakage = _set_hit_ratio(set(forbidden.appids), observed_appids)
    mode_correctness = 1.0 if mode == expectation.mode else 0.0
    forbidden_mode_leakage = 1.0 if mode and mode in forbidden.modes else 0.0
    continuity = _continuity_score(expectation, payload)
    error_rate = 1.0 if error else 0.0

    required_checks = [answer_presence, mode_correctness, 1.0 - forbidden_mode_leakage]
    if required_keyword_recall is not None:
        required_checks.append(required_keyword_recall)
    if expected_appid_recall is not None:
        required_checks.append(expected_appid_recall)
    if forbidden_keyword_leakage is not None:
        required_checks.append(1.0 - forbidden_keyword_leakage)
    if forbidden_appid_leakage is not None:
        required_checks.append(1.0 - forbidden_appid_leakage)
    if continuity is not None:
        required_checks.append(continuity)
    contract_pass = 1.0 if not error and all(value == 1.0 for value in required_checks) else 0.0
    return {
        "answer_presence": answer_presence,
        "required_keyword_recall": required_keyword_recall,
        "forbidden_keyword_leakage": forbidden_keyword_leakage,
        "expected_appid_recall": expected_appid_recall,
        "forbidden_appid_leakage": forbidden_appid_leakage,
        "mode_correctness": mode_correctness,
        "forbidden_mode_leakage": forbidden_mode_leakage,
        "continuity": continuity,
        "error_rate": error_rate,
        "contract_pass": contract_pass,
    }


def _aggregate_operational_metrics(
    records: Sequence[ConversationTurnRecord],
) -> dict[str, Any]:
    openai_keys = (
        "call_count",
        "chat_call_count",
        "embedding_call_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "estimated_cost_usd",
        "unknown_cost_call_count",
        "error_count",
        "latency_ms",
    )
    tavily_keys = (
        "request_count",
        "external_call_count",
        "cache_hit_count",
        "cache_miss_count",
        "credits",
        "estimated_cost_usd",
        "error_count",
    )
    steam_keys = (
        "request_count",
        "attempt_count",
        "success_count",
        "error_count",
        "latency_ms",
    )
    corpus_keys = ("check_count", "collected_count", "reused_count", "indexed_count")
    openai: dict[str, Any] = {key: 0 for key in openai_keys}
    tavily: dict[str, Any] = {key: 0 for key in tavily_keys}
    steam: dict[str, Any] = {key: 0 for key in steam_keys}
    corpus: dict[str, Any] = {key: 0 for key in corpus_keys}
    models: dict[str, dict[str, Any]] = {}
    endpoints: dict[str, dict[str, int]] = {}
    external_call_count = 0
    estimated_cost_usd = 0.0
    turns_with_telemetry = 0

    for record in records:
        telemetry = record.telemetry if isinstance(record.telemetry, dict) else {}
        if telemetry:
            turns_with_telemetry += 1
        external_call_count += int(telemetry.get("external_call_count") or 0)
        estimated_cost_usd += float(telemetry.get("estimated_cost_usd") or 0.0)
        source_openai = telemetry.get("openai") if isinstance(telemetry.get("openai"), dict) else {}
        source_tavily = telemetry.get("tavily") if isinstance(telemetry.get("tavily"), dict) else {}
        source_steam = telemetry.get("steam") if isinstance(telemetry.get("steam"), dict) else {}
        source_corpus = telemetry.get("corpus") if isinstance(telemetry.get("corpus"), dict) else {}
        for key in openai_keys:
            openai[key] += source_openai.get(key) or 0
        for key in tavily_keys:
            tavily[key] += source_tavily.get(key) or 0
        for key in steam_keys:
            steam[key] += source_steam.get(key) or 0
        for key in corpus_keys:
            corpus[key] += source_corpus.get(key) or 0

        source_models = source_openai.get("models")
        if isinstance(source_models, dict):
            for model, source_row in source_models.items():
                if not isinstance(source_row, dict):
                    continue
                target = models.setdefault(str(model), {key: 0 for key in openai_keys})
                for key in openai_keys:
                    target[key] += source_row.get(key) or 0
        source_endpoints = source_steam.get("endpoints")
        if isinstance(source_endpoints, dict):
            for endpoint, source_row in source_endpoints.items():
                if not isinstance(source_row, dict):
                    continue
                target = endpoints.setdefault(
                    str(endpoint),
                    {
                        "request_count": 0,
                        "attempt_count": 0,
                        "success_count": 0,
                        "error_count": 0,
                    },
                )
                for key in target:
                    target[key] += int(source_row.get(key) or 0)

    for row in [openai, *models.values()]:
        row["estimated_cost_usd"] = round(float(row["estimated_cost_usd"]), 8)
        row["latency_ms"] = round(float(row["latency_ms"]), 3)
    openai["models"] = dict(sorted(models.items()))
    tavily["credits"] = round(float(tavily["credits"]), 4)
    tavily["estimated_cost_usd"] = round(float(tavily["estimated_cost_usd"]), 8)
    tavily["cache_hit_rate"] = (
        round(tavily["cache_hit_count"] / tavily["request_count"], 6)
        if tavily["request_count"]
        else None
    )
    steam["latency_ms"] = round(float(steam["latency_ms"]), 3)
    steam["endpoints"] = dict(sorted(endpoints.items()))
    turn_count = len(records)
    return {
        "turns_with_telemetry": turns_with_telemetry,
        "external_call_count": external_call_count,
        "estimated_cost_usd": round(estimated_cost_usd, 8),
        "mean_estimated_cost_usd_per_turn": (
            round(estimated_cost_usd / turn_count, 8) if turn_count else None
        ),
        "mean_external_calls_per_turn": (
            round(external_call_count / turn_count, 6) if turn_count else None
        ),
        "openai": openai,
        "tavily": tavily,
        "steam": steam,
        "corpus": corpus,
    }


def summarize_conversation_records(
    records: Sequence[ConversationTurnRecord],
) -> dict[str, Any]:
    latencies = [record.latency_ms for record in records]
    metric_names = sorted({name for record in records for name in record.metrics})
    metrics: dict[str, float | None] = {}
    for name in metric_names:
        values = [
            float(record.metrics[name])
            for record in records
            if record.metrics.get(name) is not None
        ]
        metrics[name] = round(sum(values) / len(values), 6) if values else None
    category_rows: list[dict[str, Any]] = []
    for category in sorted({record.category for record in records}):
        selected = [record for record in records if record.category == category]
        selected_operations = _aggregate_operational_metrics(selected)
        category_rows.append(
            {
                "category": category,
                "turn_count": len(selected),
                "contract_pass_rate": _mean_metric(selected, "contract_pass"),
                "error_rate": _mean_metric(selected, "error_rate"),
                "latency_ms_mean": round(
                    sum(record.latency_ms for record in selected) / len(selected), 3
                ),
                "estimated_cost_usd": selected_operations["estimated_cost_usd"],
                "external_call_count": selected_operations["external_call_count"],
            }
        )
    return {
        "metric_family": "deterministic_contract_heuristics",
        "case_count": len({record.case_id for record in records}),
        "turn_count": len(records),
        "metrics": metrics,
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "operations": _aggregate_operational_metrics(records),
        "categories": category_rows,
    }


def save_conversation_benchmark(
    records: Sequence[ConversationTurnRecord],
    *,
    details_path: Path,
    summary_path: Path,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(
        "\n".join(json.dumps(record.to_dict(), ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = summarize_conversation_records(records)
    if run_metadata:
        summary["run"] = dict(run_metadata)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


class Stage4BenchmarkRunner:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        *,
        generator: AnswerGenerator | None = None,
        reranker: Reranker | None = None,
        top_k: int = 5,
        rerank_candidates: int = 24,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.generator = generator
        self.reranker = reranker
        self.top_k = max(1, int(top_k))
        self.rerank_candidates = max(self.top_k, int(rerank_candidates))
        self.retriever = HybridTimeAwareRetriever(index)

    def run(
        self,
        cases: Sequence[GoldenCase],
        strategies: Sequence[str] = STAGE4_STRATEGIES,
        *,
        generate_answers: bool = True,
    ) -> list[BenchmarkRecord]:
        unknown = [strategy for strategy in strategies if strategy not in SUPPORTED_STRATEGIES]
        if unknown:
            raise ValueError(f"Unknown benchmark strategies: {unknown}")
        records: list[BenchmarkRecord] = []
        for strategy in strategies:
            for case in cases:
                records.append(self._run_case(case, strategy, generate_answers=generate_answers))
        return records

    def _run_case(
        self,
        case: GoldenCase,
        strategy: str,
        *,
        generate_answers: bool,
    ) -> BenchmarkRecord:
        spec = self.retriever.build_search_spec(case.question)
        started = time.perf_counter()
        results, trace = self._search(case.question, spec, strategy)
        answer = ""
        if generate_answers:
            if self.generator is None:
                raise RuntimeError("generator is required unless --retrieval-only is used")
            if strategy in {"agentic", "agentic_hyde"} and hasattr(self.generator, "generate_agentic"):
                answer = self.generator.generate_agentic(case.question, results, trace)  # type: ignore[attr-defined]
            else:
                answer = self.generator.generate(case.question, results)
        latency_ms = (time.perf_counter() - started) * 1000
        coverage = evaluate_evidence_coverage(spec, results)
        metrics = score_case(case, spec, results, answer, latency_ms)
        return BenchmarkRecord(
            case.case_id,
            case.category,
            strategy,
            case.question,
            answer,
            round(latency_ms, 3),
            len(results),
            spec.to_dict(),
            coverage.to_dict(),
            metrics,
            [result.to_dict(include_content=False) for result in results],
            trace,
        )

    def _search(
        self,
        question: str,
        spec: SearchSpec,
        strategy: str,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        if strategy == "basic":
            embedding = self.embedder.embed_query(question)
            return self._basic_dense(embedding, spec), {"strategy": "basic"}
        if strategy in {"agentic", "agentic_hyde"}:
            if self.generator is None:
                raise RuntimeError("agentic benchmark strategies require a generator for planning/HyDE")
            coordinator = AgenticRAGCoordinator(
                self.retriever,
                self.embedder,
                self.generator,
                config=AgenticRAGConfig(
                    max_steps=3,
                    per_step_k=self.rerank_candidates if self.reranker else self.top_k,
                    use_hyde=strategy == "agentic_hyde",
                ),
                reranker=self.reranker,
                rerank_candidates=self.rerank_candidates,
            )
            return coordinator.search(question, k=self.top_k)

        embedding = self.embedder.embed_query(augment_query(question, spec.intent))
        candidate_k = self.rerank_candidates if strategy == "reranker" else self.top_k
        results = self.retriever.retrieve(
            question,
            embedding,
            k=candidate_k,
            search_spec=spec,
        )
        if strategy == "reranker":
            if self.reranker is None:
                raise RuntimeError("reranker strategy requires a configured reranker")
            results = self.reranker.rerank(question, results, top_n=self.top_k)
        return results, {"strategy": strategy, "search_spec": spec.to_dict()}

    def _basic_dense(self, embedding: Sequence[float], spec: SearchSpec) -> list[SearchResult]:
        allowed_sections = set(spec.primary_sections) | set(spec.secondary_sections)
        scored: list[SearchResult] = []
        for document, vector in zip(self.index.documents, self.index.embeddings, strict=True):
            metadata = document.metadata
            if allowed_sections and str(metadata.get("section") or "") not in allowed_sections:
                continue
            if spec.game_keys and str(metadata.get("game_key") or "") not in spec.game_keys:
                continue
            scored.append(SearchResult(document, _cosine(embedding, vector), intent=spec.intent))
        scored.sort(key=lambda result: result.score, reverse=True)
        for rank, result in enumerate(scored[: self.top_k], start=1):
            result.rank = rank
        return scored[: self.top_k]


def score_case(
    case: GoldenCase,
    spec: SearchSpec,
    results: Sequence[SearchResult],
    answer: str,
    latency_ms: float,
) -> dict[str, dict[str, float | None]]:
    metadata = [result.document.metadata for result in results]
    retrieved_games = {str(row.get("game_key") or "") for row in metadata}
    retrieved_sections = {str(row.get("section") or "") for row in metadata}
    evidence_text = " ".join(result.document.page_content for result in results).casefold()
    relevant = [
        row
        for row in metadata
        if (not case.expected_game_keys or str(row.get("game_key") or "") in case.expected_game_keys)
        and (not case.expected_sections or str(row.get("section") or "") in case.expected_sections)
    ]
    coverage = evaluate_evidence_coverage(spec, results)
    retrieval = {
        "game_recall": _set_recall(set(case.expected_game_keys), retrieved_games),
        "section_recall": _set_recall(set(case.expected_sections), retrieved_sections),
        "evidence_keyword_recall": _keyword_recall(case.expected_evidence_keywords, evidence_text),
        "context_precision": len(relevant) / len(results) if results else 0.0,
        "claim_evidence_coverage": coverage.coverage_ratio,
    }

    answer_text = answer.casefold()
    generation = {
        "answer_keyword_recall": _keyword_recall(case.expected_answer_keywords, answer_text) if answer else None,
        "answer_present": 1.0 if answer else (None if not answer else 0.0),
    }
    citations = _extract_citations(answer)
    valid_citations = [value for value in citations if 1 <= value <= len(results)]
    citation = {
        "citation_validity": len(valid_citations) / len(citations) if citations else (0.0 if answer else None),
        "citation_source_coverage": len(set(valid_citations)) / len(results) if results and answer else (0.0 if answer else None),
        "claim_citation_coverage": _claim_citation_coverage(case.expected_answer_keywords, answer),
    }

    date_values = {
        str(value)
        for row, result in zip(metadata, results, strict=False)
        for value in (
            row.get("source_date"), row.get("patch_date"), result.latest_patch_date
        )
        if value
    }
    temporal = {
        "patch_date_accuracy": (
            1.0 if case.expected_patch_date in date_values else 0.0
        ) if case.expected_patch_date else None,
        "dated_context_ratio": (
            sum(bool(row.get("source_date") or row.get("patch_date")) for row in metadata) / len(metadata)
            if metadata else 0.0
        ),
    }

    retrieved_appids = {
        int(row["appid"])
        for row in metadata
        if str(row.get("appid") or "").isdigit()
    }
    recommendation = {
        "recommended_game_recall": _set_recall(set(case.expected_appids), retrieved_appids),
        "constraint_claim_coverage": coverage.coverage_ratio if spec.recommendation else None,
    }
    return {
        "retrieval": retrieval,
        "generation": generation,
        "citation": citation,
        "temporal": temporal,
        "recommendation": recommendation,
        "operations": {"latency_ms": round(latency_ms, 3), "result_count": float(len(results))},
    }


def summarize_records(records: Sequence[BenchmarkRecord]) -> list[dict[str, Any]]:
    strategies = sorted({record.strategy for record in records})
    summary: list[dict[str, Any]] = []
    for strategy in strategies:
        selected = [record for record in records if record.strategy == strategy]
        row: dict[str, Any] = {"strategy": strategy, "case_count": len(selected)}
        metric_values: dict[str, list[float]] = {}
        for record in selected:
            for group, metrics in record.metrics.items():
                for name, value in metrics.items():
                    if value is not None and math.isfinite(float(value)):
                        metric_values.setdefault(f"{group}.{name}", []).append(float(value))
        for name, values in sorted(metric_values.items()):
            row[name] = round(sum(values) / len(values), 6)
        summary.append(row)
    return summary


def save_benchmark(
    records: Sequence[BenchmarkRecord],
    *,
    details_path: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(
        "\n".join(json.dumps(record.to_dict(), ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = summarize_records(records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in summary for key in row}, key=lambda key: (key not in {"strategy", "case_count"}, key))
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def _set_recall(expected: set[Any], actual: set[Any]) -> float | None:
    return len(expected & actual) / len(expected) if expected else None


def _keyword_recall(keywords: Sequence[str], text: str) -> float | None:
    return sum(keyword.casefold() in text for keyword in keywords) / len(keywords) if keywords else None


def _claim_citation_coverage(keywords: Sequence[str], answer: str) -> float | None:
    if not answer:
        return None
    relevant = [
        line
        for line in answer.splitlines()
        if line.strip() and any(keyword.casefold() in line.casefold() for keyword in keywords)
    ]
    if not relevant:
        return 0.0 if keywords else None
    return sum(bool(_extract_citations(sentence)) for sentence in relevant) / len(relevant)


def _extract_citations(answer: str) -> list[int]:
    citations: list[int] = []
    for group in re.findall(r"\[근거\s*([0-9,\s]+)\]", answer):
        citations.extend(int(value) for value in re.findall(r"\d+", group))
    return citations


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("contract string collections must be lists")
    output = tuple(str(item).strip() for item in value if str(item).strip())
    if len(set(output)) != len(output):
        raise ValueError("contract string collections must not contain duplicates")
    return output


def _appid_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("contract appids must be lists")
    try:
        output = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("contract appids must contain integers") from exc
    if any(appid <= 0 for appid in output):
        raise ValueError("contract appids must be positive")
    if len(set(output)) != len(output):
        raise ValueError("contract appids must not contain duplicates")
    return output


def _optional_bool(value: dict[str, Any], key: str) -> bool | None:
    if key not in value or value[key] is None:
        return None
    if not isinstance(value[key], bool):
        raise ValueError(f"expected.{key} must be boolean or null")
    return value[key]


def _expectation_dict(expectation: ConversationTurnExpectation) -> dict[str, Any]:
    return {
        "mode": expectation.mode,
        "required_keywords": list(expectation.required_keywords),
        "appids": list(expectation.appids),
        "answer_required": expectation.answer_required,
        "context_used": expectation.context_used,
        "followup_relation": expectation.followup_relation,
    }


def _forbidden_dict(forbidden: ConversationTurnForbidden) -> dict[str, Any]:
    return {
        "keywords": list(forbidden.keywords),
        "appids": list(forbidden.appids),
        "modes": list(forbidden.modes),
    }


def _payload_appids(payload: dict[str, Any]) -> set[int]:
    """Collect verified game identities without counting unrelated numeric fields."""

    appids: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            raw_appid = value.get("appid")
            try:
                appid = int(raw_appid)
            except (TypeError, ValueError):
                appid = 0
            if appid > 0:
                appids.add(appid)
            for nested in value.values():
                if isinstance(nested, (dict, list, tuple)):
                    visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    for key in ("games", "corpus_updates"):
        visit(payload.get(key))
    return appids


def _merge_context_games(
    existing: Sequence[dict[str, Any]],
    new_games: Any,
) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for raw in new_games if isinstance(new_games, list) else []:
        if not isinstance(raw, dict):
            continue
        try:
            appid = int(raw.get("appid"))
        except (TypeError, ValueError):
            continue
        name = str(raw.get("name") or "").strip()
        if appid > 0 and name:
            latest[appid] = {"appid": appid, "name": name}
    if latest:
        return list(latest.values())

    retained: dict[int, dict[str, Any]] = {}
    for raw in existing:
        try:
            appid = int(raw.get("appid"))
        except (AttributeError, TypeError, ValueError):
            continue
        name = str(raw.get("name") or "").strip()
        if appid > 0 and name:
            retained[appid] = {"appid": appid, "name": name}
    return list(retained.values())


def _keyword_match_ratio(keywords: Sequence[str], text: str) -> float | None:
    return sum(keyword.casefold() in text for keyword in keywords) / len(keywords) if keywords else None


def _set_hit_ratio(forbidden: set[Any], actual: set[Any]) -> float | None:
    return len(forbidden & actual) / len(forbidden) if forbidden else None


def _continuity_score(
    expectation: ConversationTurnExpectation,
    payload: dict[str, Any],
) -> float | None:
    checks: list[bool] = []
    if expectation.context_used is not None:
        checks.append(bool(payload.get("conversation_context_used")) == expectation.context_used)
    if expectation.followup_relation:
        observed_relation = str(payload.get("followup_relation") or "")
        accepted_relations = (
            {"detail", "continuation"}
            if expectation.followup_relation == "detail"
            else {expectation.followup_relation}
        )
        checks.append(observed_relation in accepted_relations)
    return sum(checks) / len(checks) if checks else None


def _mean_metric(records: Sequence[ConversationTurnRecord], name: str) -> float | None:
    values = [float(record.metrics[name]) for record in records if record.metrics.get(name) is not None]
    return round(sum(values) / len(values), 6) if values else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return round(ordered[index], 3)
