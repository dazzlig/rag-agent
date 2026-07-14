from __future__ import annotations

import csv
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from steam_rag.agents.agentic_rag import AgenticRAGConfig, AgenticRAGCoordinator
from steam_rag.common.interfaces import AnswerGenerator, Embedder
from steam_rag.common.models import SearchResult
from steam_rag.rag_search.hybrid_retriever import HybridTimeAwareRetriever, _cosine, augment_query
from steam_rag.rag_search.reranker import Reranker
from steam_rag.rag_search.search_spec import SearchSpec, evaluate_evidence_coverage
from steam_rag.rag_search.vector_store import VectorIndex


STAGE4_STRATEGIES = ("agentic", "agentic_hyde")
SUPPORTED_STRATEGIES = ("basic", "hybrid", "reranker", *STAGE4_STRATEGIES)


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
