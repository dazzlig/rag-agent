from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from steam_rag.evaluation_tools.semantic_quality import (
    DEFAULT_JUDGE_MAX_COMPLETION_TOKENS,
    DEFAULT_RAGAS_MAX_COMPLETION_TOKENS,
    DEFAULT_RAGAS_MODEL,
    QualityJudgeDecision,
    OpenAIQualityJudge,
    build_judge_inputs,
    build_ragas_inputs,
    load_jsonl,
    prepare_semantic_evaluation,
    run_llm_judge,
    run_ragas_evaluation,
)


def _detail(
    case_id: str,
    turn_id: str,
    *,
    answer: str = "근거에 따른 답변 [근거 1]",
    error: str = "",
    sources: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "category": "comparison",
        "title": "테스트",
        "turn_id": turn_id,
        "turn_number": int(turn_id.removeprefix("T")),
        "question": f"질문 {turn_id}",
        "answer": answer,
        "mode": "research" if answer else "",
        "expected": {
            "mode": "research",
            "required_keywords": ["근거"],
            "appids": [10],
            "answer_required": True,
        },
        "forbidden": {"keywords": [], "appids": [99], "modes": ["recommendation"]},
        "observed_appids": [10] if answer else [],
        "error": error,
        "metrics": {"contract_pass": 1.0 if answer and not error else 0.0},
        "payload": {
            "sources": sources if sources is not None else [
                {
                    "rank": 1,
                    "game": "Game",
                    "appid": "10",
                    "section": "about",
                    "source_type": "steam_corpus",
                    "snippet": "공식 게임 설명 근거",
                }
            ],
            "evidence_coverage": {"coverage_ratio": 1.0},
            "games": [
                {
                    "appid": 10,
                    "name": "Game",
                    "genres": ["RPG"],
                    "popular_tags": ["Story Rich"],
                    "store_summary": "검증된 후보 설명",
                }
            ],
        },
    }


class _FakeJudge:
    model = "fake-judge"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def judge(self, row: dict[str, object]) -> QualityJudgeDecision:
        self.calls.append(str(row["id"]))
        return QualityJudgeDecision(
            verdict="pass",
            target_entity_correctness=4,
            requirement_satisfaction=4,
            evidence_grounding=3,
            temporal_correctness=None,
            conversation_continuity=4,
            readability=3,
            critical_errors=[],
            rationale="대상과 요구를 지켰고 근거가 답변을 뒷받침한다.",
            confidence=0.9,
        )


class _FakeRagas:
    model = "fake-ragas"
    embedding_model = "fake-embedding"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def evaluate(self, row: dict[str, object]) -> dict[str, float]:
        self.calls.append(str(row["id"]))
        return {"faithfulness": 0.8, "answer_relevancy": 0.6}


class _NullThenValidRagas(_FakeRagas):
    def evaluate(self, row: dict[str, object]) -> dict[str, float | None]:
        self.calls.append(str(row["id"]))
        if len(self.calls) == 1:
            return {"faithfulness": None, "answer_relevancy": None}
        return {"faithfulness": 0.8, "answer_relevancy": 0.6}


class SemanticQualityTest(unittest.TestCase):
    def test_ragas_default_model_remains_gpt_4o_mini(self) -> None:
        from steam_rag.cli import _parser
        from steam_rag.evaluation_tools.semantic_quality import OpenAIRagasEvaluator

        args = _parser().parse_args(
            ["evaluate-quality", "--prepared-dir", "data/eval/runs/example"]
        )

        self.assertEqual(DEFAULT_RAGAS_MODEL, "gpt-4o-mini")
        self.assertEqual(args.ragas_model, "gpt-4o-mini")
        self.assertEqual(OpenAIRagasEvaluator().model, "gpt-4o-mini")
        self.assertEqual(
            OpenAIRagasEvaluator().max_completion_tokens,
            DEFAULT_RAGAS_MAX_COMPLETION_TOKENS,
        )

    def test_separate_engine_runs_preserve_both_combined_summaries(self) -> None:
        from steam_rag.cli import _merge_quality_run_summaries

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "judge-summary.json").write_text(
                json.dumps({"completed_count": 42}),
                encoding="utf-8",
            )
            merged = _merge_quality_run_summaries(
                root,
                {
                    "prepared_dir": str(root),
                    "resume": True,
                    "ragas": {"completed_count": 15},
                },
                invoked_engines=["ragas"],
            )

            self.assertEqual(merged["engines"], ["ragas", "judge"])
            self.assertEqual(merged["invoked_engines"], ["ragas"])
            self.assertEqual(merged["judge"], {"completed_count": 42})

    def test_openai_judge_uses_bounded_structured_output(self) -> None:
        decision = QualityJudgeDecision(
            verdict="partial",
            target_entity_correctness=4,
            requirement_satisfaction=2,
            evidence_grounding=3,
            temporal_correctness=None,
            conversation_continuity=None,
            readability=3,
            critical_errors=["추천 조건 일부 누락"],
            rationale="대상은 맞지만 요구 조건을 일부만 충족했다.",
            confidence=0.8,
        )

        class Completions:
            def __init__(self) -> None:
                self.kwargs: dict[str, object] = {}

            def parse(self, **kwargs: object) -> object:
                self.kwargs = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(parsed=decision))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                )

        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        judge = OpenAIQualityJudge("gpt-5-mini", client=client)

        result = judge.judge({"id": "C001/T1", "question": "질문", "answer": "답변"})

        self.assertEqual(result.verdict, "partial")
        self.assertIs(completions.kwargs["response_format"], QualityJudgeDecision)
        self.assertEqual(
            completions.kwargs["max_completion_tokens"],
            DEFAULT_JUDGE_MAX_COMPLETION_TOKENS,
        )
        self.assertEqual(completions.kwargs["reasoning_effort"], "minimal")
        self.assertEqual(completions.kwargs["verbosity"], "low")

    def test_input_selection_preserves_sources_and_conversation_history(self) -> None:
        details = [
            _detail("C001", "T1"),
            _detail("C001", "T2", answer="", error="LookupError: missing"),
            _detail("C002", "T1", sources=[]),
        ]

        ragas = build_ragas_inputs(details)
        judge = build_judge_inputs(details)

        self.assertEqual([row["id"] for row in ragas], ["C001/T1"])
        self.assertIn("공식 게임 설명 근거", ragas[0]["contexts"][0])
        self.assertEqual(len(judge), 3)
        self.assertEqual(judge[0]["candidate_evidence"][0]["appid"], 10)
        self.assertEqual(judge[1]["conversation_history"][0]["question"], "질문 T1")
        self.assertFalse(judge[1]["requires_llm"])
        self.assertIn("LookupError", judge[1]["automatic_failure"])

    def test_ragas_prefers_persisted_original_evidence_over_ui_snippet(self) -> None:
        detail = _detail("C003", "T1")
        detail["payload"]["evidence_contexts"] = [
            {
                "source_id": "chunk:1",
                "game": "Game",
                "appid": 10,
                "section": "about",
                "content": "원본 근거 전체 문장과 뒤쪽 세부 정보",
            }
        ]

        ragas = build_ragas_inputs([detail])

        self.assertIn("원본 근거 전체 문장과 뒤쪽 세부 정보", ragas[0]["contexts"][0])
        self.assertNotIn("공식 게임 설명 근거", ragas[0]["contexts"][0])

    def test_prepare_writes_reproducible_selection_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            details_path = root / "details.jsonl"
            metadata_only = _detail(
                "C002",
                "T1",
                sources=[{"title": "후보 발굴 참고 자료", "url": "https://example.com"}],
            )
            details = [
                _detail("C001", "T1"),
                _detail("C001", "T2", answer="", error="x"),
                metadata_only,
            ]
            details_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details),
                encoding="utf-8",
            )

            summary = prepare_semantic_evaluation(details_path, root / "quality")

            self.assertEqual(summary["turn_count"], 3)
            self.assertEqual(summary["ragas"]["selected_count"], 1)
            self.assertEqual(summary["ragas"]["source_metadata_only_excluded_ids"], ["C002/T1"])
            self.assertEqual(summary["llm_judge"]["requires_llm_count"], 2)
            self.assertEqual(summary["llm_judge"]["automatic_failure_count"], 1)
            self.assertTrue((root / "quality" / "selection-summary.json").exists())

    def test_judge_auto_fails_errors_and_resumes_completed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = build_judge_inputs(
                [_detail("C001", "T1"), _detail("C001", "T2", answer="", error="boom")]
            )
            judge = _FakeJudge()
            output = root / "judge.jsonl"
            summary_path = root / "judge-summary.json"

            first = run_llm_judge(
                rows,
                judge=judge,
                output_path=output,
                summary_path=summary_path,
            )
            second = run_llm_judge(
                rows,
                judge=judge,
                output_path=output,
                summary_path=summary_path,
            )

            self.assertEqual(judge.calls, ["C001/T1"])
            self.assertEqual(first["completed_count"], 2)
            self.assertEqual(first["automatic_failure_count"], 1)
            self.assertEqual(first["verdicts"], {"pass": 1, "partial": 0, "fail": 1})
            self.assertEqual(second["pending_count"], 0)
            self.assertEqual([row["status"] for row in load_jsonl(output)], ["complete", "automatic_failure"])

    def test_ragas_runner_limits_new_rows_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = build_ragas_inputs([_detail("C001", "T1"), _detail("C002", "T1")])
            evaluator = _FakeRagas()
            output = root / "ragas.jsonl"
            summary_path = root / "ragas-summary.json"

            first = run_ragas_evaluation(
                rows,
                evaluator=evaluator,
                output_path=output,
                summary_path=summary_path,
                limit=1,
            )
            second = run_ragas_evaluation(
                rows,
                evaluator=evaluator,
                output_path=output,
                summary_path=summary_path,
            )

            self.assertEqual(first["completed_count"], 1)
            self.assertEqual(first["pending_count"], 1)
            self.assertEqual(second["completed_count"], 2)
            self.assertEqual(second["metrics"]["faithfulness"], 0.8)
            self.assertEqual(evaluator.calls, ["C001/T1", "C002/T1"])

    def test_ragas_null_metrics_are_errors_and_resume_retries_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = build_ragas_inputs([_detail("C001", "T1")])
            evaluator = _NullThenValidRagas()
            output = root / "ragas.jsonl"
            summary_path = root / "ragas-summary.json"

            first = run_ragas_evaluation(
                rows,
                evaluator=evaluator,
                output_path=output,
                summary_path=summary_path,
            )
            second = run_ragas_evaluation(
                rows,
                evaluator=evaluator,
                output_path=output,
                summary_path=summary_path,
            )

            self.assertEqual(first["completed_count"], 0)
            self.assertEqual(first["evaluation_error_count"], 1)
            self.assertEqual(second["completed_count"], 1)
            self.assertEqual(second["evaluation_error_count"], 0)
            self.assertEqual(evaluator.calls, ["C001/T1", "C001/T1"])

    def test_ragas_resume_retries_legacy_complete_rows_with_null_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = build_ragas_inputs([_detail("C001", "T1")])
            output = root / "ragas.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "id": "C001/T1",
                        "status": "complete",
                        "metrics": {"faithfulness": None, "answer_relevancy": None},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            evaluator = _FakeRagas()

            summary = run_ragas_evaluation(
                rows,
                evaluator=evaluator,
                output_path=output,
                summary_path=root / "ragas-summary.json",
            )

            self.assertEqual(evaluator.calls, ["C001/T1"])
            self.assertEqual(summary["completed_count"], 1)
            self.assertEqual(summary["evaluation_error_count"], 0)
            self.assertEqual(load_jsonl(output)[0]["metrics"]["faithfulness"], 0.8)

    def test_ragas_resume_does_not_reuse_a_different_model_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = build_ragas_inputs([_detail("C001", "T1")])
            output = root / "ragas.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "id": "C001/T1",
                        "model": "gpt-5-mini",
                        "embedding_model": "fake-embedding",
                        "status": "complete",
                        "metrics": {"faithfulness": 0.1, "answer_relevancy": 0.1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            evaluator = _FakeRagas()

            run_ragas_evaluation(
                rows,
                evaluator=evaluator,
                output_path=output,
                summary_path=root / "ragas-summary.json",
            )

            saved = load_jsonl(output)[0]
            self.assertEqual(evaluator.calls, ["C001/T1"])
            self.assertEqual(saved["model"], "fake-ragas")
            self.assertEqual(saved["metrics"]["faithfulness"], 0.8)


if __name__ == "__main__":
    unittest.main()
