from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from steam_rag.evaluation_tools.benchmark import (
    ConversationCase,
    ServiceConversationBenchmarkRunner,
    load_conversation_golden_set,
    save_conversation_benchmark,
    summarize_conversation_records,
)


class FakeConversationRuntime:
    def __init__(self, *, fail_on_call: int = 0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on_call = fail_on_call

    def ask(
        self,
        question: str,
        *,
        top_k: int = 6,
        history: list[dict[str, str]] | None = None,
        context_games: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "question": question,
                "top_k": top_k,
                "history": list(history or []),
                "context_games": list(context_games or []),
            }
        )
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("synthetic service failure")
        if len(self.calls) == 1:
            return {
                "mode": "recommendation",
                "answer": "친구와 즐기는 협동 추천입니다.",
                "games": [{"appid": 10, "name": "Example Co-op"}],
                "recommendation": {
                    "excluded_appids": [99],
                    "reference_game": {"appid": 98, "name": "Seed Game"},
                },
                "conversation_context_used": False,
                "followup_relation": "standalone",
            }
        return {
            "mode": "research",
            "answer": "Example Co-op의 전투와 협동 방식을 자세히 설명합니다.",
            "games": [{"appid": 10, "name": "Example Co-op"}],
            "conversation_context_used": True,
            "followup_relation": "continuation",
        }


def _case() -> ConversationCase:
    return ConversationCase.from_dict(
        {
            "id": "TEST",
            "category": "detail_followup",
            "title": "fake scenario",
            "turns": [
                {
                    "id": "T1",
                    "question": "협동 게임 추천해줘",
                    "expected": {
                        "mode": "recommendation",
                        "required_keywords": ["협동"],
                        "appids": [10],
                    },
                    "forbidden": {
                        "keywords": ["싱글플레이 전용"],
                        "appids": [99],
                        "modes": ["research"],
                    },
                },
                {
                    "id": "T2",
                    "question": "첫 게임 전투를 자세히 알려줘",
                    "expected": {
                        "mode": "research",
                        "required_keywords": ["전투"],
                        "appids": [10],
                        "context_used": True,
                        "followup_relation": "continuation",
                    },
                    "forbidden": {
                        "keywords": ["Hades II"],
                        "appids": [1145350],
                        "modes": ["recommendation"],
                    },
                },
            ],
        }
    )


class ConversationBenchmarkTests(unittest.TestCase):
    def test_versioned_golden_set_has_twenty_valid_multiturn_cases(self) -> None:
        cases = load_conversation_golden_set(Path("data/eval/conversation_golden_set_v1.jsonl"))

        self.assertEqual(len(cases), 20)
        self.assertTrue(all(2 <= len(case.turns) <= 4 for case in cases))
        self.assertEqual(len({case.case_id for case in cases}), 20)
        self.assertEqual(
            {case.category for case in cases},
            {
                "recommendation",
                "seed_similarity",
                "comparison",
                "correction_followup",
                "detail_followup",
                "price_sale_upcoming",
                "time_aware_update",
                "alias_localized_names",
            },
        )
        self.assertTrue(
            all(turn.expected.mode and turn.forbidden is not None for case in cases for turn in case.turns)
        )

    def test_runner_forwards_history_and_verified_context_between_turns(self) -> None:
        runtime = FakeConversationRuntime()
        records = ServiceConversationBenchmarkRunner(runtime, top_k=7).run([_case()])

        self.assertEqual(len(records), 2)
        self.assertEqual(runtime.calls[0]["history"], [])
        self.assertEqual(runtime.calls[0]["context_games"], [])
        self.assertEqual(runtime.calls[1]["top_k"], 7)
        self.assertEqual(
            runtime.calls[1]["history"],
            [{"role": "user", "content": "협동 게임 추천해줘"}],
        )
        self.assertEqual(
            runtime.calls[1]["context_games"],
            [{"appid": 10, "name": "Example Co-op"}],
        )
        self.assertEqual(records[0].observed_appids, [10])
        self.assertEqual(records[0].metrics["forbidden_appid_leakage"], 0.0)
        self.assertTrue(all(record.metrics["contract_pass"] == 1.0 for record in records))

    def test_latest_nonempty_game_set_replaces_older_context_like_the_web_client(self) -> None:
        case = ConversationCase.from_dict(
            {
                "id": "REPLACE",
                "category": "correction_followup",
                "title": "replace context",
                "turns": [
                    {
                        "id": "T1",
                        "question": "첫 추천",
                        "expected": {"mode": "recommendation"},
                        "forbidden": {},
                    },
                    {
                        "id": "T2",
                        "question": "다른 게임으로 바꿔줘",
                        "expected": {"mode": "recommendation"},
                        "forbidden": {},
                    },
                    {
                        "id": "T3",
                        "question": "방금 게임 자세히",
                        "expected": {"mode": "research"},
                        "forbidden": {},
                    },
                ],
            }
        )

        class ReplacingRuntime:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def ask(self, question: str, **kwargs: Any) -> dict[str, Any]:
                self.calls.append({"question": question, **kwargs})
                call_number = len(self.calls)
                if call_number == 1:
                    return {
                        "mode": "recommendation",
                        "answer": "첫 추천",
                        "games": [{"appid": 10, "name": "Old Game"}],
                    }
                if call_number == 2:
                    return {
                        "mode": "recommendation",
                        "answer": "새 추천",
                        "games": [{"appid": 20, "name": "New Game"}],
                    }
                return {
                    "mode": "research",
                    "answer": "새 게임 상세",
                    "games": [],
                }

        runtime = ReplacingRuntime()
        ServiceConversationBenchmarkRunner(runtime).run([case])

        self.assertEqual(runtime.calls[1]["context_games"], [{"appid": 10, "name": "Old Game"}])
        self.assertEqual(runtime.calls[2]["context_games"], [{"appid": 20, "name": "New Game"}])

    def test_errors_are_scored_without_aborting_the_remaining_scenario(self) -> None:
        runtime = FakeConversationRuntime(fail_on_call=2)
        records = ServiceConversationBenchmarkRunner(runtime).run([_case()])
        summary = summarize_conversation_records(records)

        self.assertEqual(len(records), 2)
        self.assertIn("RuntimeError", records[1].error)
        self.assertEqual(records[1].metrics["error_rate"], 1.0)
        self.assertEqual(records[1].metrics["contract_pass"], 0.0)
        self.assertEqual(summary["metrics"]["error_rate"], 0.5)
        self.assertEqual(summary["case_count"], 1)

    def test_forbidden_content_and_appids_are_reported_as_leakage(self) -> None:
        case = _case()

        class LeakyRuntime(FakeConversationRuntime):
            def ask(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                payload = super().ask(*args, **kwargs)
                payload["answer"] = "Hades II는 싱글플레이 전용입니다."
                payload["games"] = [{"appid": 1145350, "name": "Hades II"}]
                return payload

        records = ServiceConversationBenchmarkRunner(LeakyRuntime()).run([case])

        self.assertGreater(records[0].metrics["forbidden_keyword_leakage"] or 0.0, 0.0)
        self.assertEqual(records[1].metrics["forbidden_appid_leakage"], 1.0)
        self.assertTrue(all(record.metrics["contract_pass"] == 0.0 for record in records))

    def test_results_save_as_jsonl_and_machine_readable_summary(self) -> None:
        records = ServiceConversationBenchmarkRunner(FakeConversationRuntime()).run([_case()])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = save_conversation_benchmark(
                records,
                details_path=root / "details.jsonl",
                summary_path=root / "summary.json",
            )

            self.assertEqual(summary["metric_family"], "deterministic_contract_heuristics")
            self.assertEqual(len((root / "details.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            self.assertIn('"p95"', (root / "summary.json").read_text(encoding="utf-8"))

    def test_loader_rejects_missing_explicit_forbidden_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden contract"):
            ConversationCase.from_dict(
                {
                    "id": "BAD",
                    "category": "comparison",
                    "title": "invalid",
                    "turns": [
                        {"id": "T1", "question": "one", "expected": {"mode": "research"}},
                        {
                            "id": "T2",
                            "question": "two",
                            "expected": {"mode": "research"},
                            "forbidden": {},
                        },
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
