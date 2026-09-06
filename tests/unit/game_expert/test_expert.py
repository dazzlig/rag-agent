from __future__ import annotations

import unittest
from typing import Sequence

import _bootstrap  # noqa: F401
from steam_rag.common.models import Document, SearchResult
from steam_rag.game_expert.expert import GameExpertAgent
from steam_rag.game_expert.support_scope import build_expert_profile
from steam_rag.user_workspace.store import GameState


PROFILE = build_expert_profile(
    {
        "appid": 367520,
        "name": "Hollow Knight",
        "aliases": ["할로우 나이트"],
        "key_systems": [{"name": "부적", "summary": "노치 한도 안에서 조합한다."}],
        "milestones": [
            {"order": 1, "milestone_id": "crossroads", "label": "교차로", "story_sensitive": False},
            {"order": 2, "milestone_id": "greenpath", "label": "그린패스", "story_sensitive": False},
            {
                "order": 3,
                "milestone_id": "city",
                "label": "눈물의 도시",
                "keywords": ["눈물의 도시"],
                "story_sensitive": True,
            },
        ],
        "support": {"topics": ["system", "boss"], "verified_version": "1.5.78"},
    }
)


class FakeEmbedder:
    model_name = "fake"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def build_search_spec(self, question: str):
        return None

    def retrieve(self, question, embedding, *, k=5, search_spec=None, allowed_appids=()):
        self.calls.append({"question": question, "allowed_appids": list(allowed_appids)})
        return list(self.results)


class FakeGenerator:
    def __init__(self, answer: str = "생성된 답변 [근거 1]", *, fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.context: dict = {}

    def generate(self, question, results):
        return self.answer

    def generate_expert_answer(self, question, results, context):
        self.context = context
        if self.fail:
            raise RuntimeError("API 오류")
        return self.answer


def evidence(title: str, content: str, *, section: str = "gameplay") -> SearchResult:
    return SearchResult(
        Document(
            content,
            {
                "section": section,
                "item_title": title,
                "source_type": "steam_corpus",
                "appid": 367520,
                "game_name": "Hollow Knight",
            },
        ),
        score=1.0,
    )


def agent(results: list[SearchResult], generator: FakeGenerator | None = None) -> GameExpertAgent:
    return GameExpertAgent(
        PROFILE, FakeRetriever(results), FakeEmbedder(), generator or FakeGenerator()
    )


class GameExpertAgentTests(unittest.TestCase):
    def test_retrieval_is_locked_to_this_game(self) -> None:
        retriever = FakeRetriever([evidence("부적 안내", "부적은 벤치에서 교체한다.")])
        expert = GameExpertAgent(PROFILE, retriever, FakeEmbedder(), FakeGenerator())

        expert.answer("부적은 어떻게 바꿔?", state=GameState("u", 367520, 1))

        self.assertEqual(retriever.calls[0]["allowed_appids"], [367520])

    def test_out_of_scope_topic_is_flagged_but_still_answered_with_evidence(self) -> None:
        result = agent([evidence("패치 노트", "이번 패치 내용")]).answer(
            "이번 업데이트로 뭐가 바뀌었어?", state=GameState("u", 367520, 1)
        )

        self.assertFalse(result.scope.supported)
        self.assertIn(result.scope.reason, result.unverified)
        self.assertEqual(result.applied_scope["verified_version"], "1.5.78")

    def test_story_question_without_progress_asks_before_searching(self) -> None:
        retriever = FakeRetriever([evidence("이야기", "스토리 문서", section="story")])
        expert = GameExpertAgent(PROFILE, retriever, FakeEmbedder(), FakeGenerator())

        result = expert.answer("이 캐릭터의 정체가 뭐야?", state=GameState("u", 367520, 1))

        self.assertEqual(retriever.calls, [])
        self.assertIn("진행 구간을 먼저 확인", result.answer)
        self.assertTrue(result.follow_up_questions)

    def test_spoiler_filter_removes_later_documents_and_reports_them(self) -> None:
        state = GameState("u", 367520, 1, progress="그린패스", spoiler_level="progress")
        result = agent(
            [
                evidence("그린패스 안내", "그린패스 이동"),
                evidence("눈물의 도시", "눈물의 도시 전개"),
            ]
        ).answer("여기서 뭘 해야 해?", state=state)

        titles = [item.document.metadata["item_title"] for item in result.evidence]
        self.assertEqual(titles, ["그린패스 안내"])
        self.assertIn("스포일러 범위 밖 자료 1건은 사용하지 않았습니다.", result.unverified)

    def test_only_topic_relevant_state_is_requested(self) -> None:
        empty = GameState("u", 367520, 1)
        system = agent([evidence("시스템", "설명")]).answer("소울은 어떻게 모아?", state=empty)
        boss = agent([evidence("보스", "패턴")]).answer("이 보스에서 막혔어", state=empty)

        self.assertEqual(system.follow_up_questions, [])
        self.assertEqual(len(boss.follow_up_questions), 2)

    def test_retry_message_changes_the_advice_instead_of_repeating_it(self) -> None:
        generator = FakeGenerator()
        state = GameState("u", 367520, 1, progress="그린패스")
        result = agent([evidence("보스", "패턴")], generator).answer(
            "알려준 대로 했는데 안 됐어",
            state=state,
            attempts=[{"action": "불 속성 무기", "outcome": "실패"}],
        )

        self.assertTrue(result.is_retry)
        self.assertTrue(generator.context["is_retry"])
        self.assertIn("장비", generator.context["retry_assumptions"])
        self.assertEqual(
            [item["field"] for item in result.state_change_proposals], ["attempt"]
        )

    def test_progress_mentioned_in_the_question_is_proposed_not_saved(self) -> None:
        result = agent([evidence("보스", "패턴")]).answer(
            "지금 그린패스에서 막혔어",
            state=GameState("u", 367520, 1, progress="교차로", equipment=["기본 못"]),
        )
        proposal = result.state_change_proposals[0]

        self.assertEqual(proposal["field"], "progress")
        self.assertIn("확인 후 저장", proposal["reason"])

    def test_generator_failure_falls_back_to_the_deterministic_order(self) -> None:
        result = agent(
            [evidence("부적 안내", "부적은 벤치에서 교체한다.")],
            FakeGenerator(fail=True),
        ).answer("부적은 어떻게 바꿔?", state=GameState("u", 367520, 1, progress="그린패스"))

        self.assertLess(result.answer.index("현재 상황 진단"), result.answer.index("바로 시도할 행동"))
        self.assertLess(result.answer.index("바로 시도할 행동"), result.answer.index("필요한 이유"))
        self.assertLess(result.answer.index("필요한 이유"), result.answer.index("추가 힌트"))

    def test_no_evidence_returns_a_refusal_instead_of_a_guess(self) -> None:
        result = agent([]).answer("이 보스 어떻게 잡아?", state=GameState("u", 367520, 1))

        self.assertIn("검증된 자료를 찾지 못했습니다", result.answer)
        self.assertIn("근거 없이 추측한 공략은 제공하지 않았습니다.", result.answer)
        self.assertEqual(result.evidence, [])

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            agent([]).answer("  ", state=GameState("u", 367520, 1))


if __name__ == "__main__":
    unittest.main()
