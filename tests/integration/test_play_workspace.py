from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import patch

import _bootstrap  # noqa: F401
from fastapi.testclient import TestClient

from steam_rag.api.service_app import create_service_app
from steam_rag.application.service_runtime import ServicePaths, SteamServiceRuntime
from steam_rag.common.models import Document, SearchResult


EXPERT_DIR = Path(__file__).resolve().parents[2] / "data" / "game_experts"
HOLLOW_KNIGHT = 367520


class FakeEmbedder:
    model_name = "fake"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class RecordingGenerator:
    def __init__(self) -> None:
        self.expert_context: dict = {}

    def generate(self, question, results):
        return "기본 답변"

    def generate_expert_answer(self, question, results, context):
        self.expert_context = context
        titles = ", ".join(
            str(item.document.metadata.get("item_title")) for item in results
        )
        return f"### 현재 상황 진단\n{titles} 근거로 답합니다. [근거 1]"


def evidence(title: str, content: str, *, section: str = "gameplay") -> SearchResult:
    return SearchResult(
        Document(
            content,
            {
                "appid": HOLLOW_KNIGHT,
                "game_name": "Hollow Knight",
                "section": section,
                "item_title": title,
                "source_type": "steam_corpus",
                "source_date": "2026-01-01",
            },
        ),
        score=1.0,
    )


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def build_search_spec(self, question: str):
        return None

    def retrieve(self, question, embedding, *, k=5, search_spec=None, allowed_appids=()):
        self.calls.append({"question": question, "allowed_appids": list(allowed_appids)})
        return list(self.results)


def fake_pipeline(results: list[SearchResult]) -> SimpleNamespace:
    return SimpleNamespace(
        index=SimpleNamespace(documents=[Document("hk", {"appid": HOLLOW_KNIGHT})]),
        retriever=FakeRetriever(results),
    )


class PlayWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        self.paths = ServicePaths(
            docs_dir=root / "docs",
            index_path=root / "index",
            raw_dir=root / "raw",
            catalog_path=root / "catalog.json",
            profiles_dir=root / "profiles",
            service_db=root / "service.db",
            time_analysis_dir=root / "time",
            workspace_db=root / "workspace.db",
            expert_dir=EXPERT_DIR,
        )
        self.paths.index_path.mkdir(parents=True)
        self.runtime = SteamServiceRuntime(paths=self.paths, enable_reranker=False)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _ask_play(
        self,
        question: str,
        *,
        results: list[SearchResult],
        generator: RecordingGenerator | None = None,
        thread_id: str = "",
    ) -> dict:
        generator = generator or RecordingGenerator()
        pipeline = fake_pipeline(results)
        with (
            patch(
                "steam_rag.application.service_runtime.OpenAIEmbedder",
                return_value=FakeEmbedder(),
            ),
            patch(
                "steam_rag.application.service_runtime.OpenAIAnswerGenerator",
                return_value=generator,
            ),
            patch("steam_rag.application.service_runtime.SteamAPIClient"),
            patch("steam_rag.application.service_runtime.OnDemandCorpusManager") as manager_type,
            patch(
                "steam_rag.application.service_runtime.RAGPipeline.from_path",
                return_value=pipeline,
            ),
        ):
            manager_type.return_value.ensure_questions.return_value = []
            payload = self.runtime.ask(
                question,
                workspace="play",
                user_id="tester",
                game_id=HOLLOW_KNIGHT,
                thread_id=thread_id,
            )
        payload["_pipeline"] = pipeline
        payload["_generator"] = generator
        return payload

    def test_play_request_uses_the_game_expert_and_locks_retrieval_to_one_game(self) -> None:
        payload = self._ask_play(
            "부적은 어떻게 바꿔?", results=[evidence("부적 안내", "부적은 벤치에서 교체한다.")]
        )

        self.assertEqual(payload["mode"], "play")
        self.assertEqual(payload["workspace"], "play")
        self.assertEqual(payload["_pipeline"].retriever.calls[0]["allowed_appids"], [HOLLOW_KNIGHT])
        self.assertEqual(payload["budget"]["expert_calls"], 1)
        self.assertEqual(payload["expert"]["applied_scope"]["appid"], HOLLOW_KNIGHT)
        self.assertTrue(payload["expert"]["scope"]["supported"])

    def test_play_conversation_is_stored_per_thread_and_replayed(self) -> None:
        first = self._ask_play("소울은 어떻게 모아?", results=[evidence("소울", "적을 때려 모은다.")])
        thread_id = first["thread"]["thread_id"]
        self._ask_play(
            "부적은?", results=[evidence("부적", "벤치에서 교체")], thread_id=thread_id
        )

        messages = self.runtime.play_thread_messages("tester", thread_id)
        other_thread = self.runtime.open_play_thread("tester", appid=HOLLOW_KNIGHT, topic="build")

        self.assertEqual([item["role"] for item in messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual(messages[0]["content"], "소울은 어떻게 모아?")
        self.assertEqual(self.runtime.play_thread_messages("tester", other_thread["thread_id"]), [])

    def test_discovery_conversation_never_enters_the_play_context(self) -> None:
        workspace = self.runtime.workspace
        session = workspace.create_discovery_session("tester")
        workspace.append_discovery_message(
            "tester", session.session_id, role="user", content="이번 예산은 3만원이야"
        )
        generator = RecordingGenerator()

        self._ask_play(
            "초반에 뭐부터 해야 해?",
            results=[evidence("초반 안내", "교차로부터 시작한다.")],
            generator=generator,
        )
        transcript = str(generator.expert_context)

        self.assertNotIn("예산", transcript)
        self.assertEqual(generator.expert_context["thread_messages"], [])

    def test_spoiler_setting_filters_the_documents_before_the_answer(self) -> None:
        self.runtime.update_game_state(
            "tester", HOLLOW_KNIGHT, progress="그린패스", spoiler_level="progress"
        )
        generator = RecordingGenerator()

        payload = self._ask_play(
            "여기서 뭘 해야 해?",
            results=[
                evidence("그린패스 안내", "그린패스에서 대시를 얻는다."),
                evidence("눈물의 도시", "눈물의 도시에서 벌어지는 전개", section="story"),
            ],
            generator=generator,
        )

        used = [item["title"] for item in payload["sources"]]
        self.assertEqual(used, ["그린패스 안내"])
        self.assertIn("스포일러 범위 밖 자료 1건은 사용하지 않았습니다.", payload["expert"]["unverified"])
        self.assertIn("그린패스", payload["expert"]["spoiler"]["notice"])

    def test_story_question_without_progress_asks_instead_of_answering(self) -> None:
        payload = self._ask_play(
            "이 캐릭터의 정체가 뭐야?", results=[evidence("이야기", "스토리", section="story")]
        )

        self.assertIn("진행 구간을 먼저 확인", payload["answer"])
        self.assertEqual(payload["sources"], [])
        self.assertTrue(payload["expert"]["follow_up_questions"])

    def test_retry_message_is_recorded_as_a_failed_attempt(self) -> None:
        self.runtime.update_game_state("tester", HOLLOW_KNIGHT, progress="그린패스")
        self._ask_play(
            "알려준 대로 했는데 안 됐어", results=[evidence("보스", "패턴 설명")]
        )

        attempts = self.runtime.game_state("tester", HOLLOW_KNIGHT)["attempts"]

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["outcome"], "사용자가 효과 없었다고 보고")

    def test_unindexed_game_reports_missing_documents_instead_of_guessing(self) -> None:
        pipeline = SimpleNamespace(
            index=SimpleNamespace(documents=[Document("other", {"appid": 999})]),
            retriever=FakeRetriever([]),
        )
        with (
            patch(
                "steam_rag.application.service_runtime.OpenAIEmbedder",
                return_value=FakeEmbedder(),
            ),
            patch(
                "steam_rag.application.service_runtime.OpenAIAnswerGenerator",
                return_value=RecordingGenerator(),
            ),
            patch("steam_rag.application.service_runtime.SteamAPIClient"),
            patch("steam_rag.application.service_runtime.OnDemandCorpusManager") as manager_type,
            patch(
                "steam_rag.application.service_runtime.RAGPipeline.from_path",
                return_value=pipeline,
            ),
        ):
            manager_type.return_value.ensure_questions.return_value = []
            with self.assertRaises(LookupError):
                self.runtime.ask(
                    "부적은?", workspace="play", user_id="tester", game_id=HOLLOW_KNIGHT
                )


class WorkspaceApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        paths = ServicePaths(
            docs_dir=root / "docs",
            index_path=root / "index",
            raw_dir=root / "raw",
            catalog_path=root / "catalog.json",
            profiles_dir=root / "profiles",
            service_db=root / "service.db",
            time_analysis_dir=root / "time",
            workspace_db=root / "workspace.db",
            expert_dir=EXPERT_DIR,
        )
        self.runtime = SteamServiceRuntime(paths=paths, enable_reranker=False)
        self.client = TestClient(create_service_app(self.runtime))

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_library_add_list_and_remove(self) -> None:
        saved = self.client.post(
            "/api/library",
            json={"user_id": "tester", "appid": HOLLOW_KNIGHT, "name": "Hollow Knight"},
        )
        listed = self.client.get("/api/library?user_id=tester")
        removed = self.client.delete(f"/api/library/{HOLLOW_KNIGHT}?user_id=tester")
        empty = self.client.get("/api/library?user_id=tester")

        self.assertEqual(saved.status_code, 200)
        self.assertEqual([game["appid"] for game in listed.json()["games"]], [HOLLOW_KNIGHT])
        self.assertTrue(removed.json()["removed"])
        self.assertEqual(empty.json()["games"], [])

    def test_preferences_keep_evidence_and_can_be_deleted(self) -> None:
        created = self.client.post(
            "/api/preferences",
            json={
                "user_id": "tester",
                "kind": "dislike",
                "value": "turn_based",
                "label": "턴제 전투",
                "evidence": "턴제보다 직접 움직이는 게 좋아",
            },
        )
        listed = self.client.get("/api/preferences?user_id=tester").json()["preferences"]
        deleted = self.client.delete(
            f"/api/preferences/{created.json()['preference_id']}?user_id=tester"
        )

        self.assertEqual(listed[0]["evidence"], "턴제보다 직접 움직이는 게 좋아")
        self.assertTrue(deleted.json()["removed"])
        self.assertEqual(self.client.get("/api/preferences?user_id=tester").json()["preferences"], [])

    def test_play_space_handoff_creates_a_thread_and_reports_support_scope(self) -> None:
        response = self.client.post(
            "/api/play-space",
            json={"user_id": "tester", "appid": HOLLOW_KNIGHT, "name": "Hollow Knight"},
        )
        payload = response.json()

        self.assertTrue(payload["expert_verified"])
        self.assertIn("핵심 시스템 설명", payload["support"]["topic_labels"])
        self.assertEqual(len(payload["threads"]), 1)
        self.assertEqual(payload["game_state"]["spoiler_level"], "no_spoiler")

    def test_game_state_round_trip_and_new_playthrough(self) -> None:
        self.client.put(
            f"/api/games/{HOLLOW_KNIGHT}/state",
            json={
                "user_id": "tester",
                "appid": HOLLOW_KNIGHT,
                "progress": "그린패스",
                "equipment": ["기본 못"],
                "spoiler_level": "progress",
            },
        )
        state = self.client.get(f"/api/games/{HOLLOW_KNIGHT}/state?user_id=tester").json()
        second = self.client.post(f"/api/games/{HOLLOW_KNIGHT}/playthrough?user_id=tester").json()

        self.assertEqual(state["progress"], "그린패스")
        self.assertEqual(state["equipment"], ["기본 못"])
        self.assertEqual(second["playthrough"], 2)
        self.assertEqual(second["progress"], "")
        first_again = self.client.get(f"/api/games/{HOLLOW_KNIGHT}/state?user_id=tester").json()
        self.assertEqual(first_again["progress"], "그린패스")

    def test_state_path_and_body_appid_must_match(self) -> None:
        response = self.client.put(
            f"/api/games/{HOLLOW_KNIGHT}/state",
            json={"user_id": "tester", "appid": 999, "progress": "x"},
        )

        self.assertEqual(response.status_code, 400)

    def test_compare_endpoint_returns_axes_and_missing_profiles(self) -> None:
        response = self.client.post("/api/compare", json={"appids": [1, 2]})
        payload = response.json()

        self.assertEqual(payload["mode"], "comparison")
        self.assertEqual(payload["missing_appids"], [1, 2])
        self.assertEqual(payload["comparison"]["games"], [])

    def test_health_reports_the_supported_expert_games(self) -> None:
        payload = self.client.get("/api/health").json()

        self.assertEqual(len(payload["supported_experts"]), 3)
        self.assertIn(HOLLOW_KNIGHT, [item["appid"] for item in payload["supported_experts"]])


if __name__ == "__main__":
    unittest.main()
