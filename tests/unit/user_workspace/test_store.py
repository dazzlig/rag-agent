from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from steam_rag.user_workspace.store import DEFAULT_SPOILER_LEVEL, WorkspaceStore


class WorkspaceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.store = WorkspaceStore(Path(self._directory.name) / "workspace.db")

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_discovery_and_play_conversations_use_separate_keys(self) -> None:
        session = self.store.create_discovery_session("user_a", title="탐색")
        self.store.append_discovery_message(
            "user_a", session.session_id, role="user", content="예쁜 액션 게임 찾아줘"
        )
        thread = self.store.open_play_thread("user_a", appid=367520, topic="boss")
        self.store.append_play_message(
            "user_a", thread.thread_id, appid=367520, role="user", content="이 보스에서 막혔어"
        )

        context = self.store.play_context(
            "user_a", appid=367520, thread_id=thread.thread_id
        )
        discovery = self.store.recent_discovery_messages("user_a", session.session_id)

        self.assertEqual([item["content"] for item in context["messages"]], ["이 보스에서 막혔어"])
        self.assertEqual([item["content"] for item in discovery], ["예쁜 액션 게임 찾아줘"])
        self.assertNotIn("messages", set(context) & {"discovery_messages"})

    def test_other_threads_of_the_same_game_are_not_auto_attached(self) -> None:
        boss = self.store.open_play_thread("user_a", appid=367520, topic="boss")
        build = self.store.open_play_thread("user_a", appid=367520, topic="build")
        self.store.append_play_message(
            "user_a", boss.thread_id, appid=367520, role="user", content="보스 질문"
        )
        self.store.append_play_message(
            "user_a", build.thread_id, appid=367520, role="user", content="빌드 질문"
        )

        context = self.store.play_context("user_a", appid=367520, thread_id=build.thread_id)

        self.assertEqual([item["content"] for item in context["messages"]], ["빌드 질문"])

    def test_state_is_scoped_by_user_game_and_playthrough(self) -> None:
        self.store.update_game_state("user_a", 367520, progress="그린패스")
        self.store.update_game_state("user_b", 367520, progress="눈물의 도시")
        second_run = self.store.next_playthrough("user_a", 367520)
        self.store.update_game_state("user_a", 367520, playthrough=second_run, progress="교차로")

        first = self.store.get_game_state("user_a", 367520)
        other_user = self.store.get_game_state("user_b", 367520)
        new_run = self.store.get_game_state("user_a", 367520, playthrough=second_run)

        self.assertEqual(second_run, 2)
        self.assertEqual(first.progress, "그린패스")
        self.assertEqual(other_user.progress, "눈물의 도시")
        self.assertEqual(new_run.progress, "교차로")

    def test_partial_state_update_keeps_untouched_fields(self) -> None:
        self.store.update_game_state(
            "user_a", 367520, progress="그린패스", equipment=["기본 못"], spoiler_level="progress"
        )
        updated = self.store.update_game_state("user_a", 367520, character_build="주문 위주")

        self.assertEqual(updated.progress, "그린패스")
        self.assertEqual(updated.equipment, ["기본 못"])
        self.assertEqual(updated.spoiler_level, "progress")
        self.assertEqual(updated.character_build, "주문 위주")

    def test_spoiler_level_defaults_to_the_conservative_setting(self) -> None:
        state = self.store.get_game_state("user_a", 1086940)

        self.assertEqual(state.spoiler_level, DEFAULT_SPOILER_LEVEL)
        self.assertTrue(state.is_empty)

    def test_handoff_carries_only_the_game_and_platform(self) -> None:
        session = self.store.create_discovery_session("user_a")
        self.store.append_discovery_message(
            "user_a", session.session_id, role="user", content="예산은 3만원"
        )

        handoff = self.store.handoff_to_play_space(
            "user_a", appid=367520, name="Hollow Knight", platform="steam"
        )

        self.assertEqual(handoff["appid"], 367520)
        self.assertEqual(handoff["platform"], "steam")
        self.assertEqual(len(handoff["threads"]), 1)
        self.assertNotIn("discovery_session_id", handoff)
        self.assertEqual([game["appid"] for game in self.store.list_library("user_a")], [367520])

    def test_preferences_keep_evidence_and_can_be_deleted(self) -> None:
        saved = self.store.set_preference(
            "user_a",
            kind="dislike",
            value="turn_based",
            label="턴제 전투",
            evidence="턴제보다는 직접 움직이는 게 좋아",
        )
        session_only = self.store.set_preference(
            "user_a", kind="like", value="short_game", scope="session", session_id="disc_1"
        )
        rejected = self.store.set_preference("user_a", kind="unknown", value="x")

        persistent = self.store.list_preferences("user_a", scope="persistent")
        self.assertEqual([item.value for item in persistent], ["turn_based"])
        self.assertEqual(persistent[0].evidence, "턴제보다는 직접 움직이는 게 좋아")
        self.assertIsNotNone(session_only)
        self.assertIsNone(rejected)
        self.assertTrue(self.store.delete_preference("user_a", saved.preference_id))
        self.assertEqual(self.store.list_preferences("user_a", scope="persistent"), [])

    def test_session_preferences_require_a_session_id(self) -> None:
        self.assertIsNone(
            self.store.set_preference("user_a", kind="like", value="short_game", scope="session")
        )

    def test_attempts_are_recorded_per_playthrough(self) -> None:
        self.store.record_attempt("user_a", 367520, action="불 속성 무기 사용", outcome="실패")
        self.store.record_attempt("user_a", 367520, action="", outcome="무시됨")
        self.store.record_attempt("user_a", 367520, action="회피 연습", playthrough=2)

        first = self.store.list_attempts("user_a", 367520)
        second = self.store.list_attempts("user_a", 367520, playthrough=2)

        self.assertEqual([item.action for item in first], ["불 속성 무기 사용"])
        self.assertEqual([item.action for item in second], ["회피 연습"])

    def test_library_reports_progress_and_thread_count(self) -> None:
        self.store.add_library_game("user_a", appid=367520, name="Hollow Knight")
        self.store.update_game_state("user_a", 367520, progress="그린패스")
        self.store.open_play_thread("user_a", appid=367520, topic="boss")

        entry = self.store.list_library("user_a")[0]

        self.assertEqual(entry["progress"], "그린패스")
        self.assertEqual(entry["thread_count"], 1)
        self.assertTrue(self.store.remove_library_game("user_a", 367520))
        self.assertEqual(self.store.list_library("user_a"), [])


if __name__ == "__main__":
    unittest.main()
