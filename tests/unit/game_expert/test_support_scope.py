from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from steam_rag.game_expert.support_scope import (
    GameExpertRegistry,
    build_expert_profile,
    classify_topic,
)


EXPERT_DIR = Path(__file__).resolve().parents[3] / "data" / "game_experts"


def payload(**overrides) -> dict:
    value = {
        "appid": 367520,
        "game_key": "hollow_knight",
        "name": "Hollow Knight",
        "aliases": ["할로우 나이트"],
        "platforms": ["steam"],
        "key_systems": [{"name": "부적", "summary": "노치 한도 안에서 조합한다."}],
        "milestones": [
            {"order": 2, "milestone_id": "greenpath", "label": "그린패스", "keywords": ["그린패스"]},
            {"order": 1, "milestone_id": "crossroads", "label": "교차로", "story_sensitive": False},
        ],
        "support": {"topics": ["system", "boss"], "verified_version": "1.5.78"},
    }
    value.update(overrides)
    return value


class SupportScopeTests(unittest.TestCase):
    def test_shipped_expert_files_cover_three_question_types(self) -> None:
        registry = GameExpertRegistry.load(EXPERT_DIR)
        types = {
            json.loads(path.read_text(encoding="utf-8")).get("question_type")
            for path in EXPERT_DIR.glob("*.json")
        }

        self.assertEqual(len(registry.profiles), 3)
        self.assertEqual(
            types, {"조작 중심 액션", "서사 중심 진행", "장비와 성장 시스템"}
        )

    def test_profile_resolves_korean_aliases_and_appids(self) -> None:
        profile = build_expert_profile(payload())

        self.assertTrue(profile.matches_name("할로우 나이트 초반 공략"))
        self.assertTrue(profile.matches_name("appid 367520 질문"))
        self.assertFalse(profile.matches_name("발더스 게이트 3 빌드"))

    def test_milestones_are_ordered_and_matched_by_keyword(self) -> None:
        profile = build_expert_profile(payload())

        self.assertEqual([item.order for item in profile.milestones], [1, 2])
        self.assertEqual(profile.milestone_for("지금 그린패스야").milestone_id, "greenpath")
        self.assertIsNone(profile.milestone_for("아무 곳도 아님"))

    def test_out_of_scope_topic_is_reported_instead_of_silently_answered(self) -> None:
        profile = build_expert_profile(payload())
        supported = profile.decide_scope("boss")
        unsupported = profile.decide_scope("update")

        self.assertTrue(supported.supported)
        self.assertEqual(supported.verified_version, "1.5.78")
        self.assertFalse(unsupported.supported)
        self.assertIn("검증한 지원 범위가 아닙니다", unsupported.reason)
        self.assertIn("핵심 시스템 설명", unsupported.reason)

    def test_invalid_payloads_are_skipped(self) -> None:
        self.assertIsNone(build_expert_profile({"name": "No AppID"}))
        self.assertIsNone(build_expert_profile({"appid": 1}))

    def test_registry_ignores_unreadable_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.json").write_text("{ not json", encoding="utf-8")
            (root / "ok.json").write_text(json.dumps(payload()), encoding="utf-8")

            registry = GameExpertRegistry.load(root)

            self.assertEqual(registry.supported_appids(), [367520])
            self.assertEqual(registry.summary()[0]["verified_version"], "1.5.78")
            self.assertIsNone(registry.get("not-an-int"))

    def test_topic_classification_matches_the_play_space_topics(self) -> None:
        self.assertEqual(classify_topic("이 보스에서 막혔어"), "boss")
        self.assertEqual(classify_topic("장비를 바꿔야 할까"), "build")
        self.assertEqual(classify_topic("초반에 알아야 할 것만 알려줘"), "early_guide")
        self.assertEqual(classify_topic("이번 패치로 뭐가 바뀌었어"), "update")
        self.assertEqual(classify_topic("자원 파밍은 어떻게 해"), "progression")
        self.assertEqual(classify_topic("이 시스템은 어떻게 작동해"), "system")


if __name__ == "__main__":
    unittest.main()
