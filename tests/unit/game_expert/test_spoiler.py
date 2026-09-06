from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401
from steam_rag.common.models import Document, SearchResult
from steam_rag.game_expert.spoiler import (
    build_spoiler_policy,
    filter_results,
    needs_progress_confirmation,
    redact_leaks,
    spoiler_notice,
)
from steam_rag.game_expert.support_scope import build_expert_profile


PROFILE = build_expert_profile(
    {
        "appid": 367520,
        "name": "Hollow Knight",
        "milestones": [
            {
                "order": 1,
                "milestone_id": "crossroads",
                "label": "교차로",
                "keywords": ["교차로"],
                "story_sensitive": False,
            },
            {
                "order": 2,
                "milestone_id": "greenpath",
                "label": "그린패스",
                "keywords": ["그린패스"],
                "story_sensitive": False,
            },
            {
                "order": 3,
                "milestone_id": "city",
                "label": "눈물의 도시",
                "keywords": ["눈물의 도시"],
                "story_sensitive": True,
            },
        ],
    }
)


def result(title: str, content: str, *, section: str = "gameplay", **metadata) -> SearchResult:
    payload = {"section": section, "item_title": title, "source_type": "steam_corpus"}
    payload.update(metadata)
    return SearchResult(Document(content, payload), score=1.0)


class SpoilerPolicyTests(unittest.TestCase):
    def test_progress_text_resolves_to_the_furthest_milestone(self) -> None:
        policy = build_spoiler_policy(PROFILE, spoiler_level="progress", progress="그린패스까지 왔어")

        self.assertEqual(policy.progress_order, 2)
        self.assertEqual(policy.progress_label, "그린패스")
        self.assertTrue(policy.progress_known)

    def test_documents_beyond_current_progress_are_removed_from_retrieval(self) -> None:
        policy = build_spoiler_policy(PROFILE, spoiler_level="progress", progress="그린패스")
        allowed, blocked = filter_results(
            policy,
            [
                result("그린패스 안내", "그린패스 이동 요령"),
                result("눈물의 도시 진입", "눈물의 도시에서 벌어지는 일"),
            ],
        )

        self.assertEqual([item.document.metadata["item_title"] for item in allowed], ["그린패스 안내"])
        self.assertEqual(allowed[0].rank, 1)
        self.assertEqual(blocked[0]["milestone"], "눈물의 도시")
        self.assertIn("이후 내용", blocked[0]["reason"])

    def test_no_spoiler_level_blocks_story_sensitive_material_entirely(self) -> None:
        policy = build_spoiler_policy(PROFILE, spoiler_level="no_spoiler", progress="눈물의 도시")
        allowed, blocked = filter_results(
            policy,
            [
                result("교차로 안내", "교차로 길찾기"),
                result("눈물의 도시", "눈물의 도시 이야기"),
            ],
        )

        self.assertEqual(len(allowed), 1)
        self.assertEqual(len(blocked), 1)

    def test_unlabeled_walkthrough_material_is_not_used_for_detailed_answers(self) -> None:
        policy = build_spoiler_policy(PROFILE, spoiler_level="progress", progress="그린패스")
        _, blocked = filter_results(
            policy,
            [
                result(
                    "커뮤니티 공략",
                    "임의의 공략 문서",
                    section="walkthrough",
                    source_type="community_wiki",
                )
            ],
        )

        self.assertIn("스포일러 구분이 확인되지 않은", blocked[0]["reason"])

    def test_all_level_allows_everything(self) -> None:
        policy = build_spoiler_policy(PROFILE, spoiler_level="all", progress="")
        allowed, blocked = filter_results(
            policy,
            [result("눈물의 도시", "후반 내용", section="story", source_type="community_wiki")],
        )

        self.assertEqual(len(allowed), 1)
        self.assertEqual(blocked, [])
        self.assertEqual(spoiler_notice(policy), "스포일러 제한 없이 답했습니다.")

    def test_story_question_with_unknown_progress_asks_first(self) -> None:
        unknown = build_spoiler_policy(PROFILE, spoiler_level="no_spoiler", progress="")
        known = build_spoiler_policy(PROFILE, spoiler_level="progress", progress="그린패스")

        self.assertTrue(needs_progress_confirmation("이 캐릭터의 정체가 뭐야?", unknown))
        self.assertFalse(needs_progress_confirmation("점프 조작은 어떻게 해?", unknown))
        self.assertFalse(needs_progress_confirmation("스토리가 어떻게 이어져?", known))

    def test_generated_answer_wording_is_screened_too(self) -> None:
        policy = build_spoiler_policy(PROFILE, spoiler_level="progress", progress="그린패스")
        safe, no_leaks = redact_leaks("그린패스에서 대시를 먼저 얻으세요.", policy)
        risky, leaks = redact_leaks("눈물의 도시에서 벌어지는 사건이 핵심입니다.", policy)

        self.assertEqual(no_leaks, [])
        self.assertEqual(safe, "그린패스에서 대시를 먼저 얻으세요.")
        self.assertEqual(leaks, ["눈물의 도시"])
        self.assertIn("허용한 스포일러 범위를 넘는 내용", risky)

    def test_documents_declared_as_late_spoilers_are_blocked(self) -> None:
        policy = build_spoiler_policy(PROFILE, spoiler_level="progress", progress="그린패스")
        _, blocked = filter_results(
            policy, [result("무관한 제목", "내용", spoiler_level="ending")]
        )

        self.assertIn("후반 스포일러", blocked[0]["reason"])


if __name__ == "__main__":
    unittest.main()
