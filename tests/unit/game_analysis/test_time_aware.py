from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from steam_rag.game_analysis.time_aware import (
    PatchEvent,
    analyze_patch_reviews,
    build_time_analysis_markdown,
    select_patch_event,
    structure_patch_events,
    upsert_time_analysis_markdown,
)
from steam_rag.steam_collection.markdown_documents import parse_markdown
from steam_rag.steam_collection.steam_client import ReviewRangeResult


def raw_review(day: str, voted_up: bool, text: str, recommendationid: str) -> dict:
    timestamp = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())
    return {
        "recommendationid": recommendationid,
        "timestamp_created": timestamp,
        "timestamp_updated": timestamp,
        "language": "koreana",
        "voted_up": voted_up,
        "review": text,
        "author": {},
    }


class FakeRangeClient:
    def __init__(self, reviews: list[dict]) -> None:
        self.reviews = reviews
        self.arguments: dict = {}

    def fetch_reviews_by_date_range(self, appid: int, **kwargs) -> ReviewRangeResult:
        self.arguments = {"appid": appid, **kwargs}
        return ReviewRangeResult(self.reviews, pages_fetched=3, reached_start=True, truncated=False)


class TimeAnalysisTests(unittest.TestCase):
    def test_patch_events_are_structured_and_major_update_is_selected(self) -> None:
        events = structure_patch_events(
            [
                {
                    "news_date": "2026-06-20",
                    "title": "Hotfix",
                    "contents": "Crash fix",
                    "news_type": "hotfix",
                },
                {
                    "news_date": "2026-06-10",
                    "title": "대규모 업데이트",
                    "contents": "신규 콘텐츠와 전투 밸런스 조정",
                    "news_type": "major_update",
                },
                {
                    "news_date": "2026-06-25",
                    "title": "One Year in Early Access!!",
                    "contents": "Looking back at our major update history.",
                    "news_type": "major_update",
                },
            ]
        )

        selected = select_patch_event(events)

        self.assertEqual(selected.event_type, "major_update")
        self.assertEqual(selected.importance, "high")
        self.assertIn("new_content", selected.affected_features)
        self.assertIn("combat_balance", selected.affected_features)
        self.assertNotIn("One Year in Early Access!!", [event.title for event in events])

    def test_latest_hotfix_wins_when_meaningful_update_is_too_old(self) -> None:
        selected = select_patch_event(
            [
                PatchEvent("2025-01-01", "Old Major", "major_update", "high", ["new_content"]),
                PatchEvent("2026-06-10", "Latest Hotfix", "hotfix", "low", ["bugs"]),
            ]
        )

        self.assertEqual(selected.title, "Latest Hotfix")

    def test_reviews_are_split_and_change_confidence_is_calculated(self) -> None:
        reviews = []
        for index in range(60):
            reviews.append(
                raw_review(
                    f"2026-05-{index % 20 + 1:02d}",
                    index < 30,
                    "성능 끊김과 크래시 문제" if index >= 30 else "스토리가 좋음",
                    f"before-{index}",
                )
            )
        for index in range(60):
            reviews.append(
                raw_review(
                    f"2026-06-{index % 20 + 1:02d}",
                    index < 54,
                    "성능 최적화와 전투 밸런스가 좋아짐" if index < 54 else "버그가 남음",
                    f"after-{index}",
                )
            )
        client = FakeRangeClient(reviews)
        event = PatchEvent(
            date="2026-06-01",
            title="Major Update",
            event_type="major_update",
            importance="high",
            affected_features=["performance"],
        )

        analysis = analyze_patch_reviews(
            client,
            appid=42,
            game_name="Example Game",
            patch_event=event,
            before_days=31,
            after_days=30,
            today=date(2026, 7, 1),
        )

        self.assertEqual(analysis.before.sample_size, 60)
        self.assertEqual(analysis.after.sample_size, 60)
        self.assertEqual(analysis.before.positive_ratio, 0.5)
        self.assertEqual(analysis.after.positive_ratio, 0.9)
        self.assertEqual(analysis.positive_ratio_delta_pp, 40.0)
        self.assertEqual(analysis.direction, "improved")
        self.assertEqual(analysis.confidence_label, "high")
        self.assertTrue(any(row["topic"] == "performance" for row in analysis.after.strengths))
        self.assertEqual(client.arguments["start_date"], date(2026, 5, 1))

    def test_analysis_markdown_is_replaced_instead_of_duplicated(self) -> None:
        client = FakeRangeClient(
            [
                raw_review("2026-05-20", True, "전투 좋음", "1"),
                raw_review("2026-06-10", True, "전투 좋음", "2"),
            ]
        )
        analysis = analyze_patch_reviews(
            client,
            appid=42,
            game_name="Example",
            patch_event=PatchEvent("2026-06-01", "Patch", "patch_note", "medium", ["bugs"]),
            today=date(2026, 7, 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.md"
            path.write_text("# Example\n\n## Metadata\n\n- appid: 42\n", encoding="utf-8")
            upsert_time_analysis_markdown(path, analysis)
            upsert_time_analysis_markdown(path, analysis)
            text = path.read_text(encoding="utf-8")
            documents = parse_markdown(path)

        self.assertEqual(text.count("## Patch Impact Analysis"), 1)
        self.assertIn("positive_ratio_delta_pp", build_time_analysis_markdown(analysis))
        analysis_document = next(doc for doc in documents if doc.metadata["section"] == "analysis")
        self.assertEqual(analysis_document.metadata["source_date"], "2026-06-01")
        self.assertEqual(analysis_document.metadata["change_direction"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
