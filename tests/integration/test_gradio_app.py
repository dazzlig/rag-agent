from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from steam_rag.ui.gradio_app import (
    _dataframe_records,
    _result_to_rows,
    ask_ui,
    docs_table_rows,
    index_status,
    markdown_preview,
    recommend_candidates_ui,
    ragas_current_ui,
    search_ui,
    time_analysis_ui,
)
from steam_rag.common.models import Document
from steam_rag.rag_search.vector_store import VectorIndex


SAMPLE_MD = """# Example Game

## Metadata
- game_key: example_game
- appid: 42
- name: Example Game

## Store Summary
Short summary.
"""


class GradioAppTests(unittest.TestCase):
    def test_docs_table_and_preview_read_markdown_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            docs_dir = Path(directory)
            (docs_dir / "example_game.md").write_text(SAMPLE_MD, encoding="utf-8")

            self.assertEqual(docs_table_rows(docs_dir)[0][:3], ["example_game.md", "Example Game", "42"])
            preview = markdown_preview(docs_dir, "example_game.md")

        self.assertIn("Example Game", preview)
        self.assertIn("AppID", preview)

    def test_index_status_summarizes_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "example_game.md").write_text(SAMPLE_MD, encoding="utf-8")
            index_path = root / "index.json"
            VectorIndex(
                [
                    Document(
                        "Example content",
                        {
                            "game_name": "Example Game",
                            "section": "about",
                            "chunk_id": "example",
                        },
                    )
                ],
                [[1.0, 0.0]],
                "fake-embedding",
            ).save(index_path)

            status, payload = index_status(index_path, docs_dir)

        self.assertTrue(payload["exists"])
        self.assertIn("청크 수", status)
        self.assertEqual(payload["sections"], {"about": 1})

    def test_dataframe_records_accepts_plain_list_rows(self) -> None:
        records = _dataframe_records([["Question?", "Reference answer"]])
        self.assertEqual(records, [{"question": "Question?", "reference": "Reference answer"}])

    def test_empty_question_returns_ui_message_without_raising(self) -> None:
        search_result = search_ui(
            "",
            "data/docs",
            "data/index.json",
            "data/raw",
            "data/catalog.json",
            "text-embedding-3-small",
            5,
            False,
            24,
        )
        ask_result = ask_ui(
            "",
            "data/docs",
            "data/index.json",
            "data/raw",
            "data/catalog.json",
            "text-embedding-3-small",
            "gpt-5-mini",
            5,
            False,
            24,
            [],
        )

        self.assertEqual(search_result[0], "질문을 입력하세요.")
        self.assertEqual(ask_result[1], "질문을 입력하세요.")

    def test_ragas_before_answer_returns_ui_message(self) -> None:
        rows, message = ragas_current_ui(
            {},
            "",
            ["faithfulness"],
            "gpt-4o-mini",
            "text-embedding-3-small",
        )

        self.assertEqual(rows, [])
        self.assertIn("먼저", message)

    def test_ragas_result_uses_real_headers_and_truncates_long_text(self) -> None:
        frame = _result_to_rows(
            {
                "user_input": "Hollow Knight의 특징은?",
                "response": "A" * 200,
                "retrieved_contexts": ["B" * 100, "C" * 100],
                "reference": "",
                "faithfulness": 0.987654,
                "answer_relevancy": 0.876543,
            }
        )

        self.assertEqual(
            list(frame.columns),
            ["faithfulness", "answer_relevancy", "질문", "응답", "검색 컨텍스트", "기준 답변"],
        )
        self.assertEqual(frame.loc[0, "faithfulness"], 0.9877)
        self.assertTrue(frame.loc[0, "응답"].endswith("..."))
        self.assertTrue(frame.loc[0, "검색 컨텍스트"].endswith("..."))

    def test_recommendation_ui_returns_candidate_rows_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profiles = Path(directory)
            payload = {
                "appid": 42,
                "name": "Example RPG",
                "steam_genres_normalized": ["action", "rpg"],
                "steam_categories_normalized": [],
                "popular_user_tags": [
                    {"name": "Action RPG", "normalized": "action_rpg", "rank": 1},
                    {"name": "Third Person", "normalized": "third_person", "rank": 2},
                ],
                "combat_facets": ["real_time"],
                "perspective_facets": ["third_person"],
                "dimension_facets": ["3d"],
                "playstyle_facets": ["character_progression"],
                "inferred_facets": {},
                "recent_review_summary": {"positive_ratio": None},
                "price": {"currency": "KRW", "final": 30_000 * 100},
                "searchable_terms": [],
            }
            (profiles / "example.json").write_text(json.dumps(payload), encoding="utf-8")

            summary, rows, raw = recommend_candidates_ui(
                "3D 3인칭 액션 RPG 추천", str(profiles), "gpt-5-mini", False
            )

        self.assertIn("후보 **1개**", summary)
        self.assertEqual(rows[0][2], "Example RPG")
        self.assertEqual(json.loads(raw)["detail_targets"][0]["appid"], 42)

    def test_time_analysis_ui_rejects_missing_appid_before_api_calls(self) -> None:
        summary, rows, raw = time_analysis_ui(
            None,
            "Steam Game",
            "data/docs",
            "data/index.json",
            "data/raw",
            "data/catalog.json",
            "data/profiles",
            "data/time_analysis",
            "text-embedding-3-small",
            30,
            30,
            100,
        )

        self.assertIn("실패", summary)
        self.assertEqual(rows, [])
        self.assertEqual(raw, "{}")


if __name__ == "__main__":
    unittest.main()
