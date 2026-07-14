from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import patch

import _bootstrap  # noqa: F401
from fastapi.testclient import TestClient

from steam_rag.agents.agentic_rag import AgenticRAGCoordinator, GameResearchAgent
from steam_rag.agents.multi_agent_workflow import MultiAgentConfig, SteamMultiAgentWorkflow, _should_use_web_supplement
from steam_rag.api.service_app import create_service_app
from steam_rag.application.rag_pipeline import RAGPipeline
from steam_rag.application.service_runtime import (
    _index_appids_for_variants,
    _followup_relation,
    _resolve_candidate,
    _requires_verified_discovery_scope,
    _should_use_web_discovery,
    _recommendation_markdown,
    _steam_header_image,
    SteamServiceRuntime,
)
from steam_rag.common.models import Document, SearchResult
from steam_rag.external_apis.openai_client import OpenAIAnswerGenerator
from steam_rag.rag_search.hybrid_retriever import HybridTimeAwareRetriever
from steam_rag.rag_search.vector_store import build_index


class FakeEmbedder:
    model_name = "fake"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, float("combat" in text.casefold()), float(len(text) % 5)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeGenerator:
    def expand_search_queries(self, question: str) -> list[str]:
        return [question, "Hollow Knight combat", "할로우 나이트 전투"]

    def generate_hyde(self, question: str, search_query: str, reason: str) -> str:
        return "Hollow Knight combat exploration evidence"

    def generate(self, question: str, results: Sequence[SearchResult]) -> str:
        return "멀티 에이전트 답변 [근거 1]"

    def generate_agentic(
        self,
        question: str,
        results: Sequence[SearchResult],
        metadata: dict,
    ) -> str:
        return self.generate(question, results)


class FakeWebGenerator(FakeGenerator):
    def research_web_evidence(self, question: str, **kwargs) -> list[dict]:
        return [
            {
                "game": "Hollow Knight",
                "title": "Official game page",
                "snippet": "Official description of precise action combat.",
                "publisher": "Team Cherry",
                "published_at": "2025-01-01",
                "url": "https://example.com/hollow-knight",
                "source_type": "official",
            }
        ]


class FakeRuntime:
    def __init__(self) -> None:
        self.last_history: list[dict] = []
        self.last_context_games: list[dict] = []
        self.call_count = 0

    def health(self) -> dict:
        return {"status": "ready", "documents": 2, "chunks": 10, "workflow": "LangGraph"}

    def ask(
        self,
        question: str,
        *,
        top_k: int = 6,
        history: list[dict] | None = None,
        context_games: list[dict] | None = None,
    ) -> dict:
        self.call_count += 1
        self.last_history = list(history or [])
        self.last_context_games = list(context_games or [])
        return {
            "mode": "research",
            "answer": f"{question} 답변",
            "query_variants": [question],
            "agents": [{"agent": "Answer Agent", "status": "completed", "detail": "sources=1"}],
            "games": [],
            "sources": [],
            "evidence_coverage": {"coverage_ratio": 1.0},
        }


class EmptyCompletionClient:
    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])


class MultiAgentServiceTests(unittest.TestCase):
    def test_web_search_is_reserved_for_non_steam_candidate_discovery(self) -> None:
        standard = SimpleNamespace(upcoming_required=False, sale_required=False)
        sale = SimpleNamespace(upcoming_required=False, sale_required=True)
        upcoming = SimpleNamespace(upcoming_required=True, sale_required=False)

        self.assertFalse(_should_use_web_discovery("스토리 좋은 싱글 RPG 추천해줘", standard))
        self.assertFalse(_should_use_web_discovery("현재 할인 중인 게임 추천해줘", sale))
        self.assertFalse(_requires_verified_discovery_scope("현재 할인 중인 게임 추천해줘", sale))
        self.assertTrue(_should_use_web_discovery("명조 같은 게임 추천해줘", standard))
        self.assertTrue(_should_use_web_discovery("출시 예정 기대작 알려줘", upcoming))

    def test_web_supplement_requires_explicit_external_source_request(self) -> None:
        incomplete = {"coverage_ratio": 0.25, "claims": [{"supported": False}]}

        self.assertFalse(_should_use_web_supplement("PEAK의 장점을 알려줘", incomplete))
        self.assertTrue(_should_use_web_supplement("PEAK 개발사 공식 자료도 찾아줘", incomplete))

    def test_similarity_web_search_uses_canonical_seed_and_fixed_spec(self) -> None:
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"concept_summary":"기준 명세",'
                                '"candidates":[{"name":"Tower of Fantasy","reason":"유사"}]}'
                            )
                        )
                    )
                ]
            )

        generator = object.__new__(OpenAIAnswerGenerator)
        generator.model_name = "fake"
        generator._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch("steam_rag.external_apis.openai_client.TavilySearchClient") as tavily_type:
            tavily_type.return_value.search.return_value = {
                "results": [
                    {
                        "title": "Similar games",
                        "url": "https://example.com/similar",
                        "content": "Tower of Fantasy is an anime open-world RPG on Steam.",
                        "score": 0.9,
                    }
                ],
                "usage": {"credits": 1},
            }
            result = generator.discover_game_candidates(
                "명조 같은 서브컬처 게임 추천해줘",
                reference_game={"canonical_name": "Wuthering Waves", "appid": 3513350},
                similarity_spec={
                    "must_have": ["anime", "rpg"],
                    "search_terms": ["anime", "open world", "real time"],
                },
            )

        web_query = tavily_type.return_value.search.call_args.args[0]
        self.assertIn('"Wuthering Waves"', web_query)
        self.assertIn("anime open world real time", web_query)
        system_prompt = captured["messages"][0]["content"]
        self.assertIn("다시 정의하지 않는다", system_prompt)
        self.assertEqual(result["candidates"][0]["name"], "Tower of Fantasy")

    def test_old_coordinator_name_is_a_compatibility_alias(self) -> None:
        self.assertIs(AgenticRAGCoordinator, GameResearchAgent)

    def test_langgraph_workflow_expands_queries_and_returns_agent_trace(self) -> None:
        documents = [
            Document(
                "Hollow Knight has precise melee combat and exploration.",
                {
                    "appid": 367520,
                    "game_key": "hollow_knight_367520",
                    "game_name": "Hollow Knight",
                    "section": "about",
                    "item_title": "About",
                    "chunk_id": "about-1",
                },
            )
        ]
        embedder = FakeEmbedder()
        index = build_index(documents, embedder)
        workflow = SteamMultiAgentWorkflow(
            HybridTimeAwareRetriever(index),
            embedder,
            FakeGenerator(),
            config=MultiAgentConfig(max_steps=1, per_step_k=1, use_hyde=True),
        )

        answer = workflow.invoke("Hollow Knight 전투는 어때?", k=1)

        self.assertEqual(answer.metadata["orchestrator"], "langgraph")
        self.assertGreaterEqual(len(answer.metadata["query_variants"]), 2)
        agent_names = {item["agent"] for item in answer.metadata["agent_trace"]}
        self.assertIn("Query Planner Agent", agent_names)
        self.assertIn("Game Research Agent", agent_names)
        self.assertIn("Evidence Critic Agent", agent_names)
        self.assertIn("Answer Agent", agent_names)

    def test_allowed_appid_prevents_cross_game_evidence(self) -> None:
        documents = [
            Document(
                "Target combat evidence.",
                {"appid": 367520, "game_key": "target", "game_name": "Target", "section": "about"},
            ),
            Document(
                "Unrelated but lexically strong combat evidence.",
                {"appid": 1145350, "game_key": "other", "game_name": "Other", "section": "about"},
            ),
        ]
        embedder = FakeEmbedder()
        workflow = SteamMultiAgentWorkflow(
            HybridTimeAwareRetriever(build_index(documents, embedder)),
            embedder,
            FakeGenerator(),
            config=MultiAgentConfig(max_steps=1, per_step_k=4, use_hyde=False),
        )

        answer = workflow.invoke("축약명 전투 방식은?", k=4, allowed_appids=[367520])

        self.assertTrue(answer.sources)
        self.assertEqual({source.document.metadata["appid"] for source in answer.sources}, {367520})
        self.assertEqual(answer.metadata["allowed_appids"], [367520])

    def test_web_research_is_labeled_as_supplemental_evidence(self) -> None:
        documents = [
            Document(
                "Hollow Knight has precise melee combat.",
                {
                    "appid": 367520,
                    "game_key": "hollow_knight_367520",
                    "game_name": "Hollow Knight",
                    "section": "about",
                },
            )
        ]
        embedder = FakeEmbedder()
        workflow = SteamMultiAgentWorkflow(
            HybridTimeAwareRetriever(build_index(documents, embedder)),
            embedder,
            FakeWebGenerator(),
            config=MultiAgentConfig(max_steps=1, per_step_k=2, use_hyde=False),
        )

        answer = workflow.invoke(
            "Hollow Knight 전투 방식과 개발사 공식 자료를 알려줘",
            k=2,
            allowed_appids=[367520],
        )

        self.assertIn("web", {source.document.metadata.get("section") for source in answer.sources})
        web_source = next(source for source in answer.sources if source.document.metadata.get("section") == "web")
        self.assertEqual(web_source.document.metadata["source_type"], "official")
        self.assertEqual(web_source.document.metadata["appid"], 367520)

    def test_short_localized_alias_resolves_to_existing_index_appid(self) -> None:
        documents = [
            Document(
                "Expedition evidence",
                {
                    "appid": 1903340,
                    "game_key": "clair_obscur_expedition_33_1903340",
                    "game_name": "클레르 옵스퀴르: 33 원정대",
                    "section": "about",
                },
            ),
            Document(
                "Other evidence",
                {"appid": 1145350, "game_key": "hades_ii", "game_name": "Hades II", "section": "about"},
            ),
        ]
        embedder = FakeEmbedder()
        pipeline = RAGPipeline(build_index(documents, embedder), embedder, FakeGenerator())

        appids = _index_appids_for_variants(
            pipeline,
            ["33 원정대의 전투 방식과 최근 평가는 어때?"],
            limit=1,
        )

        self.assertEqual(appids, [1903340])

    def test_consumer_site_serves_ui_health_and_chat(self) -> None:
        runtime = FakeRuntime()
        client = TestClient(create_service_app(runtime))

        page = client.get("/")
        health = client.get("/api/health")
        chat = client.post(
            "/api/chat",
            json={
                "question": "그럼 최근 평가는?",
                "top_k": 5,
                "history": [
                    {"role": "user", "content": "PEAK는 친구들과 하기 좋아?"},
                    {"role": "assistant", "content": "PEAK 분석 답변"},
                ],
                "context_games": [{"appid": 3527290, "name": "PEAK"}],
            },
        )

        self.assertEqual(page.status_code, 200)
        self.assertIn("SteamLens AI", page.text)
        self.assertNotIn('data-mode="analysis"', page.text)
        self.assertNotIn('class="main-nav"', page.text)
        self.assertIn('/assets/app.js?v=35', page.text)
        self.assertIn('/assets/app.css?v=35', page.text)
        self.assertNotIn("data-prompt=", page.text)
        self.assertNotIn("data-compose-prompt=", page.text)
        self.assertNotIn("추천 시작하기", page.text)
        self.assertNotIn("명조 같은 게임", page.text)
        self.assertNotIn("할인 게임", page.text)
        self.assertIn('id="newConversation"', page.text)
        self.assertIn('class="composer-shell"', page.text)
        self.assertGreater(page.text.index('id="conversationMessages"'), page.text.index('id="landing"'))
        self.assertGreater(page.text.index('class="composer-shell"'), page.text.index('id="conversationMessages"'))
        self.assertEqual(health.json()["status"], "ready")
        self.assertEqual(chat.status_code, 200)
        self.assertEqual(chat.json()["mode"], "research")
        self.assertEqual(runtime.last_history[0]["content"], "PEAK는 친구들과 하기 좋아?")
        self.assertEqual(runtime.last_context_games, [{"appid": 3527290, "name": "PEAK"}])

        javascript = client.get("/assets/app.js")
        stylesheet = client.get("/assets/app.css")
        self.assertEqual(page.headers["cache-control"], "no-store, max-age=0")
        self.assertEqual(javascript.headers["cache-control"], "no-store, max-age=0")
        self.assertEqual(stylesheet.headers["cache-control"], "no-store, max-age=0")
        self.assertIn("AbortController", javascript.text)
        self.assertIn("async function postChat", javascript.text)
        self.assertIn("request_id: clientRequestId", javascript.text)
        self.assertIn("requestId !== requestState.id", javascript.text)
        self.assertIn("const conversationRequests", javascript.text)
        self.assertNotIn("activeRequestController", javascript.text)
        self.assertIn("steamlens-conversations-v1", javascript.text)
        self.assertIn("history, context_games, top_k", javascript.text)
        self.assertIn('conversation.view = "result"', javascript.text)
        self.assertIn('message.error ? " error"', javascript.text)
        self.assertIn("renderTurnArtifacts(message.data)", javascript.text)
        self.assertIn("data: storedData", javascript.text)
        self.assertIn("shared.cloudflare.steamstatic.com", javascript.text)
        self.assertIn("data:image/svg+xml", javascript.text)
        self.assertIn("data-history-delete", javascript.text)
        self.assertIn('message.role === "user" && !message.error', javascript.text)
        self.assertNotIn("undoHistoryDelete", javascript.text)
        self.assertNotIn("historyUndoToast", page.text)
        self.assertIn('$("#newConversation").addEventListener', javascript.text)
        self.assertIn("--chat-width: 900px", stylesheet.text)
        self.assertIn(".hero { display: block", stylesheet.text)

    def test_duplicate_chat_request_id_reuses_cached_result(self) -> None:
        runtime = FakeRuntime()
        with TestClient(create_service_app(runtime)) as client:
            payload = {
                "request_id": "recommendation_request_1234",
                "question": "스토리가 좋고 전투가 재미있는 싱글 RPG 추천해줘",
                "top_k": 6,
            }
            first = client.post("/api/chat", json=payload)
            second = client.post("/api/chat", json=payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(runtime.call_count, 1)

    def test_followup_question_is_rewritten_as_standalone_query(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="PEAK의 최근 Steam 사용자 평가는 어떤가?"
                                )
                            )
                        ]
                    )
                )
            )
        )
        generator = object.__new__(OpenAIAnswerGenerator)
        generator.model_name = "fake"
        generator._client = client

        rewritten = generator.rewrite_followup_question(
            "그럼 최근 평가는?",
            [
                {"role": "user", "content": "PEAK는 친구들과 하기 좋아?"},
                {"role": "assistant", "content": "협동 플레이를 지원합니다."},
            ],
        )

        self.assertEqual(rewritten, "PEAK의 최근 Steam 사용자 평가는 어떤가?")

    def test_followup_rewriter_does_not_feed_assistant_interpretation_back(self) -> None:
        completion = EmptyCompletionClient()
        generator = object.__new__(OpenAIAnswerGenerator)
        generator.model_name = "fake"
        generator._client = completion

        generator.rewrite_followup_question(
            "근데 명조 같은 서브컬처 게임을 원해",
            [
                {"role": "user", "content": "명조 같은 게임 추천해줘"},
                {
                    "role": "assistant",
                    "content": "명조는 인디 생존 샌드박스와 협동 감성으로 해석했습니다.",
                },
            ],
        )

        prompt = completion.kwargs["messages"][-1]["content"]
        self.assertIn("명조 같은 게임 추천해줘", prompt)
        self.assertNotIn("인디 생존 샌드박스", prompt)

    def test_corrective_followup_excludes_previous_recommendations(self) -> None:
        runtime = SteamServiceRuntime()
        expected = {"mode": "recommendation", "answer": "ok"}
        context_games = [
            {"appid": 1621690, "name": "Core Keeper"},
            {"appid": 108600, "name": "Project Zomboid"},
        ]
        question = "근데 추천 게임들에서 명조 같은 서브컬처 게임들을 원해"
        with (
            patch("steam_rag.application.service_runtime.OpenAIAnswerGenerator") as generator_type,
            patch.object(runtime, "_recommend", return_value=expected) as recommend,
        ):
            generator_type.return_value.rewrite_followup_question.return_value = question
            payload = runtime.ask(
                question,
                history=[{"role": "user", "content": "명조 같은 게임 추천해줘"}],
                context_games=context_games,
            )

        generator_type.return_value.rewrite_followup_question.assert_called_once_with(
            question,
            [{"role": "user", "content": "명조 같은 게임 추천해줘"}],
            context_games=[],
        )
        recommend.assert_called_once_with(question, excluded_appids={1621690, 108600})
        self.assertEqual(payload["followup_relation"], "correction")
        self.assertEqual(payload["excluded_appids"], [108600, 1621690])

    def test_followup_relation_preserves_normal_candidate_refinement(self) -> None:
        relation = _followup_relation(
            "그중 협동 가능한 것만 알려줘",
            [{"role": "user", "content": "친구와 할 게임 추천해줘"}],
            [{"appid": 3527290, "name": "PEAK"}],
        )

        self.assertEqual(relation, "continuation")

    def test_followup_rewriter_binds_official_candidate_to_appid(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="Library Of Ruina는 어떤 게임인지 자세히 설명해줘."
                                )
                            )
                        ]
                    )
                )
            )
        )
        generator = object.__new__(OpenAIAnswerGenerator)
        generator.model_name = "fake"
        generator._client = client

        rewritten = generator.rewrite_followup_question(
            "라이브러리 오브 루이나는 어떤 게임인지 자세히 설명해줘.",
            [{"role": "assistant", "content": "추천 결과에 Library Of Ruina가 포함됨"}],
            context_games=[
                {"appid": 1256670, "name": "Library Of Ruina"},
                {"appid": 524220, "name": "NieR:Automata"},
            ],
        )

        self.assertIn("Library Of Ruina", rewritten)
        self.assertIn("appid: 1256670", rewritten)

    def test_runtime_uses_rewritten_followup_for_retrieval(self) -> None:
        runtime = SteamServiceRuntime()
        expected = {"mode": "research", "answer": "ok"}
        with (
            patch("steam_rag.application.service_runtime.OpenAIAnswerGenerator") as generator_type,
            patch.object(runtime, "_research", return_value=expected) as research,
        ):
            generator_type.return_value.rewrite_followup_question.return_value = (
                "PEAK의 최근 Steam 사용자 평가는 어떤가?"
            )
            payload = runtime.ask(
                "그럼 최근 평가는?",
                history=[{"role": "user", "content": "PEAK는 친구들과 하기 좋아?"}],
                context_games=[{"appid": 3527290, "name": "PEAK"}],
            )

        generator_type.return_value.rewrite_followup_question.assert_called_once_with(
            "그럼 최근 평가는?",
            [{"role": "user", "content": "PEAK는 친구들과 하기 좋아?"}],
            context_games=[{"appid": 3527290, "name": "PEAK"}],
        )

        research.assert_called_once_with(
            "PEAK의 최근 Steam 사용자 평가는 어떤가?",
            top_k=6,
        )
        self.assertTrue(payload["conversation_context_used"])
        self.assertEqual(payload["resolved_question"], "PEAK의 최근 Steam 사용자 평가는 어떤가?")

    def test_candidate_resolution_rejects_generic_and_non_game_results(self) -> None:
        class Client:
            def search_store(self, term: str, *, count: int = 8) -> list[dict]:
                return [{"appid": 10, "name": term}]

            def fetch_app_details(self, appid: int, **kwargs) -> dict:
                return {"type": "dlc", "name": "Not A Game"}

        self.assertIsNone(_resolve_candidate(Client(), "Games"))
        self.assertIsNone(_resolve_candidate(Client(), "Not A Game"))

    def test_existing_document_cards_use_current_steam_image_host(self) -> None:
        url = _steam_header_image(3527290)

        self.assertEqual(
            url,
            "https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/3527290/header.jpg",
        )

    def test_recommendation_output_hides_internal_fields_and_explains_every_game(self) -> None:
        answer = _recommendation_markdown(
            "현재 세일 중이고 평가가 좋은 게임 추천해줘",
            [
                {
                    "name": "Example RPG",
                    "store_summary": "깊이 있는 이야기와 전술적인 전투를 함께 제공하는 싱글 플레이 RPG입니다.",
                    "genres": ["RPG"],
                    "popular_tags": ["풍부한 스토리"],
                    "matched_tags": ["story_rich"],
                    "matched_facets": ["playstyle_facets:story_rich"],
                    "positive_ratio": 0.91,
                    "discount_percent": 40,
                    "release_date": "",
                }
            ],
            {},
            SimpleNamespace(sale_required=True, upcoming_required=False),
        )

        self.assertIn("깊이 있는 이야기와 전술적인 전투", answer)
        self.assertIn("최근 표본 긍정 91%", answer)
        self.assertIn("40% 할인", answer)
        self.assertNotIn("playstyle_facets", answer)
        self.assertNotIn("story_rich", answer)

    def test_empty_model_output_falls_back_to_retrieved_evidence(self) -> None:
        client = EmptyCompletionClient()
        generator = object.__new__(OpenAIAnswerGenerator)
        generator.model_name = "fake"
        generator._client = client
        result = SearchResult(
            Document(
                "PEAK supports cooperative climbing with friends.",
                {
                    "game_name": "PEAK",
                    "section": "about",
                    "item_title": "게임 소개",
                },
            ),
            1.0,
            rank=1,
        )

        answer = generator.generate_service_answer("PEAK는 협동 게임이야?", [result], {})

        self.assertIn("PEAK", answer)
        self.assertIn("[근거 1]", answer)
        self.assertNotIn("근거를 바탕으로 답변을 만들지 못했습니다", answer)
        self.assertNotIn("max_completion_tokens", client.kwargs)


if __name__ == "__main__":
    unittest.main()
