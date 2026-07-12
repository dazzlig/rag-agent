from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from .models import SearchResult


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overwriting process variables."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK is not installed. Run `poetry install` first.") from exc
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI()


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model_name = model
        self._client = _client()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model_name, input=list(texts))
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIAnswerGenerator:
    def __init__(self, model: str = "gpt-5-mini") -> None:
        self.model_name = model
        self._client = _client()

    def generate(self, question: str, results: Sequence[SearchResult]) -> str:
        return self._generate_with_context(question, results)

    def generate_agentic(
        self,
        question: str,
        results: Sequence[SearchResult],
        metadata: dict[str, Any],
    ) -> str:
        plan_summary = "\n".join(
            f"- step {step.get('step')}: {step.get('goal')} / sufficient={step.get('sufficient')} / sections={step.get('sections')}"
            for step in metadata.get("steps", [])
            if isinstance(step, dict)
        )
        instruction = (
            "Agentic RAG 조사 로그는 내부 참고용으로만 사용하고 최종 답변에 길게 노출하지 않는다. "
            "여러 단계에서 상충되는 근거가 있으면 최신성, 섹션 역할, 리뷰 표본 여부를 구분해서 설명한다."
        )
        return self._generate_with_context(
            question,
            results,
            extra_user_context=f"Agentic RAG 조사 로그:\n{plan_summary or '- 없음'}",
            extra_system_instruction=instruction,
        )

    def generate_hyde(self, question: str, search_query: str, reason: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 RAG 검색 품질을 높이기 위한 HyDE 문서 생성기다. "
                    "실제 사실을 단정하지 말고, 검색될 법한 짧은 가상 문서를 만든다. "
                    "게임명, 장르, 전투 방식, 시점, 업데이트/리뷰 단서를 포함하되 5문장 이내로 작성한다."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"원 질문:\n{question}\n\n"
                    f"검색 목표:\n{reason}\n\n"
                    f"검색 질의:\n{search_query}\n\n"
                    "이 질의에 잘 맞는 검색용 가상 문서를 한국어와 핵심 영어 키워드를 섞어 작성해줘."
                ),
            },
        ]
        response = self._client.chat.completions.create(model=self.model_name, messages=messages)
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def _generate_with_context(
        self,
        question: str,
        results: Sequence[SearchResult],
        *,
        extra_user_context: str = "",
        extra_system_instruction: str = "",
    ) -> str:
        system_content = (
            "당신은 Steam 게임 추천 및 분석 도우미다. 제공된 근거만 사용해 한국어로 답한다. "
            "근거가 부족하면 명확히 밝힌다. 각 핵심 주장 뒤에 [근거 N] 형식으로 출처를 표시한다. "
            "최신 리뷰 질문은 source_date와 리뷰 표본임을 명시하고, 업데이트 질문은 최신 패치 날짜를 우선한다. "
            "답변은 사용자가 읽기 쉽게 짧은 문단과 표를 우선 사용한다. 같은 결론을 반복하지 않는다. "
            "플레이스타일/전투/시점 질문은 공식/스토어 근거(metadata, store_summary, about)를 1차 근거로, 리뷰는 보조 근거로 분리한다. "
            "가격/할인 질문은 price_* metadata와 price_collected_at이 있을 때만 답하고, 이 값은 수집 시점 기준이지 실시간 가격이 아니라고 명시한다. "
            "가격 metadata가 없으면 현재 가격/할인은 제공된 근거만으로 확인할 수 없다고 답한다. "
            "업데이트 질문에서는 news_type이 hotfix, patch_note, major_update, content_update인 근거를 우선하고 sale_promo, community_event, franchise_promo는 패치 근거로 쓰지 않는다. "
            "추천 질문은 조건별 적합도와 주의점을 표로 보여준다. 비교 질문은 같은 항목끼리 비교한다. "
            "2.5D, 쿼터뷰, 아이소메트릭, 1인칭/3인칭 같은 시점 판단은 명시 근거가 없으면 '근거 부족'으로 표시한다. "
            "dimension_facets=3d만으로 '3D 시점'이라고 단정하지 말고 '현재 메타데이터상 차원 분류'라고 표현한다. "
            "출력 형식: 1) 직접 답변 2~3문장, 2) 판단 표, 3) 최근 업데이트/리뷰 표(질문에 있을 때), 4) 근거와 한계, 5) 사용한 근거 표. "
            "표의 열은 되도록 '항목 | 판단 | 신뢰도 | 근거 요약'을 사용한다. "
            "'요약 결론'과 '간단 결론'처럼 중복 섹션을 둘 다 만들지 않는다. "
            "'페이싯' 대신 '분류 항목' 또는 'facet'이라고 쓴다. "
            "마지막은 한 줄 최종 판단으로 끝내고 긴 재요약을 반복하지 않는다."
        )
        if extra_system_instruction:
            system_content = f"{system_content} {extra_system_instruction}"
        context_blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            metadata = result.document.metadata
            context_blocks.append(
                f"[근거 {index}]\n"
                f"game={metadata.get('game_name') or metadata.get('game_key')}\n"
                f"section={metadata.get('section')}\n"
                f"title={metadata.get('item_title')}\n"
                f"source_date={metadata.get('source_date')}\n"
                f"news_type={metadata.get('news_type')}\n"
                f"relevance_type={metadata.get('relevance_type')}\n"
                f"latest_patch_date={result.latest_patch_date}\n"
                f"latest_patch_title={result.latest_patch_title}\n"
                f"is_free={metadata.get('is_free')}\n"
                f"price_available={metadata.get('price_available')}\n"
                f"price_currency={metadata.get('price_currency')}\n"
                f"price_initial={metadata.get('price_initial')}\n"
                f"price_final={metadata.get('price_final')}\n"
                f"price_discount_percent={metadata.get('price_discount_percent')}\n"
                f"price_initial_formatted={metadata.get('price_initial_formatted')}\n"
                f"price_final_formatted={metadata.get('price_final_formatted')}\n"
                f"price_collected_at={metadata.get('price_collected_at')}\n"
                f"steam_tags={metadata.get('steam_tags_normalized')}\n"
                f"popular_tags_source={metadata.get('popular_tags_source')}\n"
                f"popular_tags_collected_at={metadata.get('popular_tags_collected_at')}\n"
                f"steam_genres={metadata.get('steam_genres_normalized')}\n"
                f"combat_facets={metadata.get('combat_facets')}\n"
                f"perspective_facets={metadata.get('perspective_facets')}\n"
                f"dimension_facets={metadata.get('dimension_facets')}\n"
                f"playstyle_facets={metadata.get('playstyle_facets')}\n"
                f"matched_query_facets={result.matched_facets}\n"
                f"sentiment={metadata.get('sentiment')}\n"
                f"url={metadata.get('url')}\n"
                f"{result.document.page_content}"
            )
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": (
                    f"질문:\n{question}\n\n"
                    f"{extra_user_context + chr(10) + chr(10) if extra_user_context else ''}"
                    "검색 근거:\n" + "\n\n".join(context_blocks)
                ),
            },
        ]
        response = self._client.chat.completions.create(model=self.model_name, messages=messages)
        content = response.choices[0].message.content
        return content.strip() if content else "답변을 생성하지 못했습니다."
