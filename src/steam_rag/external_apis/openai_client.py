from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from steam_rag.common.models import SearchResult
from steam_rag.common.telemetry import tracked_openai_call
from steam_rag.external_apis.tavily_client import TavilySearchClient, compact_tavily_results


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
        response = tracked_openai_call(
            model=self.model_name,
            operation="embedding",
            call=lambda: self._client.embeddings.create(
                model=self.model_name,
                input=list(texts),
            ),
        )
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIAnswerGenerator:
    def __init__(self, model: str = "gpt-5-mini") -> None:
        self.model_name = model
        self._client = _client()

    def generate(self, question: str, results: Sequence[SearchResult]) -> str:
        return self._generate_with_context(question, results)

    def _chat_completion(self, **kwargs: Any) -> Any:
        model = str(kwargs.get("model") or self.model_name)
        return tracked_openai_call(
            model=model,
            operation="chat",
            call=lambda: self._client.chat.completions.create(**kwargs),
        )

    def rewrite_followup_question(
        self,
        question: str,
        history: Sequence[dict[str, str]],
        *,
        context_games: Sequence[dict[str, Any]] = (),
    ) -> str:
        """Resolve omitted entities in a follow-up without carrying old answers as evidence."""

        # Assistant prose is deliberately excluded.  A previous recommendation is
        # an untrusted proposal, not conversation truth; feeding it back here can
        # preserve a bad concept interpretation across corrective follow-ups.
        transcript = "\n".join(
            f"{str(item.get('role') or '')}: {str(item.get('content') or '')[:700]}"
            for item in history[-8:]
            if str(item.get("role") or "") == "user"
            and str(item.get("content") or "").strip()
        )
        verified_games = [
            {"appid": int(item["appid"]), "name": str(item["name"]).strip()}
            for item in context_games
            if item.get("appid") and str(item.get("name") or "").strip()
        ]
        candidate_text = "\n".join(
            f"- {item['name']} (appid: {item['appid']})" for item in verified_games
        ) or "- 없음"
        response = self._chat_completion(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Steam 게임 서비스의 후속 질문 재작성 Agent다. 이전 대화에서 생략된 게임명, "
                        "비교 대상과 추천 조건만 복원해 최신 사용자 질문을 하나의 독립적인 한국어 질문으로 "
                        "바꾼다. 질문에 답하지 말고, 이전 AI 답변의 사실을 근거로 추가하지 않는다. "
                        "검증된 이전 게임 후보가 제공되면 그 목록 밖의 게임을 추측하지 않는다. 사용자가 번역명, "
                        "음역명, 줄임말, 순서(첫 번째 등)로 후보를 가리키면 반드시 목록의 정확한 공식 이름과 "
                        "`appid: 숫자`를 질문에 함께 넣는다. "
                        "추천의 후속 조건이면 '추천' 의도를, 업데이트 후속 질문이면 업데이트 분석 의도를 "
                        "문장에 유지한다. '아니', '그런 게임 말고', '원한 건', '원해'처럼 기존 추천을 "
                        "교정하는 표현이 있으면 기존 후보에 질문을 묶지 말고 최신 사용자 문장에 명시된 "
                        "기준 게임과 조건을 그대로 유지한다. 사용자가 새 주제로 전환했으면 원 질문을 "
                        "그대로 반환한다. "
                        "설명이나 따옴표 없이 재작성한 질문 한 문장만 반환한다."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"이전 대화:\n{transcript}\n\n"
                        f"검증된 이전 게임 후보:\n{candidate_text}\n\n"
                        f"최신 질문:\n{question}"
                    ),
                },
            ],
        )
        content = ""
        if getattr(response, "choices", None):
            content = str(response.choices[0].message.content or "").strip().strip('"')
        content = content[:1200] or question
        if verified_games and not re.search(r"appid\s*:\s*\d+", content, flags=re.IGNORECASE):
            normalized_content = re.sub(r"[^a-z0-9가-힣]+", "", content.casefold())
            matches = [
                item
                for item in verified_games
                if re.sub(r"[^a-z0-9가-힣]+", "", item["name"].casefold()) in normalized_content
            ]
            if len(matches) == 1:
                content = f"{content} (appid: {matches[0]['appid']})"
            elif len(verified_games) == 1:
                content = f"{content} (appid: {verified_games[0]['appid']})"
        return content[:1200]

    def ground_reference_game(self, question: str) -> dict[str, Any]:
        """Resolve a nickname/translated seed title without inventing a Steam AppID."""

        response = self._chat_completion(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "게임 유사 추천의 기준 작품을 식별하는 에이전트다. '<게임> 같은', '<게임>처럼', "
                        "'<게임>과 비슷한' 표현에서 작품명·별칭을 분리하고, 한국어 서비스명·축약명·음역명을 "
                        "전 세계에서 통용되는 공식 제목으로 정규화한다. 예: '명조'는 'Wuthering Waves', "
                        "'33 원정대'는 'Clair Obscur: Expedition 33'이다. AppID는 추측하지 않는다. "
                        "similarity_terms에는 검색 결과가 아니라 기준 작품 자체에서 확실한 특징만 최대 8개 "
                        "넣는다. 허용 예시는 anime, character_focused, character_collection, action_rpg, "
                        "turn_based, real_time, open_world, exploration, story_rich, third_person이다. "
                        "기준 작품이 명시되지 않았으면 is_similarity_request=false로 반환한다. 반드시 다음 "
                        "JSON만 반환한다: "
                        '{"is_similarity_request":true,"reference_phrase":"...",'
                        '"canonical_name":"...","aliases":["..."],"similarity_terms":["..."]}'
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        content = str(response.choices[0].message.content or "{}")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            payload = json.loads(match.group(0)) if match else {}
        if not isinstance(payload, dict):
            return {}
        aliases = payload.get("aliases")
        payload["aliases"] = [
            str(value).strip()
            for value in aliases if str(value).strip()
        ][:6] if isinstance(aliases, list) else []
        terms = payload.get("similarity_terms")
        payload["similarity_terms"] = [
            re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
            for value in terms if str(value).strip()
        ][:8] if isinstance(terms, list) else []
        payload.pop("appid", None)
        return payload

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
        coverage = metadata.get("evidence_coverage", {})
        claim_lines = []
        if isinstance(coverage, dict):
            for claim in coverage.get("claims", []):
                if isinstance(claim, dict):
                    claim_lines.append(
                        f"- {claim.get('claim_id')}: supported={claim.get('supported')} / "
                        f"evidence={claim.get('evidence_ranks')} / missing={claim.get('missing')}"
                    )
        instruction = (
            "Agentic RAG 조사 로그는 내부 참고용으로만 사용하고 최종 답변에 길게 노출하지 않는다. "
            "여러 단계에서 상충되는 근거가 있으면 최신성, 섹션 역할, 리뷰 표본 여부를 구분해서 설명한다. "
            "지원된 claim은 해당 evidence 번호를 인용하고, 지원되지 않은 claim은 추측하지 말고 근거 부족으로 표시한다."
        )
        return self._generate_with_context(
            question,
            results,
            extra_user_context=(
                f"Agentic RAG 조사 로그:\n{plan_summary or '- 없음'}\n\n"
                f"Claim 단위 evidence coverage:\n{chr(10).join(claim_lines) or '- 없음'}"
            ),
            extra_system_instruction=instruction,
        )

    def generate_service_answer(
        self,
        question: str,
        results: Sequence[SearchResult],
        metadata: dict[str, Any],
    ) -> str:
        """Generate the concise consumer-facing answer used by SteamLens."""

        coverage = metadata.get("evidence_coverage", {})
        missing = []
        if isinstance(coverage, dict):
            missing = [
                str(item.get("text") or item.get("claim_id"))
                for item in coverage.get("claims", [])
                if isinstance(item, dict) and not item.get("supported")
            ]
        context_blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            source = result.document.metadata
            context_blocks.append(
                f"[근거 {index}] 게임={source.get('game_name') or source.get('game_key')}; "
                f"구분={source.get('section')}; 출처유형={source.get('source_type') or 'steam_corpus'}; "
                f"제목={source.get('item_title')}; 발행처={source.get('publisher')}; "
                f"날짜={source.get('source_date')}; URL={source.get('url')}\n"
                f"{result.document.page_content[:1800]}"
            )
        response = self._chat_completion(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Steam 게임 추천·분석 서비스의 최종 답변 작성 Agent다. 제공된 근거만 사용한다. "
                        "한국어 1,100자 이내로 작성하고 첫 2문장 안에 직접 결론을 말한다. "
                        "이후 핵심 이유는 3~5개 짧은 항목으로 정리한다. 같은 결론을 반복하지 않는다. "
                        "내부 필드명, facet, 조사 로그, 긴 리뷰 표, 근거 번호 목록 표는 노출하지 않는다. "
                        "web 근거는 '웹 보조 근거'로 구분하고 Steam의 AppID·가격·할인·인기 태그·유저 평가 "
                        "수치를 대신하는 근거로 사용하지 않는다. "
                        "4인 비교에서 두 게임 모두 4인을 지원하면 최대 인원이 정확히 4인이라는 이유만으로 "
                        "우열을 정하지 말고, 협동 방식·장르·소통 부담과 취향 차이로 조건부 결론을 낸다. "
                        "업데이트 후 평가가 좋아졌는지 묻는 질문은 업데이트 전후 표본의 긍정률과 표본 수가 "
                        "모두 있을 때만 개선됐다고 단정한다. 최신 긍정 리뷰만 있으면 현재 반응만 설명하고 "
                        "개선 여부는 확인할 수 없다고 첫 결론에서 명시한다. "
                        "근거가 부족한 조건만 마지막에 한 문장으로 알린다. 각 주장 뒤에는 [근거 N]을 붙인다."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"질문: {question}\n"
                        f"근거가 부족한 항목: {', '.join(missing[:4]) or '없음'}\n\n"
                        + "\n\n".join(context_blocks)
                    ),
                },
            ],
        )
        content = ""
        if getattr(response, "choices", None):
            content = str(response.choices[0].message.content or "").strip()
        if not content:
            return _fallback_service_answer(results)
        return _truncate_service_answer(content, max_chars=1400)

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
        response = self._chat_completion(model=self.model_name, messages=messages)
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def expand_search_queries(self, question: str) -> list[str]:
        """Create alias/title variants, leaving AppID verification to Steam."""

        response = self._chat_completion(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Steam 게임 검색어 확장 에이전트다. 사용자가 한국어 별칭, 축약명, 번역명 또는 "
                        "영문명을 사용할 수 있다. 같은 대상을 찾기 위한 짧은 검색어를 최대 4개 만든다. "
                        "예: '33 원정대'는 '33 원정대', '클레르 옵스퀴르', "
                        "'Clair Obscur: Expedition 33', 'Expedition 33'으로 확장할 수 있다. "
                        "비교 질문에 여러 게임이 명시되면 각 게임을 찾을 수 있는 별도 검색어도 포함하되, "
                        "질문에 없는 게임을 임의로 추가하거나 AppID를 추측하지 않는다. "
                        "반드시 {\"query_variants\": [\"...\"]} JSON만 반환한다."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            payload = json.loads(match.group(0)) if match else {}
        values = payload.get("query_variants", []) if isinstance(payload, dict) else []
        return [str(value).strip() for value in values if str(value).strip()][:4]

    def discover_game_candidates(
        self,
        question: str,
        *,
        limit: int = 10,
        reference_game: dict[str, Any] | None = None,
        similarity_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use web search for candidate recall after the reference meaning is fixed."""

        reference_game = dict(reference_game or {})
        similarity_spec = dict(similarity_spec or {})
        reference_name = str(
            reference_game.get("canonical_name") or reference_game.get("name") or ""
        ).strip()
        search_terms = [
            str(value).strip()
            for value in similarity_spec.get("search_terms", [])
            if str(value).strip()
        ][:8]
        if reference_name:
            web_query = (
                f'games similar to "{reference_name}" on Steam '
                f"{' '.join(search_terms)} individual PC games"
            )
        else:
            web_query = f"{question} Steam PC 게임 추천 실제 개별 게임 제목"

        search = TavilySearchClient().search(
            web_query,
            max_results=min(max(4, limit), 8),
            search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
        )
        sources = compact_tavily_results(search, limit=8)
        if not sources:
            return {"concept_summary": "", "candidates": [], "source_urls": []}
        response = self._chat_completion(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": (
                "Tavily 검색 결과에서 Steam 게임 추천 후보를 추출한다. 기준 게임과 유사도 조건은 검색 전에 "
                "이미 확정되었으므로 검색 결과를 보고 기준 게임이나 '서브컬처'의 뜻을 다시 정의하지 않는다. "
                "concept_summary에는 제공된 기준 게임·유사도 조건을 한 문장으로 설명하고, 검색 결과에 반복된 "
                "엉뚱한 특징을 추가하지 않는다. 특정 작품과 비슷한 게임 요청은 미술 스타일만 닮은 후보를 "
                "나열하지 말고 전투 방식, 진행 구조, 세계관 표현, 캐릭터 중심성, 싱글/멀티 구조 중 제공된 "
                "핵심 조건을 두 가지 이상 만족하는 후보만 남긴다. 실시간 오픈월드 액션 게임과 생존 "
                "샌드박스·픽셀 농장·파티 퍼즐 게임을 단순히 인디 또는 협동이라는 이유로 묶지 않는다. "
                "Steam의 실제 개별 게임 상품명만 후보로 반환하고 'Games' 같은 일반 단어, 번들, DLC, "
                "프랜차이즈 허브는 제외한다. 출시 예정작 요청에는 현재 아직 출시되지 않은 게임만 넣는다. "
                "가격, 할인, 출시일, 평가, 태그는 추측하지 않는다. 다음 JSON만 출력한다: "
                '{"concept_summary":"...","candidates":[{"name":"...","reason":"..."}]}'
                )},
                {"role": "user", "content": (
                    f"요청: {question}\n"
                    f"확정된 기준 게임: {json.dumps(reference_game, ensure_ascii=False)}\n"
                    f"검증할 유사도 조건: {json.dumps(similarity_spec, ensure_ascii=False)}\n"
                    f"최대 후보 수: {max(1, min(limit, 12))}\n\n"
                    f"Tavily 검색 결과:\n{json.dumps(sources, ensure_ascii=False)}"
                )},
            ],
        )
        content = str(response.choices[0].message.content or "{}")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            payload = json.loads(match.group(0)) if match else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["source_urls"] = [item["url"] for item in sources]
        payload["search_provider"] = "tavily"
        payload["search_query"] = web_query
        payload["search_cache_hit"] = bool(search.get("cache_hit"))
        payload["search_credits"] = (
            0 if search.get("cache_hit") else int((search.get("usage") or {}).get("credits") or 0)
        )
        return payload

    def research_web_evidence(
        self,
        question: str,
        *,
        game_names: Sequence[str],
        missing_claims: Sequence[str] = (),
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Collect bounded, attributable web evidence for claims missing from Steam data."""

        query = f"{question} {' '.join(game_names)} {' '.join(missing_claims)} official developer source"
        search = TavilySearchClient(cache_ttl_seconds=6 * 60 * 60).search(
            query,
            max_results=max(3, min(limit + 2, 6)),
            search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
        )
        sources = compact_tavily_results(search, limit=6)
        if not sources:
            return []
        response = self._chat_completion(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": (
                "Tavily 검색 결과에서 Steam 게임 분석의 보조 근거를 추출한다. 대상 게임은 "
                f"{', '.join(game_names) or '질문에 명시된 게임'}이다. 다른 게임의 자료를 섞지 않는다. "
                "개발사·퍼블리셔 공식 사이트와 공식 Steam 공지를 우선하고, 그다음 신뢰할 수 있는 "
                "게임 매체를 사용한다. Steam AppID, 현재 가격, 할인율, 인기 사용자 태그, Steam 유저 "
                "평가 수치는 웹 기사로 대체하거나 추측하지 않는다. 제공된 URL만 사용해 다음 JSON을 출력한다: "
                '{"evidence":[{"game":"...","title":"...","claim":"...",'
                '"snippet":"...","publisher":"...","published_at":"YYYY-MM-DD 또는 빈 문자열",'
                '"url":"https://...","source_type":"official 또는 media"}]}'
                )},
                {"role": "user", "content": (
                    f"질문: {question}\n부족한 주장: {', '.join(missing_claims) or '공식·외부 맥락'}\n"
                    f"최대 근거 수: {max(1, min(limit, 5))}\n\n"
                    f"Tavily 검색 결과:\n{json.dumps(sources, ensure_ascii=False)}"
                )},
            ],
        )
        content = str(response.choices[0].message.content or "{}")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            payload = json.loads(match.group(0)) if match else {}
        rows = payload.get("evidence", []) if isinstance(payload, dict) else []
        sources_by_url = {item["url"].rstrip("/"): item for item in sources}
        return [
            {
                **dict(row),
                "url": sources_by_url[str(row.get("url") or "").rstrip("/")]["url"],
            }
            for row in rows
            if isinstance(row, dict)
            and str(row.get("url") or "").rstrip("/") in sources_by_url
        ][: max(1, min(limit, 5))]

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
                f"patch_date={metadata.get('patch_date')}\n"
                f"patch_title={metadata.get('patch_title')}\n"
                f"patch_event_type={metadata.get('patch_event_type')}\n"
                f"patch_importance={metadata.get('patch_importance')}\n"
                f"before_sample_size={metadata.get('before_sample_size')}\n"
                f"before_positive_ratio={metadata.get('before_positive_ratio')}\n"
                f"after_sample_size={metadata.get('after_sample_size')}\n"
                f"after_positive_ratio={metadata.get('after_positive_ratio')}\n"
                f"positive_ratio_delta_pp={metadata.get('positive_ratio_delta_pp')}\n"
                f"change_direction={metadata.get('change_direction')}\n"
                f"confidence_label={metadata.get('confidence_label')}\n"
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
        response = self._chat_completion(model=self.model_name, messages=messages)
        content = response.choices[0].message.content
        return content.strip() if content else "답변을 생성하지 못했습니다."


def _truncate_service_answer(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    boundary = text.rfind("\n", 0, max_chars)
    if boundary < max_chars // 2:
        boundary = text.rfind(". ", 0, max_chars)
    if boundary < max_chars // 2:
        boundary = max_chars
    return text[:boundary].rstrip() + "\n\n※ 핵심 내용만 표시했습니다."


def _fallback_service_answer(results: Sequence[SearchResult]) -> str:
    """Return useful evidence when the model finishes without visible text."""

    if not results:
        return "현재 확인할 수 있는 검색 근거가 없습니다. 게임 이름이나 조건을 조금 더 구체적으로 입력해 주세요."
    game = str(
        results[0].document.metadata.get("game_name")
        or results[0].document.metadata.get("game_key")
        or "대상 게임"
    )
    lines = [
        f"**{game}** 관련 근거는 정상적으로 검색됐지만 답변 생성이 완료되지 않아 핵심 근거를 대신 보여드립니다.",
        "",
    ]
    for index, result in enumerate(results[:4], start=1):
        metadata = result.document.metadata
        title = str(metadata.get("item_title") or metadata.get("section") or f"근거 {index}")
        snippet = re.sub(r"\s+", " ", result.document.page_content).strip()
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "…"
        lines.append(f"- **{title}**: {snippet} [근거 {index}]")
    lines.append("")
    lines.append("잠시 후 같은 질문을 다시 시도하면 자연어 분석 답변을 생성할 수 있습니다.")
    return "\n".join(lines)
