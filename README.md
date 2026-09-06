# Steam Game RAG Agent

Steam 게임 설명, 리뷰, 뉴스/패치 문서를 검색해 근거 기반 답변을 생성하는 RAG 프로젝트입니다. 현재 실행형 파이프라인은 8주차 노트북의 Hybrid Time-aware Retrieval V2를 모듈로 분리한 버전입니다.

## 파이프라인

1. `data/docs_timeaware_playstyle/*.md`를 section/item 단위로 파싱합니다.
2. `source_date`, `game_key`, `section`, `sentiment`, `relevance_type` 메타데이터를 유지한 채 청킹합니다.
3. OpenAI 임베딩을 생성해 로컬 Chroma 벡터스토어로 저장합니다.
4. 질문에서 게임명과 의도(`gameplay`, `review`, `news`, `after_update`)를 판별합니다.
5. section 필터 안에서 dense search와 BM25를 수행하고 RRF로 결합합니다.
6. gameplay 질문은 Steam Store 인기 태그, 정규화된 Steam tag, 전투/시점/차원 facet 일치도를 반영합니다.
7. 절대 최신성, 게임별 후보군 내부 상대 최신성, 문서 품질 보너스로 재정렬합니다.
8. 검색 근거만 사용하도록 제한한 프롬프트로 한국어 답변과 `[근거 N]` 인용을 생성합니다.

`업데이트 이후 평가` 질문은 최신 patch-like 뉴스 날짜를 찾아 이후 리뷰를 우대합니다. 날짜가 없는 문서는 최신성 점수만 0점 처리하며 검색 후보 자체에서 제거하지 않습니다.

Gradio에서는 `Agentic RAG + HyDE` 모드를 선택할 수 있습니다. 이 모드는 질문을 여러 검색 목표로 나누고, 단계별 HyDE 가상 문서를 생성해 검색한 뒤 근거 충분성을 확인합니다. 자세한 설계는 `docs/agentic_hyde_rag_design.md`에 정리했습니다.

Agentic 검색은 4단계부터 `SearchSpec`과 claim 단위 evidence coverage를 사용합니다. 50개 Golden Set으로 `Agentic`과 `Agentic+HyDE`를 비교하고 검색·생성·인용·시간·추천 지표를 분리합니다. Basic/Hybrid/Reranker는 Agentic 내부 검색 기반이므로 4단계 기본 비교에서는 제외합니다. 자세한 내용은 `docs/agent_evaluation_stage4.md`에 정리했습니다.

50문항 평가에서는 HyDE의 context precision이 0.692에서 0.708로 소폭 증가했지만 claim coverage는 0.9367에서 0.9267로 감소했고 평균 latency는 2.38초에서 14.70초로 증가했습니다. 따라서 Gradio 기본 검색 방식은 `Agentic RAG`이며 HyDE는 별도 선택 옵션입니다.

## 게임 전문가 에이전트 (기획안 v0.2)

`게임전문가에이전트_기획안.md` v0.2를 반영해 **게임을 고르는 순간부터 막힌 구간을 해결하는 순간까지**를 하나의 흐름으로 구현했습니다. 항목별 구현 위치는 `docs/game_expert_agent_v0_2.md`에 정리했습니다.

- **조건 판정을 충족·위반·미확인으로 나눕니다.** 전투 방식이 확인되지 않은 게임을 조건 충족으로 취급하지 않고, 후보 카드에 "잘 맞는 점 / 선택 전 확인 / 정보 상태"를 함께 보여줍니다. 위반이 확인된 게임만 후보에서 제거합니다.
- **"턴제보다는 액션"을 턴제 요구로 읽지 않습니다.** 사용자가 명시적으로 제외한 조건은 제외 조건으로 등록하고, "액션"이라는 단어만으로 실시간 전투를 확정하지 않습니다.
- **탐색 공간과 게임별 플레이 공간을 분리합니다.** 탐색 대화는 `user_id + discovery_session_id`, 공략 대화는 `user_id + game_id + thread_id`로 저장하며, 공략 컨텍스트를 만드는 경로는 탐색 대화 테이블을 읽지 않습니다.
- **게임별 전문가는 공통 코드 + 게임별 설정입니다.** `data/game_experts/*.json`에 게임 프로필, 지원 범위, 진행 구간, 자료 이용 조건을 두고 하나의 실행 코드가 이를 불러옵니다. 초기 3개 게임은 Hollow Knight(조작 중심 액션), Baldur's Gate 3(서사 중심 진행), Monster Hunter: World(장비와 성장 시스템)입니다.
- **스포일러는 검색 단계에서 제한합니다.** 진행도와 스포일러 설정에 따라 문서를 걸러내고, 차단 사유와 답변 문구 검사 결과를 함께 남깁니다.
- **한 요청의 추가 검색 2회, 전문가 호출 3개 상한**을 두고 호출 수를 응답의 `budget`에 기록합니다.

화면은 **게임 찾기 · 비교 · 내 게임 · 게임별 플레이 공간 · 내 취향**으로 나뉩니다. 주요 진입점은 "게임 찾기"와 "내 게임"입니다.

지원 범위의 `verified_version`과 `last_reviewed`는 아직 비어 있습니다. 개발자가 직접 검증한 뒤 채우는 값이며, 비어 있는 동안 답변은 버전 미확인으로 표시합니다.

## 실행

### 사용자용 웹사이트

Gradio는 RAG 성능 검증용으로 유지하고, 실제 서비스 체험 화면은 별도의 FastAPI 웹사이트로 실행합니다.

```powershell
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe scripts\run_service.py --inbrowser
```

비공식 콘셉트 또는 출시 예정작의 웹 후보 검색을 사용하려면 `.env`에 Tavily 키를 추가합니다.

```dotenv
TAVILY_API_KEY=tvly-...
TAVILY_SEARCH_DEPTH=basic
```

일반 장르·태그·할인 추천은 웹 검색 없이 Steam 프로필만 사용합니다. `명조 같은 게임`,
`서브컬처 게임`, `출시 예정 기대작`처럼 Steam 분류만으로 후보를 만들기 어려운 질문과 사용자가
공식 웹 자료를 명시적으로 요구한 분석에만 Tavily를 호출합니다. 검색 결과는 24시간 캐시됩니다.

기본 주소는 `http://127.0.0.1:8000`입니다. 사용자용 사이트는 LangGraph 기반으로 Query Planner, Query Expansion, Game Research, Evidence Critic, Answer Agent를 실행하며, 수집·MD 생성·Chroma 인덱싱은 결정론적 서비스로 분리합니다.

Python 3.11과 Poetry 환경을 기준으로 합니다. `.env`의 기존 `OPENAI_API_KEY`를 사용합니다.

사용자용 서비스는 기본적으로 `BAAI/bge-reranker-v2-m3`를 lazy-load해 Agentic 검색 후보를
재정렬합니다. 모델은 첫 실제 검색 때 한 번 로드되며, 저사양 환경이나 진단 실행에서는 다음처럼
비활성화할 수 있습니다.

```dotenv
STEAM_RAG_ENABLE_RERANKER=0
# STEAM_RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

```powershell
poetry install
poetry run steam-rag build
poetry run steam-rag inspect
poetry run steam-rag search "할로우 나이트의 최근 패치는?"
poetry run steam-rag ask "발더스 게이트 3 업데이트 이후 유저 평가는 어때?"
```

질문에 포함된 게임 MD가 없거나 오래된 경우 Steam에서 수집하고 해당 게임만 증분 인덱싱할 수 있습니다.

```powershell
# 일반 게임명 자동 식별용 카탈로그 생성(.env의 STEAM_WEB_API_KEY 사용)
poetry run steam-rag sync-catalog

# MD 생성 + 증분 인덱싱 후 답변
poetry run steam-rag ask "ELDEN RING의 최근 평가는?" --auto-collect

# 카탈로그 없이 AppID로 직접 수집 가능
poetry run steam-rag ensure "appid: 1245620 최근 패치"
```

추천 서비스는 174,169개 상세 프로필을 선수집하지 않습니다. Registry만 전체 동기화하고, 질문 후보가 부족할 때 Core Profile을 작은 배치로 추가하며 Top 5만 상세 수집합니다.

```powershell
poetry run steam-rag sync-registry
poetry run steam-rag recommend-service "2D 턴제 RPG 게임 추천" --max-new-profiles 20
poetry run steam-rag recommend-service "최근 업데이트 이후 평가가 좋아진 액션 RPG 추천" --enrich-details
```

운영 구조와 TTL·작업 Queue 정책은 `docs/service_profile_collection.md`에 정리했습니다.

기본 경로와 모델은 다음과 같습니다.

- 문서: `data/docs_timeaware_playstyle`
- 벡터스토어: `data/chroma/steam_rag_timeaware_playstyle` (Chroma persistent store)
- 임베딩: `text-embedding-3-small`
- 답변: `gpt-5-mini`

모든 값은 CLI 옵션으로 바꿀 수 있습니다.

```powershell
poetry run steam-rag build --docs data/docs_timeaware_playstyle --index data/chroma/custom
poetry run steam-rag ask "최근 평가가 좋은 게임은?" --index data/chroma/custom --top-k 7
```

## 테스트

테스트는 OpenAI API를 호출하지 않습니다.

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

GitHub Actions도 Python 3.11에서 Poetry lock 검사, 전체 테스트, `compileall`, 웹 JavaScript
문법 검사를 같은 순서로 실행합니다.

20개 시나리오·42개 턴으로 구성된 서비스 대화 Golden Set은 먼저 외부 API 호출 없이 스키마와
계약을 검사할 수 있습니다.

```powershell
$env:PYTHONPATH = "src"
python -m steam_rag.cli evaluate-conversations --validate-only
```

실제 서비스 응답까지 평가하는 `evaluate-conversations` 실행은 OpenAI·Steam·Tavily 호출과 비용이
발생할 수 있으므로 명시적으로 실행합니다. 상세 설계와 휴리스틱/RAGAS 구분은
`docs/agent_evaluation_stage4.md`를 참고하세요.

## 주요 모듈

- `src/steam_rag/steam_collection/`: Steam Store/Review/News 수집, 인기 태그 HTML 수집, 표준 Markdown 생성
- `src/steam_rag/rag_search/`: Chroma 벡터스토어, hybrid retrieval, reranker, SearchSpec와 evidence coverage
- `src/steam_rag/agents/`: Agentic RAG 계획, HyDE 검색, LangGraph 기반 멀티 에이전트 워크플로우
- `src/steam_rag/game_recommendation/`: 추천 후보 생성, Steam 프로필 저장소, 유사도 랭킹
- `src/steam_rag/game_analysis/`: 패치 전후 리뷰 구간·긍정률·주요 장단점·변화 신뢰도 분석 및 인덱싱
- `src/steam_rag/game_metadata/`: Steam 인기 태그/genre/category 정규화와 play-style facet 추출/매칭
- `src/steam_rag/application/`: RAG 파이프라인과 서비스 런타임 오케스트레이션
- `src/steam_rag/api/`: FastAPI 서비스 앱
- `src/steam_rag/ui/`: Gradio 검증 UI와 사용자용 웹 UI 정적 파일
- `src/steam_rag/evaluation_tools/`: Agentic/HyDE 검색 평가와 멀티턴 서비스 계약 평가
- `src/steam_rag/external_apis/`: OpenAI와 Tavily 어댑터
- `src/steam_rag/cli.py`: build/search/ask/inspect/recommend/time-analysis 명령
- 플레이스타일 metadata는 Steam 인기 태그, genre/category, 상점 설명, 최근 리뷰에서 자동 생성하며 수동 profile 파일은 실행 경로에서 사용하지 않음
- `docs/playstyle_retrieval_design.md`: facet taxonomy와 검색 점수 설계
- `docs/on_demand_ingestion.md`: 질의 기반 MD 자동 생성 구조와 운영 정책
- `docs/agentic_hyde_rag_design.md`: Agentic RAG + HyDE 설계
