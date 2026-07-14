# 4단계: Agent 검색·서비스 대화 평가

## 목적

4단계 평가는 서로 다른 두 층을 분리한다.

1. 검색 평가: 50개 단일 질문으로 `Agentic`과 `Agentic+HyDE`의 검색 품질과 비용을 비교한다.
2. 서비스 대화 평가: 20개 한국어 대화 시나리오로 추천, 비교, 후속 질문, 조건 수정, 가격·출시 상태, 별칭 해석이 실제 서비스 흐름에서 유지되는지 검사한다.

단일 검색 점수가 높아도 후속 질문에서 게임 문맥이 사라지거나, 비교 대상 중 하나가 누락되거나, 무료 게임을 100% 할인으로 표현할 수 있다. 따라서 두 평가는 서로 대체하지 않는다.

## 50문항 검색 Golden Set

`data/eval/stage4_golden_set.jsonl`은 게임 플레이, 리뷰, 패치, 패치 전후 변화, 가격, 추천, 비교 질문을 포함한다. 각 문항에는 예상 intent, 게임, section, evidence keyword, answer keyword, AppID와 필요한 경우 patch date가 기록된다.

기본 비교 전략은 다음 두 가지다.

| 전략 | 구성 |
|---|---|
| Agentic | SearchSpec, Hybrid 검색, BGE reranker, claim evidence coverage 기반 반복 검색 |
| Agentic+HyDE | Agentic 구성에 단계별 가상 문서 생성 추가 |

기존 50문항 retrieval-only 측정에서는 HyDE가 Context precision을 소폭 높였지만 평균 지연 시간이 약 6.2배 증가했다. 따라서 현재 기본값은 Agentic이며 HyDE는 선택 기능이다.

```powershell
$env:PYTHONPATH='src'
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli evaluate-stage4 --retrieval-only
```

## 20개 멀티턴 서비스 Golden Set

`data/eval/conversation_golden_set_v1.jsonl`은 20개 시나리오, 총 42개 턴으로 구성된다. 모든 시나리오는 2~4개 턴을 가지며 모든 턴에 `expected`와 `forbidden` 계약을 명시한다.

| 범주 | 검증 대상 |
|---|---|
| recommendation | 일반 추천 후 상세 질문 |
| seed_similarity | 명조·페르소나·몬헌처럼 기준 게임을 먼저 고정하는 추천 |
| comparison | 두 게임의 AppID와 근거가 모두 준비되는지 검사 |
| correction_followup | 기존 후보 제외 및 새 조건 유지 |
| detail_followup | 추천 결과의 게임을 정식명 없이 다시 물어보는 흐름 |
| price_sale_upcoming | 무료·할인·가격 미정·출시 예정 상태 구분 |
| time_aware_update | 패치 날짜, 전후 평가, 표본 수 |
| alias_localized_names | 33 원정대, 몬헌 월드 같은 한국어 별칭 해석 |

스키마만 검사하는 명령은 API, LLM, Steam 또는 Tavily를 호출하지 않는다.

```powershell
$env:PYTHONPATH='src'
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli evaluate-conversations --validate-only
```

실제 서비스 런타임을 평가할 때만 아래 명령을 명시적으로 실행한다. 이 명령은 OpenAI, Steam, Tavily 설정과 로컬 서비스 데이터를 사용할 수 있으므로 비용과 실행 시간을 확인해야 한다.

```powershell
$env:PYTHONPATH='src'
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli evaluate-conversations
```

일부 시나리오만 점검하려면 `--limit 3`처럼 제한한다.

출력 파일:

- 턴별 상세: `data/eval/conversation_benchmark_details.jsonl`
- 전체·범주별 요약: `data/eval/conversation_benchmark_summary.json`

## 결정적 계약 지표

멀티턴 러너는 LLM 평가자를 호출하지 않고 다음 항목을 문자열, AppID와 payload 상태로 계산한다.

- `answer_presence`: 답변 존재 여부
- `required_keyword_recall`: 턴별 필수 키워드 재현율
- `forbidden_keyword_leakage`: 금지된 내부 문구나 잘못된 게임명의 노출률
- `expected_appid_recall`: 필요한 게임 AppID가 결과에 포함된 비율
- `forbidden_appid_leakage`: 제외한 게임 AppID가 다시 섞인 비율
- `mode_correctness`: `recommendation`과 `research` 분기 정확도
- `continuity`: 후속 문맥 사용과 correction/detail 관계 유지
- `error_rate`: 예외가 발생한 턴의 비율
- `contract_pass`: 해당 턴의 모든 명시 계약 통과 여부
- `latency_ms`: 평균, P50, P95, 최댓값

후속 질문의 `detail` 계약은 현재 런타임 payload의 `continuation`과 동등하게 취급한다. `correction`은 후보 제외 로직과 직접 연결되므로 정확히 일치해야 한다.

예외는 한 턴의 실패로 저장하고 다음 시나리오 평가를 계속한다. 따라서 한 번의 네트워크 오류 때문에 전체 결과가 사라지지 않는다.

대화 상태 전달도 소비자 웹 클라이언트와 동일하게 맞춘다. history에는 최근 사용자 질문만 최대 8개 전달하고 assistant 답변 문장은 다시 입력하지 않는다. context game은 모든 과거 후보를 누적하지 않으며, 가장 최근 assistant 턴이 반환한 유효한 게임 목록으로 교체한다. 최근 턴에 게임이 없을 때만 이전 목록을 유지한다. AppID 점수도 화면에 노출되는 `games`와 수집 결과인 `corpus_updates`를 기준으로 계산하며, 내부 `reference_game`이나 `excluded_appids`는 추천 결과로 세지 않는다.

## 휴리스틱 지표와 RAGAS·LLM Judge의 구분

이 저장소에서 `keyword recall`, AppID recall, mode correctness, continuity와 forbidden leakage는 **결정적 계약 휴리스틱**이다. 빠르고 재현 가능하며 회귀 테스트에 적합하지만, 문장의 자연스러움이나 추천 이유의 의미적 타당성을 판정하지 않는다.

RAGAS 또는 LLM Judge는 별도 평가 계층이다. 다음과 같은 의미 품질을 측정할 때 사용한다.

- faithfulness와 factual correctness
- 질문에 대한 답변 관련성
- 추천 이유의 설득력과 사용자 조건 충족 정도
- 한국어 자연스러움과 가독성
- 인용된 근거가 실제 주장에 충분한지 여부

따라서 결정적 계약 점수를 RAGAS 점수로 표시하거나, Golden Set 키워드 일치를 정답성 검증으로 표현하면 안 된다. 권장 운영 방식은 모든 PR에서 오프라인 계약 테스트를 실행하고, 모델·프롬프트·검색 정책 변경 시 표본 또는 전체 시나리오에 RAGAS/LLM Judge를 추가 실행하는 것이다.

## 단위 테스트

`tests/unit/evaluation_tools/test_conversation_benchmark.py`는 외부 네트워크나 LLM 없이 가짜 런타임으로 다음을 검사한다.

- 20개/2~4턴 스키마와 범주 구성
- history와 context game 전달
- 최근 게임 목록 교체 및 내부 reference/excluded AppID 비계상
- 턴 오류 격리와 error rate
- 금지 키워드·AppID 누출 탐지
- JSONL 상세와 JSON 요약 저장

기존 Stage 4 검색 평가 테스트는 그대로 유지된다.
