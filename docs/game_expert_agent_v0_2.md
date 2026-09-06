# 게임 전문가 에이전트 기획안 v0.2 구현 대응표

기획안(`게임전문가에이전트_기획안.md`, 2026-09-05)의 항목을 이 저장소의 어느 코드가
담당하는지 정리한 문서다. 기획안이 "제안"이라고 밝힌 수치(50개 검증셋, 3개 게임,
8~10주 일정)는 구현 완료를 뜻하지 않으므로, 아래 표에도 **구현된 메커니즘**과
**아직 채워야 할 데이터**를 나눠 적었다.

## 1. 무엇을 새로 만들었나

| 기획안 | 구현 |
| --- | --- |
| 4.1 필수·제외·선호 조건, 충족·위반·미확인 판정 | `src/steam_rag/game_recommendation/constraints.py` |
| 4.1 "액션"만으로 실시간 전투를 확정하지 않음, 명시적 제외 처리 | `game_recommendation/query_parser.py`의 `NEGATED_PREFERENCE`, `DIRECT_CONTROL_PHRASE` |
| 4.2 후보 카드의 잘 맞는 점·선택 전 확인·정보 상태 | `constraints.CandidateConstraintReport`, `service_runtime._candidate_payload`, `ui/web/app.js`의 `renderCandidateDetail` |
| 4.3 같은 기준 비교 | `game_recommendation/comparison.py`, `POST /api/compare`, 비교 화면 |
| 4.3 거절 이유를 반영한 재검색 | `service_runtime._candidate_feedback`, `OpenAIAnswerGenerator.interpret_candidate_feedback` |
| 4.4 탐색 공간과 게임별 플레이 공간 분리 | `user_workspace/store.py`, `service_runtime._play`, `ChatRequest.workspace` |
| 4.4 주제별 공략 대화 | `WorkspaceStore.open_play_thread`, `/api/games/threads` |
| 4.5 다섯 화면 | `ui/web/index.html`의 `space-nav`와 `#compareSpace`·`#librarySpace`·`#tasteSpace`·`#playSpace` |
| 6.1 세 역할 | 통합 = `SteamServiceRuntime`, 탐색·비교 = `SteamMultiAgentWorkflow` + `DynamicRecommendationService`, 게임별 전문가 = `game_expert/expert.py` |
| 6.2 공통 전문가 코드 + 게임별 설정 | `game_expert/support_scope.py`, `data/game_experts/*.json` |
| 7 추가 조사와 호출 상한 | `service_runtime._investigate_unverified_conditions`, `tools/game_tools.ToolBudget` |
| 8 공략 답변 원칙과 스포일러 범위 | `game_expert/expert.py`, `game_expert/spoiler.py` |
| 9.1 지원 범위 명시 | `SupportScope`, `GameExpertProfile.decide_scope` |
| 11 기억과 개인화, 대화 분리 키 | `user_workspace/store.py` |
| 12 좁은 도구 | `tools/game_tools.py` |

## 2. 조건 판정: 충족 · 위반 · 미확인

기획안 7절은 "필수 조건의 판정은 충족·위반·미확인으로 구분한다. 전투 방식이
미확인인 게임을 조건 충족으로 취급하지 않는다"고 정한다.

이전 구현은 boolean hard filter였다. 조건에 해당하는 메타데이터가 **없는** 게임과
조건을 **위반한** 게임을 똑같이 제거했기 때문에, 사용자에게 "왜 후보가 없는지"와
"무엇을 아직 확인하지 못했는지"를 구분해서 말할 수 없었다.

새 규칙은 프로필 데이터의 존재 여부만으로 판정한다.

| 프로필 상태 | 판정 | 후보 처리 |
| --- | --- | --- |
| 요청 값이 해당 항목에 있음 | `satisfied` | 확인된 후보로 먼저 노출 |
| 항목에 값이 있는데 요청 값이 없음 | `violated` | 후보에서 제거하고 사유 기록 |
| 항목 자체가 비어 있음 | `unverified` | 후보로 남기되 "선택 전 확인"에 표시 |

`satisfied`는 다시 근거 등급으로 나뉜다. Steam 공식 장르·카테고리·인기 태그에서
나온 값은 `confirmed`, 스토어 설명이나 리뷰 문장에서 해석한 값은 `interpreted`이며
후자는 카드의 "선택 전 확인"에 남는다(기획안 4.2 "확인된 사실, 자료에 대한 해석,
취향에 대한 예상이 구분되어야 한다").

`RecommendationProfileIndex.search`는 이제 `violated`만 제거하고, 확인된 후보를
미확인 후보보다 앞에 정렬한다. 확인된 후보가 하나도 없으면 답변이 그 사실을 먼저
알린다.

### 2.1 창립 사례 회귀 테스트

기획안 2.1의 실패 경험(그림체는 마음에 들지만 턴제라 환불)을 그대로 테스트로
고정했다. `tests/unit/game_recommendation/test_constraints.py`의
`test_founding_case_pretty_art_but_turn_based_combat_is_rejected`가 확인하는 것:

1. "턴제보다는 직접 움직이고 공격하는 액션이 좋아"에서 `turn_based`를 **요구
   조건으로 읽지 않는다**(이전 파서의 실제 버그였다).
2. 턴제를 제외 조건으로 등록한다.
3. "액션"이라는 단어만으로 `real_time`을 확정하지 않고, 사용자가 말한 직접 조작만
   조건으로 삼는다.
4. 턴제 전투가 확인된 게임은 후보에서 제거되고 사유가 남는다.

## 3. 공간 분리와 컨텍스트 규칙

기획안 11절은 저장 키까지 분리하라고 정한다. `WorkspaceStore`는 이를 테이블
수준에서 강제한다.

| 대화 | 키 | 테이블 |
| --- | --- | --- |
| 탐색 | `user_id + discovery_session_id` | `discovery_sessions`, `discovery_messages` |
| 공략 | `user_id + game_id + thread_id` | `play_threads`, `play_messages` |
| 게임 상태 | `user_id + game_id + playthrough` | `game_states`, `game_attempts` |

`WorkspaceStore.play_context()`는 공략 요청 컨텍스트를 만드는 유일한 경로이며,
탐색 대화 테이블을 **읽지 않는다**. 프롬프트 지시가 아니라 조회 경로 자체로
분리를 보장하기 위해서다. 같은 게임의 다른 주제 대화도 자동 첨부하지 않는다.

탐색에서 플레이 공간으로 넘어갈 때는 `handoff_to_play_space()`가 게임 ID·이름·
플랫폼만 옮긴다. 탐색 대화 전체, 다른 후보, 이번 예산은 넘기지 않는다(기획안 4.4).

회차는 `playthrough`로 분리해 같은 게임을 다시 시작해도 이전 진행도를 덮어쓰지
않는다.

## 4. 게임별 전문가

기획안 6.2대로 게임마다 프로세스를 두지 않는다. `GameExpertAgent` 하나가
`data/game_experts/<game>.json`의 설정을 불러 동작한다.

```
GameExpertProfile
├── appid / aliases / platforms / editions
├── key_systems      확인된 시스템과 그 출처
├── milestones       스포일러 범위를 정하는 순서 있는 진행 구간
├── knowledge_sources 자료의 이용 조건·버전·스포일러 구분 여부
└── support          확인된 주제·구간·버전·마지막 검토 시점
```

초기 3개 게임은 기획안 9.1이 요구한 서로 다른 질문 유형을 덮는다.

| 게임 | AppID | 질문 유형 |
| --- | --- | --- |
| Hollow Knight | 367520 | 조작 중심 액션 |
| Baldur's Gate 3 | 1086940 | 서사 중심 진행 |
| Monster Hunter: World | 582010 | 장비와 성장 시스템 |

**아직 채워야 할 값**: 각 파일의 `support.verified_version`,
`support.last_reviewed`, `key_systems[].verified_at`은 비어 있다. 개발자가 직접
검증한 뒤 채우는 자리이며, 비어 있는 동안 답변은 "버전 미확인"으로 표시한다.
지원 범위 밖 주제는 `decide_scope()`가 그 사실을 먼저 알린다.

### 4.1 답변 순서와 재시도

답변은 **현재 상황 진단 → 바로 시도할 행동 → 필요한 이유 → 추가 힌트** 순서를
지킨다. LLM 호출이 실패하거나 빈 응답이면 같은 함수 안에서 결정론적 답변으로
대체한다(별도 formatter 서비스를 만들지 않는다).

"알려준 대로 했는데 안 됐어"류 문장은 `RETRY_PATTERN`이 잡아내고, 이전 시도를
반복하지 않도록 장비·전략·조작·공략 버전 중 무엇을 다시 볼지 밝히게 한다. 실패한
시도는 `game_attempts`에 기록된다.

### 4.2 필요한 정보만 묻기

기획안 8-2대로 주제별로 필요한 상태만 요구한다.

| 주제 | 필요한 상태 |
| --- | --- |
| 보스 공략 | 진행 구간, 장비 |
| 장비·빌드 | 빌드 |
| 성장·자원 | 진행 구간 |
| 초반 가이드 · 시스템 설명 · 업데이트 | 없음 |

## 5. 스포일러

기획안 8절은 "답변 문구만 조심해서 해결하지 않는다"고 정한다. `spoiler.py`는
검색 결과 자체를 걸러낸다.

| 설정 | 스토리 자료 | 구간 자료 |
| --- | --- | --- |
| `no_spoiler`(기본) | 전면 차단 | 첫 구간까지 허용 |
| `progress` | 현재 진행 구간까지 | 현재 진행 구간까지(진행도 미확인이면 첫 구간까지) |
| `all` | 제한 없음 | 제한 없음 |

* 문서의 제목과 본문을 milestone 키워드로 검사하고, 차단 사유를 남긴다.
* 스포일러 구분이 확인되지 않은 공략 자료는 상세 답변에 쓰지 않는다.
* 진행도가 불분명한데 스토리 질문이면 검색 전에 짧게 되묻는다.
* 생성된 답변 문구도 다시 검사해 범위를 넘는 표현을 알린다.

`no_spoiler`에서도 스토리 비중이 없는 첫 구간 자료는 허용한다. 기획안 4.4의
예시("스포일러 없이 초반에 알아야 할 것만 알려줘")가 동작해야 하기 때문이다.

## 6. 도구와 호출 상한

기획안 12절이 제시한 도구 이름과 이 저장소의 구현은 다음과 같이 대응한다.
같은 계약을 이미 가진 코드에 얇은 wrapper를 새로 만들지 않았다.

| 기획안 도구 | 구현 |
| --- | --- |
| `search_games` | `DynamicRecommendationService.recommend` |
| `get_game_facts` | `tools/game_tools.get_game_facts` |
| `compare_candidates` | `tools/game_tools.compare_candidates` |
| `search_game_knowledge` | `HybridTimeAwareRetriever.retrieve(allowed_appids=[appid])` + 스포일러 필터 |
| `get_user_game_state` | `tools/game_tools.get_user_game_state` |

`ToolBudget`은 기획안 7절의 초기 운영값(한 요청의 추가 검색 2회, 전문가 호출 최대
3개)을 강제하고, 15절이 요구한 "한 요청에서 호출한 전문가 수와 추가 검색 횟수"를
응답의 `budget` 필드로 남긴다. 상한에 도달하면 예외 대신 확인한 결과와 남은
미확인 항목을 반환한다.

## 7. API

| 엔드포인트 | 용도 |
| --- | --- |
| `POST /api/chat` | `workspace`(`discovery`/`play`), `game_id`, `thread_id`, `playthrough`로 공간을 구분 |
| `GET/POST/DELETE /api/library` | 내 게임 |
| `GET/POST/DELETE /api/preferences` | 내 취향 |
| `POST /api/play-space` | 탐색 → 플레이 공간 이동 |
| `GET/POST /api/games/{appid}/threads`, `/api/games/threads` | 주제별 공략 대화 |
| `GET/PUT /api/games/{appid}/state` | 진행도·장비·스포일러 설정 |
| `POST /api/games/{appid}/playthrough` | 새 회차 시작 |
| `POST /api/compare` | 같은 기준 비교 |

## 8. 평가에서 확인할 축 (기획안 14.2)

구현이 로그로 남기므로 자동 집계할 수 있는 항목:

* **필수 조건 준수** — 응답의 `recommendation.hard_constraint_gate`,
  `selection.constraint_gate`에 충족·위반·미확인이 AppID별로 남는다.
* **컨텍스트 분리** — 공략 응답의 `expert.applied_scope.appid`와 근거의 `appid`가
  일치하는지 확인한다. `play_context()`가 다른 대화를 읽지 않으므로 혼입은 저장
  구조상 발생하지 않아야 한다.
* **스포일러** — `expert.spoiler.blocked`에 차단 사유가, 답변 문구 검사 결과가
  trace의 `Spoiler Policy / redacted`에 남는다.
* **운영 가능성** — `budget`과 기존 `telemetry`가 요청당 호출 수와 비용 원인을
  남긴다.

## 9. 아직 하지 않은 것

* 지원 범위 값(`verified_version`, `last_reviewed`)의 실제 검증.
* 검증용 50개 게임 집합과 100개 평가 요청의 구성.
* 이미지 입력, 계정 연동, 가격 실시간 확인(기획안 13.1의 P1 이후 항목).
* 사용자 인증. 현재 `user_id`는 요청 파라미터이며 기본값은 `local`이다.
