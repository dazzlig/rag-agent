# 20개 시나리오·42턴 운영 평가

`evaluate-conversations`는 각 턴의 응답 품질과 실제 운영 비용·호출량을 함께 기록한다.

## 기록되는 지표

- OpenAI: 전체/채팅/임베딩 호출 수, 입력·출력·전체·캐시 입력 토큰, 모델별 호출과 예상 비용
- Tavily: 논리 요청 수, 실제 외부 호출 수, 캐시 hit/miss, 사용 credit, 예상 비용
- Steam: 논리 요청 수, 재시도를 포함한 HTTP attempt 수, 성공/오류 수, endpoint별 호출량
- Corpus: 문서 확인, 신규 수집, 기존 문서 재사용, Chroma 인덱싱 횟수
- 성능: 전체 지연시간 mean/P50/P95/max와 범주별 평균 지연시간·비용·외부 호출 수

비용은 청구서가 아닌 로컬 추정치다. 기본 단가는 2026-07-14 기준으로 `gpt-5-mini` 입력
$0.25/1M tokens, 캐시 입력 $0.025/1M tokens, 출력 $2.00/1M tokens,
`text-embedding-3-small` 입력 $0.02/1M tokens,
Tavily Pay-as-you-go $0.008/credit를 사용한다. 모델 단가가 바뀌면
`STEAM_RAG_PRICING_JSON`, Tavily 단가가 바뀌면 `TAVILY_USD_PER_CREDIT` 환경 변수로 덮어쓴다.

- OpenAI: https://openai.com/index/introducing-gpt-5-for-developers/
- Embedding: https://developers.openai.com/api/docs/models/text-embedding-3-small
- Tavily: https://docs.tavily.com/documentation/api-credits

## 1. Golden Set 검증

외부 API를 호출하지 않는다.

```powershell
$env:PYTHONPATH='src'
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli evaluate-conversations --validate-only
```

## 2. 한 시나리오 smoke test

키·잔액·출력 경로를 먼저 검증한다.

```powershell
$env:PYTHONPATH='src'
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli evaluate-conversations --limit 1 --run-id smoke-01 --run-label "42턴 전 사전 점검" --cache-state mixed
```

## 3. 전체 20개 시나리오·42턴 실행

```powershell
$env:PYTHONPATH='src'
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli evaluate-conversations --run-id full-42-v1 --run-label "20개 시나리오 42턴 기준선" --cache-state mixed
```

## 결과 파일

- `data/eval/conversation_benchmark_details.jsonl`: 턴별 질문·답변·계약 점수·지연시간·telemetry
- `data/eval/conversation_benchmark_summary.json`: 전체/범주별 품질, P50/P95, 비용·호출·토큰 집계
- `data/eval/conversation_benchmark_manifest.json`: 실행 ID, Golden Set SHA-256, Git commit, 모델, 캐시 상태, 출력 경로

`--cache-state`는 기록용 라벨이며 캐시를 자동 삭제하지 않는다. 같은 조건을 비교하려면 run ID와
cache state를 반드시 기록한다. 기본 결과 경로는 이전 실행을 덮어쓰므로 장기 비교용 실행에서는
`--details-output`, `--summary-output`, `--manifest-output`에 고유 경로를 지정한다.

```powershell
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli evaluate-conversations `
  --run-id full-42-v1 `
  --cache-state mixed `
  --details-output data/eval/runs/full-42-v1-details.jsonl `
  --summary-output data/eval/runs/full-42-v1-summary.json `
  --manifest-output data/eval/runs/full-42-v1-manifest.json
```

## 4. 의미 품질 평가 준비

42턴 실행 결과의 `details.jsonl`에서 RAGAS와 LLM Judge 입력을 먼저 고정한다. 이 명령은 외부 API를 호출하지 않는다.

```powershell
$env:PYTHONPATH='src'
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli prepare-quality-evaluation `
  --details data\eval\runs\full-42-v1-details.jsonl `
  --output-dir data\eval\runs\full-42-v1-quality
```

생성 파일:

- `ragas-input.jsonl`: 답변과 실제 근거 본문이 모두 있는 턴만 포함한다.
- `judge-input.jsonl`: 42턴 전체와 이전 대화, expected/forbidden 계약, AppID, 근거를 포함한다.
- `selection-summary.json`: 선정 수, 제외 사유, ID 목록을 기록한다.

source 제목이나 URL만 있고 본문이 저장되지 않은 턴은 RAGAS context로 사용하지 않는다. RAGAS가 URL의 실제 페이지를 읽는 것이 아니기 때문에 주소 문자열을 context로 넣으면 faithfulness가 왜곡된다. 이런 턴은 LLM Judge에는 포함하되 `evidence persistence` 문제로 따로 본다.

## 5. RAGAS·LLM Judge 실행

먼저 각 엔진 1개씩 smoke 평가한다. OpenAI 비용이 발생한다.

```powershell
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli evaluate-quality `
  --prepared-dir data\eval\runs\full-42-v1-quality `
  --ragas-model gpt-4o-mini `
  --judge-model gpt-5-mini `
  --ragas-limit 1 `
  --judge-limit 1
```

smoke 결과를 확인한 뒤 같은 명령에서 limit를 생략하면 남은 턴을 이어서 평가한다. 기본 `--resume`이 완료된 ID를 건너뛰므로 중단 후 다시 실행해도 같은 턴을 재호출하지 않는다.

```powershell
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli evaluate-quality `
  --prepared-dir data\eval\runs\full-42-v1-quality `
  --ragas-model gpt-4o-mini `
  --judge-model gpt-5-mini
```

필요한 엔진만 `--engines ragas` 또는 `--engines judge`로 실행할 수 있다.

결과 파일:

- `ragas-results.jsonl`, `ragas-summary.json`
- `judge-results.jsonl`, `judge-summary.json`
- `quality-run-summary.json`

RAGAS는 기존 평가 설정인 `gpt-4o-mini`로 reference-free `faithfulness`, `answer_relevancy`를 계산한다. LLM Judge는 `gpt-5-mini`를 사용한다. 정답 답변과 정답 evidence가 없는 현재 Golden Set에서 context recall은 최종 지표로 사용하지 않는다. LLM Judge는 대상 게임 정확성, 요구 조건 충족, 근거 정합성, 최신성, 대화 연속성, 가독성을 각각 0~4점으로 평가한다. 실행 오류나 빈 답변은 추가 Judge 호출 없이 자동 실패 처리한다.

CLI의 RAGAS 기본 모델은 `gpt-4o-mini`, Judge 기본 모델은 `gpt-5-mini`로 서로 분리한다. `gpt-5-mini` 같은 기본 temperature 전용 모델을 RAGAS에 명시적으로 지정하는 경우에만 RAGAS 내부의 `temperature=0.01/0.3` 덮어쓰기를 전달하지 않는다. 두 RAGAS 지표가 `null`이면 완료가 아니라 오류로 기록하며, 다음 실행의 기본 `--resume` 동작에서 자동 재시도한다. 저장된 모델 또는 embedding 모델이 현재 실행 설정과 다르면 완료 결과라도 재사용하지 않는다. LLM Judge는 구조화 결과를 쓰기 전에 출력 한도에 도달하지 않도록 최소 추론 강도와 별도 completion 예산을 사용한다.

호환성 수정 후에는 먼저 1행씩 smoke test를 수행한다.

```powershell
c:\Users\asguug\.pyenv\pyenv-win\versions\3.11.9\python.exe -m steam_rag.cli evaluate-quality `
  --prepared-dir data\eval\runs\full-42-v1-quality `
  --ragas-model gpt-4o-mini `
  --judge-model gpt-5-mini `
  --ragas-limit 1 `
  --judge-limit 1
```

`ragas-summary.json`의 두 metric이 숫자이고 `evaluation_error_count`가 0인지, `judge-results.jsonl`의 새 행이 `status=complete`인지 확인한 뒤 전체 재개 명령을 실행한다. 결과 파일을 삭제하거나 `--no-resume`을 사용하지 않는다.

RAGAS와 Judge를 `--engines ragas`, `--engines judge`로 나누어 실행해도 `quality-run-summary.json`은 두 엔진의 최신 summary를 함께 보존한다. `invoked_engines`는 마지막 명령에서 실제 호출한 엔진을, `engines`는 통합 파일에 포함된 엔진을 뜻한다. RAGAS generation은 긴 비교 답변에서도 중간 종료되지 않도록 최대 2,400 completion tokens를 허용한다. 특정 행이 `LLMDidNotFinishException`으로 남으면 같은 RAGAS 명령을 다시 실행해 그 오류 행만 재평가한다.

## 6. 대화 상태·조건·근거 계약

소비자 웹과 42턴 평가 러너는 응답의 `conversation_state`를 다음 요청에 그대로 전달한다.

- `active_games`: 현재 턴에서 검증된 정식 게임명과 Steam AppID. 후속 상세 질문과 비교 검색의 허용 AppID가 된다.
- `recommendation_query`: 장르, 카테고리, 가격, 할인, 출시 상태 등 이전 추천의 hard constraint다.
- `similarity_spec`: 기준 게임 AppID와 must/should/excluded 유사 조건이다. `그중 협동만`, `턴제는 빼고` 같은 후속 문장은 이 계약 위에 delta로 적용한다.
- `last_mode`, `last_resolved_question`: intent routing과 재현성 확인에 사용한다.

추천 후보는 `RecommendationProfileIndex`의 hard filter를 통과한 항목만 출력한다. 응답의 `recommendation.hard_constraint_gate`에는 실제 적용 조건, 검사 프로필 수, hard-filter 일치 수와 최종 AppID를 남긴다. 단일 게임의 가격·할인·리뷰·패치 질문은 추천 표현이 일부 포함되어도 `research`로 우선 라우팅한다.

근거는 화면과 평가 목적을 분리해 저장한다.

- `sources`: UI 카드용 220자 스니펫
- `evidence_contexts`: 검색 당시 전체 청크 본문, 원본 metadata, AppID, section, date, score, stable `source_id`
- `claim_citations`: claim별 `evidence_ranks`와 실제 `source_id` 연결

`prepare-quality-evaluation`은 `evidence_contexts`가 있으면 이를 우선 사용하고, 이전 실행처럼 이 필드가 없는 경우에만 `sources`로 폴백한다. 따라서 개선 전후 RAGAS를 비교할 때는 새 run ID를 사용해야 하며, 기존 `full-42-v1` 결과와 섞어 resume하지 않는다.
