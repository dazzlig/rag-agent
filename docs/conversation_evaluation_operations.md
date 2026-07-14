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
