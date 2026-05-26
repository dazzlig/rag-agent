# Week 5 Retrospective & Retrieval Strategy ADR

## 0. 요약

이번 주차의 목표는 4주차에서 선택한 chunking 전략을 baseline으로 고정한 뒤, retrieval 단계를 고도화하는 것이었다.

구현한 핵심 내용은 다음과 같다.

- 4주차 최종 chunking 전략인 `B_recursive_large_1000_200`을 5주차 baseline으로 사용
- Metadata Filtering Retriever 구현
- SemanticChunker 보조 실험
- BM25 sparse retrieval 구현
- Dense retrieval과 BM25 retrieval을 RRF(Reciprocal Rank Fusion)로 결합
- `BAAI/bge-reranker-v2-m3` 기반 Cross-Encoder Re-ranker 적용
- Dense / BM25 / Hybrid RRF / Hybrid RRF + Re-ranker ablation 비교
- RAGAS 기반 평가 및 Error Case 분석
- 6주차 Agentic RAG 도입 근거 정리

최종적으로 선택한 retrieval 전략은 다음과 같다.

> **Hybrid RRF + BGE Cross-Encoder Re-ranker**

다만 4주차 최종 chunking 전략 B와 단순 비교하면 5주차 최종 점수는 일부 하락했다. 따라서 이번 주차 결과는 “절대 성능이 무조건 상승했다”가 아니라, **동일한 5주차 조건 안에서 Dense only 대비 Hybrid/Rerank가 retrieval 품질을 개선했고, 동시에 metadata filtering과 section routing의 필요성이 더 명확해졌다**고 해석하는 것이 타당하다.

---

## 1. 4주차 Baseline 요약

5주차 baseline은 4주차에서 가장 안정적인 답변 품질을 보였던 다음 chunking 전략으로 고정했다.

```text
B_recursive_large_1000_200
= section 분리 후 RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

4주차까지의 핵심 관찰은 다음과 같았다.

- 단순 chunking size 조정만으로도 Faithfulness와 Answer Relevancy가 개선되었다.
- 그러나 Context Precision은 일부 전략에서 하락했다.
- Steam 데이터는 `metadata`, `store_summary`, `about`, `review`, `news`가 섞여 있으므로, chunking만으로는 질문 의도에 맞는 section을 항상 상위에 배치하기 어렵다.
- 따라서 5주차에서는 retrieval 단계에서 keyword matching, hybrid search, reranking, metadata filtering을 실험했다.

---

## 2. Week 3 Baseline

3주차 baseline 점수는 다음과 같다.

| Metric | Week 3 Baseline |
|---|---:|
| Faithfulness | 0.9259 |
| Answer Relevancy | 0.4284 |
| Context Precision | 1.0000 |

3주차 baseline은 Context Precision이 높았지만, Answer Relevancy가 낮았다. 이는 검색된 context가 평가 기준상 관련성이 높게 잡히더라도, 최종 답변이 사용자의 질문 의도에 충분히 잘 맞지 않았을 가능성을 보여준다.

---

## 3. Week 4 Chunking 실험 결과

4주차 chunking 전략별 결과는 다음과 같다.

| Strategy | Faithfulness | Δ | Answer Relevancy | Δ | Context Precision | Δ |
|---|---:|---:|---:|---:|---:|---:|
| A_recursive_baseline_800_120 | 0.9667 | +0.0408 | 0.6564 | +0.2280 | 0.9625 | -0.0375 |
| B_recursive_large_1000_200 | 1.0000 | +0.0741 | 0.6669 | +0.2385 | 0.9347 | -0.0653 |
| C_markdown_header_recursive_800_120 | 0.9167 | -0.0092 | 0.4428 | +0.0144 | 0.9347 | -0.0653 |
| D_markdown_h2_recursive_1000_200 | 0.9697 | +0.0438 | 0.5431 | +0.1147 | 1.0000 | 0.0000 |

4주차에서는 `B_recursive_large_1000_200`이 Faithfulness 1.0000, Answer Relevancy 0.6669로 가장 높았다. Context Precision은 D 전략이 가장 높았지만, 답변 품질 지표를 함께 고려해 B 전략을 5주차 baseline으로 선택했다.

---

## 4. 왜 retrieval 단계 개선이 필요한가

4주차 결과에서 확인한 문제는 chunking 자체보다 **retrieval ranking과 context 구성**에 가까웠다.

Steam 게임 문서는 다음과 같은 특성을 가진다.

- 게임명, DLC명, 패치명, 태그명처럼 정확한 문자열 매칭이 중요한 정보가 있다.
- 플레이 스타일, 분위기, 전투 루프, 협동성처럼 의미 기반 검색이 중요한 질문도 많다.
- 같은 게임 문서 안에서도 `store_summary`, `about`, `review`, `news`는 서로 역할이 다르다.
- 리뷰 section은 실제 플레이 경험을 반영할 수 있으므로 gameplay 질문에 보조 근거가 될 수 있지만, 구조적 설명이 필요한 질문에서는 `about`/`store_summary`가 primary evidence가 되어야 한다.
- news section은 업데이트 질문에는 중요하지만, 일반적인 게임 특징 질문에서는 노이즈가 될 수 있다.

따라서 5주차에서는 Dense only를 넘어서 다음 구성을 실험했다.

1. BM25 sparse retrieval
2. Dense + BM25 Hybrid Search
3. RRF 기반 ranking fusion
4. Cross-Encoder 기반 Re-ranker
5. Metadata filtering과 section-aware retrieval의 필요성 분석

---

## 5. 구현 내용

### 5.1 Metadata Filtering Retriever

질문에서 게임명과 질문 의도를 rule 기반으로 추정했다.

- 게임명 추정: `Hollow Knight`, `Monster Hunter: World`, `Baldur's Gate 3`, `No Man's Sky`, `Cyberpunk 2077`
- 의도 추정:
  - `gameplay`: 플레이 스타일, 전투, 루프, 분위기, 월드, 협동, 서사 등
  - `review`: 리뷰, 평가, 반응, 민심 등
  - `news`: 업데이트, 패치, 뉴스 등
  - `general`: 그 외 일반 질문

초기에는 gameplay 질문에서 `about`, `store_summary`, `metadata`를 우선 section으로 보았다. 이후 Error Case 분석 과정에서 `review`도 gameplay에 대한 player sentiment를 보강하는 secondary evidence가 될 수 있음을 확인했다.

따라서 최종 개선 방향은 단순 hard filtering이 아니라 다음과 같은 evidence role 분리로 정리했다.

| Intent | Primary Evidence | Secondary Evidence | Noise 가능성 높은 Evidence |
|---|---|---|---|
| gameplay / 주요 특징 | `about`, `store_summary` | `metadata`, `review` | unrelated `news` |
| review / 반응 | `review` | `store_summary`, `about` | 다른 게임 `review`, unrelated `news` |
| news / update | `news` | `metadata` | `review`, unrelated links |
| general summary | `store_summary`, `about`, `metadata` | `review`, `news` | 다른 게임 chunk |

---

### 5.2 SemanticChunker 보조 실험

4주차 미완료 항목이었던 SemanticChunker도 보조 실험으로 적용했다.

다만 5주차의 핵심 비교는 retrieval 전략 비교이므로, chunking 변수를 새로 바꾸지 않고 4주차 B 전략을 계속 baseline으로 유지했다.

정리하면 다음과 같다.

- SemanticChunker는 의미 단위 분할 가능성을 확인하기 위한 보조 실험이다.
- 필수 ablation 비교에서는 `B_recursive_large_1000_200` chunk를 그대로 사용했다.
- 이렇게 해야 chunking 변화와 retrieval 변화가 섞이지 않는다.

---

### 5.3 BM25 Sparse Retrieval

BM25는 exact keyword matching을 보완하기 위해 추가했다.

Steam 데이터에서는 다음 유형의 질문에서 BM25가 유리할 수 있다.

- 특정 게임명 포함 질문
- 업데이트/패치/뉴스 제목 관련 질문
- 태그명, 장르명, DLC명, 고유명사 포함 질문
- 영문 원문과 사용자의 한국어/영어 혼합 질의가 함께 등장하는 경우

BM25는 매우 빠르게 동작했지만, 단독으로 사용할 경우 의미 기반 질문에서 context ranking이 불안정했다.

---

### 5.4 Hybrid Search: Dense + BM25 + RRF

Hybrid Search는 Dense retrieval과 BM25 retrieval의 결과를 각각 가져온 뒤 RRF로 결합했다.

```text
Dense top-k
+ BM25 top-k
→ Reciprocal Rank Fusion
→ Hybrid top-k
```

Dense retrieval은 의미 기반 유사도에 강하고, BM25는 exact keyword matching에 강하다. RRF는 서로 다른 scoring scale을 직접 비교하지 않고 rank만 이용해 결과를 결합하므로 Dense score와 BM25 score를 결합하기에 적합했다.

---

### 5.5 Cross-Encoder Re-ranker

Re-ranker는 Hybrid RRF로 가져온 top-20 후보를 대상으로 적용했다.

```text
Hybrid RRF top-20
→ BGE Cross-Encoder score 계산
→ score 기준 top-5 재정렬
```

사용 모델은 다음과 같다.

```text
BAAI/bge-reranker-v2-m3
```

이 모델을 선택한 이유는 Steam 데이터가 영어 중심 store/news 문서와 한국어 사용자 질의가 섞이는 다국어 검색 문제에 가깝기 때문이다. `bge-reranker-v2-m3`는 multilingual reranker로 제공되며, Steam 문서처럼 영어/한국어가 섞일 수 있는 corpus에 적합하다고 판단했다.

초기에는 `FlagEmbedding`의 `FlagReranker`를 사용하려 했으나, 로컬 환경의 `transformers 5.x`와 tokenizer 호환 문제로 오류가 발생했다. 따라서 동일한 모델 가중치를 유지하되, `transformers.AutoTokenizer`와 `AutoModelForSequenceClassification`으로 직접 로드해 Cross-Encoder reranking을 구현했다.

---

## 6. Week 5 Ablation 결과

5주차에서는 같은 질문 세트에 대해 다음 4가지 구성을 비교했다.

1. Dense only
2. BM25 only
3. Hybrid RRF
4. Hybrid RRF + BGE Re-ranker

결과는 다음과 같다.

| Retriever | Faithfulness | Answer Relevancy | Context Precision | Avg Latency(s) |
|---|---:|---:|---:|---:|
| BM25 only | 0.9833 | 0.6283 | 0.4806 | 0.0005 |
| Dense only | 0.8772 | 0.4574 | 0.6839 | 0.0288 |
| Hybrid RRF | 0.9444 | 0.5675 | 0.8389 | 0.0243 |
| Hybrid RRF + Re-ranker | 0.9603 | 0.5746 | 0.8475 | 0.2817 |

---

## 7. 결과 해석

### 7.1 5주차 내부 비교

5주차 내부 ablation만 보면 `hybrid_rerank`가 최종 전략으로 가장 적합하다.

- Context Precision이 0.8475로 가장 높다.
- Faithfulness도 0.9603으로 안정적이다.
- Dense only 대비 Context Precision이 크게 개선되었다.

```text
Dense only Context Precision: 0.6839
Hybrid RRF + Re-ranker Context Precision: 0.8475
개선폭: +0.1636
```

즉, 5주차의 목표였던 retrieval 고도화 관점에서는 Hybrid Search와 Re-ranker 적용이 효과가 있었다.

---

### 7.2 BM25 only 해석

BM25 only는 Faithfulness 0.9833, Answer Relevancy 0.6283으로 가장 높았다. 하지만 Context Precision은 0.4806으로 가장 낮았다.

이는 BM25가 특정 키워드가 잘 맞는 일부 질문에서는 좋은 답변을 만들 수 있지만, 전체 검색 결과의 상위 chunk 정렬 품질은 불안정하다는 의미로 해석했다.

따라서 BM25는 단독 전략보다는 Dense retrieval과 결합하는 보조 retrieval 방식으로 사용하는 것이 적합하다.

---

### 7.3 Hybrid RRF vs Hybrid RRF + Re-ranker

Hybrid RRF는 평균 latency 0.0243초로 매우 빠르면서 Context Precision 0.8389를 기록했다. Re-ranker를 추가하면 Context Precision이 0.8475로 소폭 상승했지만 latency는 0.2817초로 증가했다.

따라서 실제 서비스 환경에서는 다음과 같은 전략을 고려할 수 있다.

- 기본 질의: Hybrid RRF
- 정확도가 중요한 질의: Hybrid RRF + Re-ranker
- 특정 게임/section이 명확한 질의: metadata filter + Hybrid RRF + Re-ranker

---

## 8. 4주차와 5주차 비교

4주차 최종 전략 B와 5주차 최종 전략 `hybrid_rerank`를 단순 비교하면 다음과 같다.

| Metric | Week 4 B | Week 5 Hybrid RRF + Re-ranker | 변화 |
|---|---:|---:|---:|
| Faithfulness | 1.0000 | 0.9603 | -0.0397 |
| Answer Relevancy | 0.6669 | 0.5746 | -0.0923 |
| Context Precision | 0.9347 | 0.8475 | -0.0872 |

따라서 절대 점수만 보면 4주차 B 전략보다 5주차 최종 점수는 일부 하락했다.

하지만 4주차는 chunking 전략 비교였고, 5주차는 retrieval 전략 비교였다. 두 실험은 목적이 다르므로, 이를 단순히 성능 퇴보로만 해석하기는 어렵다. 5주차의 핵심 비교 대상은 4주차 결과가 아니라, 동일한 chunking 전략 B를 고정한 상태에서 Dense / BM25 / Hybrid / Hybrid+Rerank가 어떤 차이를 보이는지였다.

5주차 내부 비교에서는 `hybrid_rerank`가 Dense only보다 Context Precision을 크게 개선했다. 반면 4주차 B보다 점수가 낮아진 것은 Hybrid Search 과정에서 다른 게임 chunk, 부적절한 section, primary/secondary evidence 구분 부족 문제가 아직 남아 있음을 보여준다.

결론적으로 이번 주차는 다음과 같이 해석한다.

> **5주차는 4주차 대비 절대 점수 향상에 성공한 실험이라기보다, retrieval 전략 간 비교를 통해 Hybrid+Rerank의 상대적 장점과 metadata/section routing의 필요성을 확인한 실험이다.**

---

## 9. Error Case 분석

### Error Case 1. Monster Hunter: World gameplay 질문에서 primary evidence 순위가 약함

- 질문: `Monster Hunter: World의 핵심 플레이 루프는 무엇인가요?`
- 사용 retriever: `hybrid_rerank`
- RAGAS 결과:
  - Faithfulness: 1.0000
  - Answer Relevancy: 0.6676
  - Context Precision: 0.5000

검색 결과는 다음과 같았다.

| Rank | Game Key | Section | Chunk Index | 해석 |
|---:|---|---|---:|---|
| 1 | monster_hunter_world | metadata | 46 | 게임 기본 정보. 보조 근거로는 가능하지만 플레이 루프의 primary evidence로는 약함 |
| 2 | monster_hunter_world | about | 48 | 적절한 primary evidence |
| 3 | monster_hunter_world | review | 53 | 플레이어 경험을 보여주는 secondary evidence로 가능 |
| 4 | monster_hunter_world | news | 57 | gameplay 질문에서는 노이즈 가능성 높음 |
| 5 | monster_hunter_world | store_summary | 47 | 적절한 primary evidence |

이 케이스에서 review chunk가 들어온 것 자체를 무조건 오검색이라고 보기는 어렵다. 리뷰에는 실제 플레이어가 게임의 전투, 반복 루프, 협동성, 재미 요소를 평가한 내용이 포함될 수 있기 때문이다.

문제는 질문이 “핵심 플레이 루프”라는 구조적 설명을 요구하는데, `about`과 `store_summary`보다 `metadata`가 rank 1에 있고 `news`가 top-5에 포함되었다는 점이다. 즉, section 자체보다 **primary evidence와 secondary evidence의 역할 구분이 약한 것**이 문제다.

- 원인 가설:
  - Hybrid Search가 `Monster Hunter: World`라는 게임명 매칭을 강하게 반영하면서 같은 게임의 여러 section을 넓게 가져왔다.
  - Re-ranker가 section별 역할을 충분히 구분하지 못했다.
- 6주차 개선 가설:
  - Agentic RAG에서 intent를 `gameplay`로 분류한다.
  - `about`/`store_summary`를 primary evidence로 우선 검색한다.
  - `review`는 player sentiment를 보강하는 secondary evidence로 제한적으로 결합한다.
  - `news`는 업데이트 의도가 없는 gameplay 질문에서는 낮은 우선순위로 둔다.

---

### Error Case 2. Cyberpunk 2077 리뷰 질문에서 다른 게임 리뷰 혼입

- 질문: `Cyberpunk 2077 최근 리뷰에서는 어떤 반응이 있나요?`
- 사용 retriever: `hybrid_rerank`
- RAGAS 결과:
  - Faithfulness: 1.0000
  - Answer Relevancy: 0.5935
  - Context Precision: 0.8875

검색 결과는 다음과 같았다.

| Rank | Game Key | Section | Chunk Index | 해석 |
|---:|---|---|---:|---|
| 1 | cyberpunk_2077 | review | 24 | 적절함 |
| 2 | cyberpunk_2077 | review | 22 | 적절함 |
| 3 | no_mans_sky | review | 76 | 다른 게임 리뷰 |
| 4 | hollow_knight | review | 36 | 다른 게임 리뷰 |
| 5 | baldurs_gate_3 | review | 14 | 다른 게임 리뷰 |

이 케이스는 section은 `review`로 맞았지만, 질문 대상 게임이 아닌 다른 게임의 review chunk가 섞였다. 이는 명확한 entity filtering 실패 사례다.

- 원인 가설:
  - review intent는 잘 잡았지만 `game_key=cyberpunk_2077`이 hard filter로 강제되지 않았다.
  - 리뷰 section끼리는 “최근 반응”, “positive/negative”, “user feedback” 같은 표현이 유사해 다른 게임 리뷰가 후보군에 포함되었다.
- 6주차 개선 가설:
  - Agentic RAG에서 먼저 게임명을 추출해 `game_key=cyberpunk_2077`을 확정한다.
  - 이후 `section=review`와 함께 metadata filter를 적용한다.
  - 특정 게임이 명시된 질문에서는 다른 game_key chunk를 최종 context에서 제외한다.

---

### Error Case 3. Cyberpunk 2077 주요 특징 질문에서 primary/secondary evidence 구분 부족

- 질문: `Cyberpunk 2077 문서에서 확인되는 주요 특징은 무엇인가요?`
- 사용 retriever: `hybrid_rerank`
- RAGAS 결과:
  - Faithfulness: 0.8462
  - Answer Relevancy: 0.4454
  - Context Precision: 0.8333

검색 결과는 다음과 같았다.

| Rank | Game Key | Section | Chunk Index | 해석 |
|---:|---|---|---:|---|
| 1 | cyberpunk_2077 | store_summary | 20 | 적절한 primary evidence |
| 2 | cyberpunk_2077 | metadata | 19 | 일부 적절 |
| 3 | cyberpunk_2077 | review | 24 | 사용자 체감 특징을 보강할 수 있음 |
| 4 | cyberpunk_2077 | news | 28 | 일반 특징 요약에는 노이즈 가능성 있음 |
| 5 | cyberpunk_2077 | news | 27 | 일반 특징 요약에는 노이즈 가능성 있음 |

주요 특징 질문에서는 `store_summary`, `about`, `metadata`가 primary evidence가 되는 것이 적절하다. review는 실제 플레이어가 체감한 특징을 보강할 수 있으므로 secondary evidence로는 유효하다. 그러나 news는 특정 업데이트나 공지 중심일 수 있어 일반적인 게임 특징을 요약하는 데는 노이즈가 될 수 있다.

- 원인 가설:
  - 질문이 general intent로 해석되면서 같은 게임의 여러 section이 함께 검색되었다.
  - section별 evidence role을 구분하지 않고 relevance score만으로 top-5를 구성했다.
- 6주차 개선 가설:
  - Agentic RAG에서 질문 유형에 따라 primary evidence와 secondary evidence를 분리한다.
  - 예: `primary 3개 + secondary 2개`와 같은 section quota를 적용한다.
  - “주요 특징 요약” 질의는 `store_summary/about/metadata` 중심으로 검색하고, review/news는 필요할 때만 보조적으로 결합한다.

---

## 10. 최종 선택 Retrieval 전략

최종 retrieval 전략은 다음과 같다.

```text
Hybrid RRF + BGE Cross-Encoder Re-ranker
```

선택 이유는 다음과 같다.

1. Dense only보다 Context Precision이 크게 높다.
2. BM25 only보다 retrieval ranking이 안정적이다.
3. Hybrid RRF만으로도 성능이 좋지만, Re-ranker를 추가하면 top-5 context 품질이 소폭 개선된다.
4. Steam 데이터는 영어/한국어 혼합 질의 가능성이 있고, `BAAI/bge-reranker-v2-m3`는 multilingual reranker이므로 도메인 특성과 맞다.
5. latency는 증가하지만 현재 데이터 규모에서는 감당 가능한 수준이다.

다만 최종 서비스 관점에서는 모든 질의에 항상 reranker를 적용하기보다 다음과 같이 구분하는 것이 더 효율적일 수 있다.

| 상황 | 추천 Retrieval |
|---|---|
| 빠른 응답이 중요한 일반 질의 | Hybrid RRF |
| 정확한 답변이 중요한 질의 | Hybrid RRF + Re-ranker |
| 특정 게임이 명시된 질의 | Metadata Filter + Hybrid RRF + Re-ranker |
| 리뷰/뉴스처럼 section이 명확한 질의 | Intent Routing + Section Filter + Re-ranker |

---

## 11. ADR: Week 5 Retrieval Strategy

### Status

Accepted

---

### Context

Steam 게임 추천 RAG에서는 의미 기반 검색과 키워드 기반 검색이 모두 필요하다.

- 플레이 스타일, 분위기, 전투 루프, 서사처럼 의미적 해석이 필요한 질문이 있다.
- 게임명, 패치명, 뉴스 제목, 태그명처럼 정확한 키워드 매칭이 중요한 질문도 있다.
- 문서에는 `metadata`, `store_summary`, `about`, `review`, `news`가 함께 존재한다.
- 리뷰는 gameplay나 주요 특징 질문에서 secondary evidence가 될 수 있지만, 다른 게임 리뷰가 섞이면 오검색이다.
- 뉴스는 업데이트 질문에서는 primary evidence지만, 일반 특징 질문에서는 노이즈가 될 수 있다.

---

### Decision

최종 retrieval 전략으로 다음 방식을 채택한다.

```text
Hybrid RRF + BGE Cross-Encoder Re-ranker
```

구체적인 파이프라인은 다음과 같다.

1. 4주차 최종 chunking 전략 B로 chunk 생성
2. Dense retriever로 의미 기반 후보 검색
3. BM25 retriever로 키워드 기반 후보 검색
4. RRF로 Dense/BM25 결과 결합
5. Cross-Encoder Re-ranker로 top-20 후보를 top-5로 재정렬

---

### Rationale

`hybrid_rerank`는 5주차 ablation에서 Context Precision 0.8475로 가장 높았다. Dense only의 Context Precision 0.6839보다 크게 개선되었으며, BM25 only보다 검색 결과의 상위 chunk 정렬 품질이 안정적이었다.

BM25 only는 Faithfulness와 Answer Relevancy가 높았지만 Context Precision이 낮아, 단독 retrieval 전략으로는 불안정하다고 판단했다. 반면 Hybrid RRF는 Dense와 BM25의 장점을 결합해 빠르고 안정적인 결과를 보였다. Re-ranker는 latency를 증가시키지만, top-5 context 품질을 추가로 개선했다.

따라서 retrieval 품질을 우선하는 현재 과제에서는 `Hybrid RRF + BGE Cross-Encoder Re-ranker`를 최종 전략으로 선택한다.

---

### Trade-offs

| 항목 | 장점 | 포기한 것 / 비용 |
|---|---|---|
| Dense only 대비 | 키워드 matching 보완, Context Precision 개선 | BM25 인덱스 추가 관리 |
| BM25 only 대비 | 의미 기반 질문 대응력 개선 | sparse retrieval만큼 빠르지는 않음 |
| Hybrid RRF 대비 | top-5 context 품질 개선 | latency 증가 |
| Metadata filtering 미적용 대비 | 향후 오검색 감소 가능 | intent/entity 추출 로직 필요 |
| Re-ranker 적용 | 최종 후보 재정렬 가능 | 모델 로딩/추론 비용 증가 |

---

### Alternatives Considered

#### Dense only

구현이 단순하고 의미 기반 질문에 강하지만, 게임명/패치명/뉴스 제목 등 정확한 키워드가 중요한 질문에서 한계가 있었다.

#### BM25 only

매우 빠르고 exact keyword matching에 강하다. 그러나 Context Precision이 0.4806으로 가장 낮아, 전체 retrieval ranking 품질이 불안정했다.

#### Hybrid RRF without Re-ranker

Context Precision 0.8389로 충분히 좋은 성능을 보였고 latency도 낮았다. 실제 서비스에서 빠른 응답이 필요하면 좋은 대안이다. 다만 이번 과제에서는 retrieval 품질을 우선해 Re-ranker를 추가한 전략을 선택했다.

#### SemanticChunker 기반 재인덱싱

의미 단위 chunking을 시도할 수 있지만, 5주차의 핵심은 retrieval 전략 비교였기 때문에 chunking 변수를 새로 바꾸지 않았다.

---

### Consequences

이번 결정으로 retrieval 품질은 Dense only 대비 개선되었다. 그러나 Error Case 분석에서 다음 문제가 남아 있음을 확인했다.

- 특정 게임 질문에 다른 게임 chunk가 섞일 수 있다.
- gameplay/general 질문에서 primary evidence와 secondary evidence가 구분되지 않는다.
- news section이 일반 특징 질문에 노이즈로 포함될 수 있다.
- review section은 무조건 제거할 대상이 아니라, 질문 intent에 따라 보조 근거로 활용해야 한다.

따라서 6주차에서는 Agentic RAG를 통해 query understanding과 retrieval routing을 강화한다.

---

## 12. 6주차 Agentic RAG 개선 방향

6주차에서는 다음 방향으로 개선한다.

### 12.1 Entity Extraction

질문에서 게임명을 먼저 추출한다.

```text
Cyberpunk 2077 최근 리뷰에서는 어떤 반응이 있나요?
→ game_key = cyberpunk_2077
```

특정 게임이 명시된 질문에서는 해당 `game_key`를 hard filter로 적용한다.

---

### 12.2 Intent Routing

질문 intent를 먼저 분류한다.

```text
gameplay / review / news / general
```

intent에 따라 primary section과 secondary section을 다르게 구성한다.

---

### 12.3 Section Quota Retrieval

무조건 top-5를 점수순으로 가져오는 대신, section별 역할을 반영한다.

예시:

```text
gameplay 질문:
- primary: about/store_summary top-3
- secondary: review top-1~2
- metadata: 필요 시 1개
- news: 업데이트 의도 없으면 제외 또는 낮은 우선순위
```

```text
review 질문:
- primary: same game review top-4
- secondary: store_summary/about top-1
- other game review: 제외
```

---

### 12.4 Query Transformation

질문이 너무 넓거나 모호한 경우 query를 변환한다.

예시:

```text
원 질문:
Cyberpunk 2077 문서에서 확인되는 주요 특징은 무엇인가요?

변환 질의:
Cyberpunk 2077 gameplay features open-world action RPG story combat character progression
```

---

### 12.5 Agentic Retrieval Flow

6주차에서 목표로 하는 흐름은 다음과 같다.

```text
User Question
→ Entity Extractor
→ Intent Classifier
→ Query Transformer
→ Section-aware Retriever
→ Hybrid RRF
→ Re-ranker
→ Context Validator
→ Answer Generator
```

---

## 13. 제출 파일 정리

이번 주차 산출물은 다음과 같다.

```text
notebooks/week5_hybrid_reranking_ablation.ipynb

data/eval/week5_four_retriever_results.csv
data/eval/week5_rag_answers_for_ragas.csv
data/eval/week5_ragas_results_by_question.csv
data/eval/week5_final_ablation_summary.csv

docs/week5_retrospective.md
docs/adr/week5_retrieval_strategy.md
```

이번 요청에서는 회고와 ADR을 하나의 파일에 통합했으므로, 저장 파일은 다음과 같이 둘 수 있다.

```text
docs/week5_retrospective.md
```

또는 ADR을 별도 파일로 분리하려면 이 문서의 `ADR: Week 5 Retrieval Strategy` 섹션을 다음 경로로 옮긴다.

```text
docs/adr/week5_retrieval_strategy.md
```

---

## 14. 최종 발표 요약

이번 주차에서는 4주차에서 선택한 chunking 전략 B를 baseline으로 고정하고, Dense only, BM25 only, Hybrid RRF, Hybrid RRF + Re-ranker를 비교했다.

BM25 only는 매우 빠르고 일부 답변 지표가 높았지만 Context Precision이 낮아 단독 전략으로는 불안정했다. Dense only는 의미 검색에는 유효했지만, Steam 데이터처럼 게임명과 패치명, 뉴스 제목이 중요한 corpus에서는 한계가 있었다.

Hybrid RRF는 Dense와 BM25의 장점을 결합해 Context Precision을 크게 개선했다. 여기에 BGE Cross-Encoder Re-ranker를 추가한 `hybrid_rerank`가 5주차 내부 비교에서 가장 높은 Context Precision을 기록했기 때문에 최종 전략으로 선택했다.

다만 4주차 최종 chunking 전략 B와 비교하면 절대 RAGAS 점수는 일부 하락했다. 이는 Hybrid Search 과정에서 다른 게임 chunk나 부적절한 section이 후보군에 포함되는 문제가 아직 남아 있음을 보여준다. 따라서 6주차에서는 Agentic RAG를 도입해 게임명 추출, intent routing, section quota, query transformation을 적용할 계획이다.

---

## 15. 참고 자료

- RAGAS Metrics: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- RAGAS Context Precision: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
- RAGAS Faithfulness: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- BAAI bge-reranker-v2-m3: https://huggingface.co/BAAI/bge-reranker-v2-m3
- BGE Reranker v2 Documentation: https://bge-model.com/bge/bge_reranker_v2.html
