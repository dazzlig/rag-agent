# Week 6 Retrospective: Agentic RAG

## 1. 이번 주차 목표

6주차의 목표는 새로운 retriever를 추가하는 것이 아니라, 5주차에서 최종 선택한 retrieval 전략을 baseline으로 유지한 상태에서 LangGraph 기반 Agentic RAG 구조를 구현하는 것이었다.

5주차까지는 다음과 같은 고정된 RAG 파이프라인을 사용했다.

```text
question
→ hybrid_rerank retrieval
→ context 구성
→ answer generation
```

하지만 이 구조는 검색 결과가 질문에 충분한지 판단하거나, 검색 결과가 부족할 때 query를 다시 작성해 재검색하는 흐름이 없었다. 따라서 6주차에서는 다음 구조를 추가했다.

```text
retrieve
→ grade_documents
→ rewrite_query
→ retrieve retry
→ generate
```

이번 주차의 핵심은 기능을 많이 붙이는 것이 아니라, 5주차에서 해결하지 못한 retrieval 실패 케이스를 시스템 구조로 어떻게 다룰 것인지 실험하는 것이다.

---

## 2. 5주차 최종 Retrieval 전략

5주차에서는 4주차에서 선택한 chunking 전략을 고정한 뒤, retrieval 방식을 비교했다.

4주차 최종 chunking 전략은 다음과 같다.

```text
B_recursive_large_1000_200
= section 분리 후 RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

5주차에서 비교한 retrieval 구성은 다음 4가지였다.

| Retriever       | 설명                                              |
| --------------- | ----------------------------------------------- |
| Dense only      | Chroma vector search만 사용                        |
| BM25 only       | keyword 기반 sparse retrieval                     |
| Hybrid RRF      | Dense + BM25 결과를 RRF로 결합                        |
| Hybrid + Rerank | Hybrid RRF 후보를 BGE Cross-Encoder Re-ranker로 재정렬 |

5주차 최종 선택 전략은 다음과 같다.

```text
Hybrid RRF + BGE Cross-Encoder Re-ranker
RRF weight: BM25:Dense = 0.5:0.5
```

선택 이유는 다음과 같다.

첫째, Dense only는 플레이 스타일이나 분위기 같은 의미 기반 질문에는 유리했지만, 게임명, 패치명, 뉴스 제목, 태그명처럼 정확한 키워드 매칭이 중요한 질문에서는 한계가 있었다.

둘째, BM25 only는 일부 답변 지표는 높았지만 Context Precision이 낮았다. 즉, 키워드가 포함된 chunk를 빠르게 찾을 수는 있었지만, top-k 전체가 질문에 안정적으로 맞지는 않았다.

셋째, Hybrid RRF는 Dense와 BM25의 장점을 결합해 Context Precision을 개선했다. 여기에 BGE Cross-Encoder Re-ranker를 추가한 `hybrid_rerank`가 가장 높은 Context Precision을 보여 최종 전략으로 선택했다.

5주차 기준 결과는 다음과 같다.

| Retriever       | Faithfulness | Answer Relevancy | Context Precision | Avg Latency(s) |
| --------------- | -----------: | ---------------: | ----------------: | -------------: |
| BM25 only       |       0.9833 |           0.6283 |            0.4806 |         0.0005 |
| Dense only      |       0.8772 |           0.4574 |            0.6839 |         0.0288 |
| Hybrid RRF      |       0.9444 |           0.5675 |            0.8389 |         0.0243 |
| Hybrid + Rerank |       0.9603 |           0.5746 |            0.8475 |         0.2817 |

따라서 6주차 Agentic RAG의 baseline retriever는 5주차 최종 전략인 `Hybrid RRF + BGE Cross-Encoder Re-ranker`를 그대로 사용했다.

---

## 3. 6주차 구현 구조

6주차에서는 5주차 코드를 제거하지 않고, 5주차 최종 retrieval 파이프라인 위에 LangGraph workflow를 추가했다.

구현된 흐름은 다음과 같다.

```text
START
→ retrieve
→ grade_documents
→ relevant이면 generate
→ not_relevant이면 rewrite_query
→ retrieve 재시도
→ retry_count 초과 시 generate에서 답변 불가 처리
→ END
```

각 노드의 역할은 다음과 같다.

| Node            | 역할                                    |
| --------------- | ------------------------------------- |
| retrieve        | 5주차 최종 `hybrid_rerank` retriever 호출   |
| grade_documents | 검색된 문서가 질문에 답변하기 충분한지 판단              |
| rewrite_query   | 검색 결과가 부족할 때 검색 친화적인 query로 재작성       |
| generate        | 충분한 context가 있으면 답변 생성, 부족하면 답변 불가 처리 |

실제 코드에서는 `week5_final_retrieve()`가 5주차 최종 retriever를 감싼다.

```text
week5_final_retrieve
→ retrieve_hybrid_rerank
→ retrieve_hybrid_rrf
→ retrieve_dense_only + retrieve_bm25_only
→ expand_query_for_bm25
→ reciprocal_rank_fusion
→ BGE Cross-Encoder Re-ranker
```

즉, 6주차 구현은 5주차 retrieval을 대체한 것이 아니라, 5주차 retrieval을 Agentic workflow 안에 넣은 구조다.

---

## 4. 5주차 이후에도 남은 실패 케이스

### 4-1. Error Case 1: Monster Hunter gameplay 질문에서 primary evidence 순위가 약함

질문:

```text
Monster Hunter: World의 핵심 플레이 루프는 무엇인가요?
```

5주차에서는 game_key는 대체로 맞췄지만, top-k 결과 안에 `metadata`, `review`, `news`, `about`, `store_summary`가 함께 섞였다.

이때 `review` chunk가 들어온 것 자체를 무조건 오검색이라고 보기는 어렵다. 리뷰에는 실제 플레이어가 느낀 전투 루프, 반복 플레이, 협동 경험 등이 포함될 수 있기 때문이다.

다만 이 질문은 “핵심 플레이 루프”라는 구조적 설명을 요구하므로, `about`과 `store_summary`가 primary evidence로 우선 배치되고, `review`는 secondary evidence로 사용되는 것이 더 적절하다. 반면 `news`는 업데이트나 프로모션 성격일 수 있으므로 gameplay 질문에서는 노이즈가 될 가능성이 높다.

6주차에서는 이를 반영해 gameplay intent에 대해 다음 section policy를 적용했다.

```text
primary: about, store_summary
secondary: metadata, review
avoid: news
```

---

### 4-2. Error Case 2: Cyberpunk 2077 리뷰 질문에서 다른 게임 리뷰가 섞임

질문:

```text
Cyberpunk 2077 최근 리뷰에서는 어떤 반응이 있나요?
```

이 케이스는 명확한 retrieval error다.

질문은 `Cyberpunk 2077`의 최근 리뷰 반응을 묻고 있으므로 `section=review`가 검색되는 것은 맞다. 하지만 검색 결과에 `No Man's Sky`, `Hollow Knight`, `Baldur's Gate 3` 등 다른 게임의 review chunk가 들어오면 답변 근거로 사용할 수 없다.

즉, 이 문제는 section 문제가 아니라 `game_key` filtering 문제다.

6주차에서는 `grade_documents` 단계에서 특정 게임 질문의 경우 같은 `game_key` 문서 비율을 검사했다. 같은 게임 문서 비율이 기준보다 낮으면 `not_relevant`로 판단하고 query rewrite를 수행하도록 했다.

---

### 4-3. Error Case 3: Cyberpunk 2077 주요 특징 질문에서 primary/secondary evidence 구분 부족

질문:

```text
Cyberpunk 2077 문서에서 확인되는 주요 특징은 무엇인가요?
```

이 질문은 특정 게임의 일반적인 특징을 요약하는 질문이다. 따라서 `store_summary`, `about`, `metadata`가 primary evidence가 되어야 한다.

다만 `review` chunk도 완전히 배제할 필요는 없다. 리뷰는 플레이어가 실제로 체감한 특징을 보강하는 secondary evidence로 사용할 수 있다. 반면 `news` chunk는 업데이트나 공지 중심일 가능성이 높기 때문에 일반 특징 요약에서는 노이즈가 될 수 있다.

6주차에서는 general 또는 gameplay 성격의 질문에서 다음과 같이 evidence role을 구분했다.

```text
primary: store_summary, about, metadata
secondary: review
avoid: news
```

---

## 5. 왜 단일 RAG 파이프라인이 아니라 Agentic 구조가 필요한가?

5주차의 Hybrid + Rerank 구조는 Dense와 BM25의 장점을 결합하고 reranker로 top-k 품질을 개선했지만, 검색 결과가 질문에 충분한지 스스로 판단하지는 못했다. 예를 들어 특정 게임의 리뷰를 물었는데 다른 게임 리뷰가 섞이거나, gameplay 질문에서 구조적 근거보다 review/news가 앞서는 경우가 있었다. 단일 RAG 파이프라인은 한 번 검색된 context를 그대로 답변 생성에 넘기기 때문에, 검색 결과가 부족하거나 잘못되었을 때 query를 다시 작성하거나 답변을 거절하는 제어 흐름이 약하다. 반면 Agentic RAG는 `retrieve → grade_documents → rewrite_query → retrieve 재시도 → generate` 구조를 통해 검색 결과를 평가하고, 부족하면 query를 변환해 재검색하며, 그래도 근거가 부족하면 답변을 생성하지 않는 방식으로 hallucination 위험을 줄일 수 있다. 따라서 이번 6주차에서는 retrieval 모델 자체를 바꾸기보다, 5주차에서 해결하지 못한 실패 케이스를 시스템 구조로 다루기 위해 Agentic RAG를 도입했다.

---

## 6. 6주차 코드에서 추가한 개선 사항

### 6-1. Section role 기반 context 구성

기존에는 retriever가 반환한 top-k 문서를 그대로 context로 사용했다. 하지만 5주차 에러 케이스를 보면 section 자체가 맞고 틀림의 문제가 아니라, 질문 intent에 따라 section의 역할이 달라지는 문제가 있었다.

따라서 6주차에서는 `get_section_policy()`를 추가해 intent별 section 역할을 정의했다.

```text
gameplay
- primary: about, store_summary
- secondary: metadata, review
- avoid: news

review
- primary: review
- secondary: store_summary, about, metadata
- avoid: news

news
- primary: news
- secondary: metadata
- avoid: review, about, store_summary

general
- primary: store_summary, about, metadata
- secondary: review
- avoid: news
```

또한 `select_docs_by_section_role()`을 통해 raw retrieval 결과를 그대로 사용하지 않고, primary evidence를 우선 배치한 뒤 secondary evidence를 보강하는 방식으로 최종 context를 구성했다.

---

### 6-2. grade_documents에서 primary evidence 여부 판단

`grade_documents` 노드는 검색된 문서가 충분한지 판단한다.

이번 구현에서는 다음 조건을 사용했다.

1. 검색 문서가 없으면 `not_relevant`
2. 특정 게임 질문에서 같은 `game_key` 비율이 낮으면 `not_relevant`
3. intent별 primary section이 하나도 없으면 `not_relevant`
4. avoid section이 많고 primary evidence가 약하면 `not_relevant`
5. 위 rule check를 통과하면 LLM Judge로 최종 yes/no 판단

이렇게 구성한 이유는 LLM Judge만 사용할 경우 판단 기준이 흔들릴 수 있고, 반대로 rule만 사용하면 문서 내용의 실제 관련성을 보지 못하기 때문이다. 따라서 rule-based check와 LLM Judge를 결합했다.

---

### 6-3. Query Rewrite + Retry

검색 결과가 부족하다고 판단되면 `rewrite_query` 노드로 이동한다.

`rewrite_query`는 답변을 생성하지 않고, 검색에 더 적합한 질의로 변환하는 역할만 한다.

예를 들어 gameplay 질문이면 다음과 같은 영어 키워드를 보강하도록 했다.

```text
gameplay, combat, loop, mechanics, store summary, about
```

리뷰 질문이면 다음 키워드를 보강한다.

```text
recent reviews, user reaction, sentiment
```

업데이트 질문이면 다음 키워드를 보강한다.

```text
update, patch, news
```

단, retry가 무한 반복되지 않도록 `retry_count`를 두었고, 최대 재시도 횟수는 2회로 제한했다.

---

### 6-4. 답변 불가 처리

재검색 후에도 충분한 context를 찾지 못하면 억지로 답변하지 않고 다음과 같이 답변하도록 했다.

```text
제공된 문서에서는 해당 질문에 대한 근거를 확인할 수 없습니다. 추가 문서가 제공되면 더 정확히 답변할 수 있습니다.
```

이는 RAG 시스템에서 근거 없는 답변 생성을 줄이기 위한 안전장치다.

---

## 7. Workflow Diagram

이번 주 구현한 LangGraph workflow는 다음 파일에 저장했다.

```text
docs/week6_workflow_diagram.md
docs/week6_workflow_diagram.png
```

흐름은 다음과 같다.

```text
START
→ retrieve
→ grade_documents
→ relevant이면 generate
→ not_relevant이고 retry_count < max_retry이면 rewrite_query
→ rewrite_query 후 retrieve 재시도
→ retry_count 초과 시 generate에서 답변 불가 처리
→ END
```

---

## 8. Baseline RAG vs Agentic RAG 정량 비교

동일한 평가 질문 세트를 사용해 5주차 최종 RAG와 6주차 Agentic RAG를 비교한다.

비교 조건은 다음과 같이 고정했다.

```text
질문 세트 동일
embedding 모델 동일
LLM 동일
vector DB 동일
chunking 전략 동일
baseline retriever 동일
변경점은 LangGraph 기반 routing / grading / rewrite 구조
```

현재까지의 비교 결과는 다음과 같다.

| 구성                            |  Faithfulness | Answer Relevancy | Context Precision | 평균 Latency(s) |
| ----------------------------- | ------------: | ---------------: | ----------------: | ------------: |
| Baseline: 5주차 Hybrid + Rerank |        0.9603 |           0.5746 |            0.8475 |        0.2817 |
| Agentic RAG                   | RAGAS 실행 후 입력 |    RAGAS 실행 후 입력 |     RAGAS 실행 후 입력 |       13.7370 |

Agentic RAG의 평균 latency는 `week6_agentic_rag_answers_for_ragas.csv` 기준 10개 질문 평균 약 13.737초로 측정되었다. 이는 baseline retriever 자체의 latency뿐 아니라 `grade_documents` LLM Judge, query rewrite, 재검색, 최종 generate까지 포함한 end-to-end latency다.

Agentic RAG의 RAGAS 점수는 `week6_agentic_rag_answers_for_ragas.csv`를 기반으로 별도 RAGAS 평가를 실행한 뒤 채울 예정이다.

---

## 9. 운영 관점 회고

### 9-1. 재검색이 실제로 도움이 된 사례

이번 실행 결과에서 `retry_count > 0`인 질문은 다음 1개였다.

```text
Cyberpunk 2077 최근 리뷰에서는 어떤 반응이 있나요?
```

초기 검색 결과에서는 같은 게임 문서 비율이 낮아 `game_key_mismatch`로 판단되었다.

실제 route history는 다음 흐름을 보였다.

```text
retrieve
→ grade: game_key_mismatch
→ rewrite
→ retrieve
→ grade: relevant
→ generate
```

즉, 5주차 Error Case 2였던 “Cyberpunk 리뷰 질문에서 다른 게임 리뷰가 섞이는 문제”를 `grade_documents`가 감지했고, query rewrite를 통해 재검색을 수행했다. 이 점에서 Agentic RAG의 retry 구조는 retrieval error를 완화하는 데 도움이 되었다.

다만 최종 context에도 일부 다른 게임 review나 news chunk가 남을 수 있었다. 따라서 이후에는 특정 게임 질문에서 section role 분류 전에 `game_key` hard filtering을 먼저 적용하는 방향이 필요하다.

---

### 9-2. 재검색이 불필요했거나 오히려 악영향을 줄 수 있는 사례

나머지 9개 질문은 `retry_count=0`이었다. 즉, 처음 검색 결과가 `grade_documents`에서 충분하다고 판단되어 바로 `generate`로 이동했다.

예를 들어 다음 질문들은 재검색 없이 답변이 생성되었다.

```text
Hollow Knight는 어떤 플레이 스타일의 게임인가요?
Monster Hunter: World의 핵심 플레이 루프는 무엇인가요?
No Man's Sky는 업데이트와 관련해서 어떤 내용이 있나요?
```

이런 질문들은 특정 section의 primary evidence가 비교적 명확하게 검색되었기 때문에 Agentic retry가 필요하지 않았다. 만약 이런 질문에서도 LLM Judge가 과도하게 엄격하게 작동해 query rewrite가 발생했다면, 답변 품질은 크게 개선되지 않으면서 latency만 증가했을 가능성이 있다.

따라서 모든 질문에 Agentic retry가 필요한 것은 아니다. 검색 결과가 명확한 질문은 baseline Hybrid + Rerank만으로도 충분할 수 있다.

---

### 9-3. Baseline 대비 latency 변화

5주차 baseline인 Hybrid + Rerank의 평균 latency는 약 0.2817초였다. 반면 6주차 Agentic RAG의 평균 latency는 약 13.7370초였다.

| 구성                        | 평균 Latency(s) |
| ------------------------- | ------------: |
| Baseline: Hybrid + Rerank |        0.2817 |
| Agentic RAG               |       13.7370 |

Agentic RAG의 latency가 크게 증가한 이유는 다음과 같다.

1. `grade_documents`에서 LLM Judge를 호출한다.
2. 검색 실패 시 `rewrite_query`에서 LLM을 추가 호출한다.
3. retry가 발생하면 retriever를 다시 호출한다.
4. 최종 답변 생성까지 포함한 end-to-end 시간을 측정했다.

따라서 Agentic RAG는 성능 개선 가능성이 있지만, 운영 관점에서는 latency와 비용 증가가 명확한 trade-off다.

---

### 9-4. Agentic RAG가 본 도메인에 필요한가?

Steam 게임 추천 RAG 도메인에서는 Agentic RAG가 필요하다. 이유는 게임 문서가 `metadata`, `store_summary`, `about`, `review`, `news`처럼 성격이 다른 section으로 구성되어 있고, 질문 intent에 따라 적절한 evidence role이 달라지기 때문이다.

예를 들어 gameplay 질문에서는 `about/store_summary`가 primary evidence이고 `review`는 secondary evidence가 될 수 있다. 리뷰 질문에서는 `review`가 primary evidence가 된다. 업데이트 질문에서는 `news`가 primary evidence가 된다.

다만 모든 질문에 Agentic RAG가 필요한 것은 아니다. Agentic RAG는 latency와 LLM 호출 비용을 증가시키기 때문에, 단순한 게임 설명 질문이나 검색 결과가 명확한 질문에는 baseline Hybrid + Rerank를 사용하고, `game_key` 혼입, section role 구분, context 부족이 예상되는 질문에만 선택적으로 적용하는 것이 더 현실적이다.

---

## 10. 실행 결과 요약

이번 실행 결과는 다음과 같다.

| 항목                      |       결과 |
| ----------------------- | -------: |
| 평가 질문 수                 |       10 |
| `grade_result=relevant` |       10 |
| `retry_count=0`         |        9 |
| `retry_count=1`         |        1 |
| `retry_count=2`         |        0 |
| 평균 latency              | 13.7370s |

해석하면, 대부분의 질문은 최초 검색 결과만으로도 답변 생성이 가능하다고 판단되었다. 하지만 Cyberpunk 2077 리뷰 질문에서는 game_key mismatch가 감지되어 query rewrite가 수행되었고, 이후 답변 생성까지 이어졌다.

이는 6주차 Agentic RAG가 5주차에서 발견한 retrieval 실패 케이스 중 일부를 실제로 감지하고 복구할 수 있음을 보여준다.

---

## 11. 이번 주 구현의 한계

이번 구현은 Agentic RAG의 최소 구조를 구현하는 데 초점을 맞췄다. 따라서 다음과 같은 한계가 있다.

1. `grade_documents`의 LLM Judge 결과가 항상 안정적인 것은 아니다.
2. Query rewrite가 항상 더 좋은 검색 결과를 보장하지는 않는다.
3. 현재 section role 기반 재구성을 적용했지만, 특정 게임 질문에서 다른 게임 chunk를 완전히 제거하지는 못했다.
4. review 질문에서 news chunk가 context에 포함되면 답변 초점이 흐려질 수 있다.
5. retry가 발생하면 latency와 비용이 증가한다.
6. 모든 질문에 Agentic RAG를 적용하면 불필요한 LLM call이 발생할 수 있다.
7. 아직 RAGAS 외의 routing accuracy, refusal accuracy, citation accuracy 평가는 포함하지 않았다.

---

## 12. 다음 개선 방향

다음 단계에서는 다음을 개선할 계획이다.

### 12-1. game_key hard filtering 강화

현재 section role 기반 context 구성을 적용했지만, 특정 게임 질문에서 다른 게임의 review chunk가 남을 수 있었다. 따라서 `select_docs_by_section_role()`에서 section 분류 전에 `game_key` hard filtering을 먼저 적용하는 것이 필요하다.

개선 방향은 다음과 같다.

```text
1. 질문에서 game_key 추출
2. 검색 결과 중 같은 game_key 문서가 있으면 우선 사용
3. 그 안에서 primary / secondary / avoid section 분류
4. 같은 게임 문서가 너무 부족할 때만 다른 게임 문서를 fallback으로 사용
```

특히 review intent에서는 다른 게임 review를 허용하지 않는 것이 더 안전하다.

---

### 12-2. review intent에서 news 사용 제한

`Cyberpunk 2077 최근 리뷰` 질문에서는 답변에 news 기반 부가 설명이 포함될 수 있었다. 하지만 사용자는 리뷰 반응을 물었기 때문에, news나 판매 성적은 main evidence가 되면 안 된다.

따라서 review intent에서는 다음 정책을 적용하는 것이 더 적절하다.

```text
primary: same game review
secondary: store_summary/about/metadata
exclude or fallback only: news
```

---

### 12-3. Agentic RAG 선택 적용

모든 질문에 Agentic RAG를 적용하면 latency가 증가한다. 따라서 운영 관점에서는 다음과 같이 분기하는 것이 좋다.

```text
단순 설명 질문 / 검색 결과 명확함 → baseline Hybrid + Rerank
특정 게임 리뷰 / section 혼입 가능성 높음 → Agentic RAG
검색 결과 부족 / context confidence 낮음 → Agentic RAG
```

---

## 13. 7주차 Evaluation으로 이어지는 점

7주차에는 Agentic RAG가 실제로 도움이 되는 질문 유형과 그렇지 않은 질문 유형을 나누어 평가할 예정이다.

추가로 검토할 평가 항목은 다음과 같다.

| 평가 항목                        | 설명                                                     |
| ---------------------------- | ------------------------------------------------------ |
| Retrieval quality            | 검색된 context가 질문에 적절한지 평가                               |
| Routing accuracy             | grade_documents가 relevant / not_relevant를 적절히 판단했는지 평가 |
| Rewrite effectiveness        | query rewrite 후 context 품질이 개선되었는지 평가                  |
| Refusal accuracy             | 근거 부족 상황에서 답변 거절을 잘했는지 평가                              |
| Latency / Cost               | Agentic 구조 도입으로 증가한 시간과 비용 측정                          |
| Citation / Evidence accuracy | 답변이 실제 context에 근거했는지 평가                               |

특히 다음 세 가지 질문군을 나누어 평가할 계획이다.

```text
1. Baseline RAG로 충분한 질문
2. Agentic RAG가 도움이 되는 질문
3. Agentic RAG가 오히려 latency만 증가시키는 질문
```

---

## 14. 최종 정리

6주차에서는 5주차 최종 retrieval 전략인 `Hybrid RRF + BGE Cross-Encoder Re-ranker`를 baseline으로 유지한 채, LangGraph 기반 Agentic RAG 구조를 추가했다.

이번 구현의 핵심은 retriever 자체를 바꾸는 것이 아니라, 검색 결과를 평가하고 부족할 경우 query rewrite와 재검색을 수행하며, 끝까지 근거가 부족하면 답변을 거절하는 구조를 만든 것이다.

또한 5주차 에러 케이스 분석을 반영해 `review` section을 무조건 제거하지 않고, 질문 intent에 따라 primary evidence와 secondary evidence를 나누었다. 이를 통해 gameplay/general 질문에서는 `about/store_summary/metadata`를 primary evidence로, `review`를 secondary evidence로 활용하고, 리뷰 질문에서는 `review`를 primary evidence로 사용하도록 구성했다.

실행 결과, 10개 평가 질문 중 1개 질문에서 query rewrite가 발생했고, 해당 질문은 5주차에서 문제로 봤던 Cyberpunk 2077 리뷰 질문이었다. 따라서 Agentic RAG는 적어도 일부 retrieval error를 감지하고 복구하는 데 도움이 되었다.

다만 평균 latency가 크게 증가했고, 최종 context에 일부 noise가 남을 수 있었다. 따라서 실제 서비스 관점에서는 모든 질문에 Agentic RAG를 적용하기보다, 검색 결과가 불확실하거나 특정 게임/section 혼입 가능성이 높은 질문에 선택적으로 적용하는 것이 더 적절하다.
