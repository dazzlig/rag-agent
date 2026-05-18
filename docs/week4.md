# Week 4 Retrospective

## 1. 데이터 품질 진단 요약

이번 4주차 데이터 품질 진단은 `data/docs/`에 저장된 Steam 게임 Markdown 문서를 대상으로 수행했다.

전체 문서 수는 5개이며, 총 토큰 수는 약 16,213개이다.  
문서별 토큰 수는 최소 2,146개, 최대 4,047개, 평균 3,242.6개로 확인되었다.

문서 구조는 Markdown 기반이며, 주요 섹션은 다음과 같다.

- Metadata
- Store Summary
- About The Game
- Recent Steam Reviews
- Steam News and Updates

현재 데이터는 표나 이미지보다는 텍스트 중심으로 구성되어 있다.  
각 문서는 게임 단위로 분리되어 있고, section 구조가 명확하기 때문에 단순 길이 기준 chunking만 적용하기보다는 Markdown 헤더와 section metadata를 함께 활용하는 방식이 적합하다고 판단했다.

---

## 2. News 데이터 진단

Steam News 데이터는 appid 기준으로 수집했기 때문에 수집 자체가 잘못된 것은 아니다.  
다만 전체 News 25개 중 뉴스 제목에 게임명이 직접 포함되지 않은 항목은 11개로 확인되었다.

이 항목들을 모두 노이즈로 단정하기는 어렵다.  
예를 들어 `Hotfix`, `Patch Version`, `Update`와 같은 제목은 게임명이 직접 포함되지 않아도 실제 패치나 업데이트 공지일 가능성이 높다.

반면 `Steam Global Top Sellers`, `Related Promotional Information`, 다른 시리즈의 `Pre-Order` 뉴스처럼 특정 게임의 플레이 스타일이나 업데이트 내용을 묻는 질문에는 검색 품질을 떨어뜨릴 수 있는 항목도 존재한다.

따라서 이번 분석에서는 `게임명 미포함 = 노이즈`로 처리하지 않고, 검색 목적에 따라 relevance type을 나누는 방식이 더 적절하다고 판단했다.

News relevance type 분포는 다음과 같다.

| Relevance Type | Count |
|---|---:|
| valid_update_or_patch | 11 |
| manual_review_needed | 8 |
| store_or_sales_related | 3 |
| franchise_or_promotion_related | 2 |
| support_or_platform_notice | 1 |

현재 데이터의 핵심 문제는 수집 오류라기보다는, 서로 다른 성격의 정보가 같은 News 섹션에 함께 들어가 있다는 점이다.

---

## 3. 잘못 검색된 chunk 예시

### 예시 1. Monster Hunter: World 질문

질문:

```text
Monster Hunter: World의 핵심 플레이 루프는 무엇인가요?
```

초기 검색 결과에서는 `about`, `store_summary`보다 `news` 섹션의 Monster Hunter Stories 3 관련 chunk가 상위에 검색되는 경우가 있었다.

이 chunk가 잘못 검색되었다고 본 이유는 질문 의도가 `gameplay`에 가깝기 때문이다.  
즉, 사용자는 Monster Hunter: World의 플레이 루프나 전투 구조를 묻고 있는데, 다른 시리즈의 프로모션 뉴스는 직접적인 답변 근거로 적합하지 않다.

원인은 다음과 같이 볼 수 있다.

- 같은 Monster Hunter IP라 semantic similarity가 높게 잡힘
- `game_key`만으로는 news 내부의 세부 relevance를 구분하기 어려움
- gameplay 질문인데도 news 섹션이 함께 검색됨

---

### 예시 2. No Man's Sky 업데이트 질문

질문:

```text
No Man's Sky는 업데이트와 관련해서 어떤 내용이 있나요?
```

검색 결과 대부분은 Xeno Arena 업데이트와 관련되어 적절했다.  
하지만 일부 chunk에는 `RELATED LINKS`나 `Read the rest of the story...`처럼 부가 링크 성격의 텍스트가 포함되어 있었다.

이 텍스트는 No Man's Sky와 관련은 있지만, 핵심 업데이트 내용 자체는 아니다.  
따라서 답변 생성 시 관련 링크 문구까지 업데이트 내용처럼 포함될 가능성이 있다.

원인은 다음과 같이 볼 수 있다.

- News 본문과 관련 링크 텍스트가 함께 저장됨
- 업데이트 내용과 부가 링크가 같은 chunk에 섞임
- News 섹션 내부에서 본문/링크/프로모션 구분이 없음

---

## 4. 전처리 및 검색 개선 가설

이번 데이터 품질 진단을 바탕으로 다음과 같은 개선 가설을 세웠다.

1. 플레이 스타일 질문에서는 `About The Game`, `Store Summary` 섹션을 우선 검색한다.
2. 최근 리뷰나 유저 반응 질문에서는 `Recent Steam Reviews` 섹션을 우선 검색한다.
3. 업데이트 관련 질문에서는 `Steam News and Updates` 섹션 중 `valid_update_or_patch` 유형을 우선 검색한다.
4. 판매 순위, 프로모션, 다른 시리즈 관련 뉴스는 기본 추천 근거에서는 낮은 우선순위로 처리한다.
5. 향후 5주차 retrieval 고도화에서는 `section`, `game_key`, `relevance_type` metadata를 활용해 질문 의도별 필터링 검색을 적용한다.

결론적으로 4주차 chunking 실험에서는 문서의 Markdown 구조와 metadata를 보존하는 chunking 전략이 필요하다고 판단했다.

---

## 5. Chunking 전략 비교

이번 실험에서는 동일한 질문 세트로 chunking 전략만 바꿔 RAGAS 점수를 비교했다.

| Strategy | Description |
|---|---|
| A_recursive_baseline_800_120 | 3주차 baseline. 섹션 분리 후 Recursive 800/120 적용 |
| B_recursive_large_1000_200 | 섹션 분리 후 Recursive 1000/200 적용 |
| C_markdown_header_recursive_800_120 | MarkdownHeaderTextSplitter로 헤더 기준 분리 후 Recursive 800/120 적용 |
| D_markdown_h2_recursive_1000_200 | H2 기준 Markdown 분리 후 Recursive 1000/200 적용 |

전략 C는 본 도메인 데이터가 Markdown 구조를 가지고 있기 때문에 선택했다.  
Markdown 문서는 섹션 구조가 명확하므로, 헤더 기반 splitter가 문서 구조를 보존하는 데 유리할 것이라고 판단했다.

다만 실제 실험 결과, H3 단위까지 분리하면 리뷰와 뉴스가 너무 잘게 쪼개지는 문제가 있었다.  
그래서 추가로 H2까지만 사용하는 전략 D도 실험했다.

---

## 6. RAGAS 점수 해석 회고

이번 4주차 실험에서는 3주차 baseline과 동일한 평가 질문 3개를 기준으로 chunking 전략별 RAGAS 점수를 비교했다.

평가에 사용한 지표는 다음 3개이다.

- Faithfulness
- Answer Relevancy
- Context Precision

Context Recall은 정답 context 라벨링이 필요하기 때문에 이번 baseline 단계에서는 제외했다.

---

## 6.1 3주차 baseline 대비 지표 변화

3주차 baseline 점수는 다음과 같다.

| Metric | Week 3 Baseline |
|---|---:|
| Faithfulness | 0.9259 |
| Answer Relevancy | 0.4284 |
| Context Precision | 1.0000 |

4주차 chunking 전략별 결과는 다음과 같다.

| Strategy | Faithfulness | Δ | Answer Relevancy | Δ | Context Precision | Δ |
|---|---:|---:|---:|---:|---:|---:|
| A_recursive_baseline_800_120 | 0.9667 | +0.0408 | 0.6564 | +0.2280 | 0.9625 | -0.0375 |
| B_recursive_large_1000_200 | 1.0000 | +0.0741 | 0.6669 | +0.2385 | 0.9347 | -0.0653 |
| C_markdown_header_recursive_800_120 | 0.9167 | -0.0092 | 0.4428 | +0.0144 | 0.9347 | -0.0653 |
| D_markdown_h2_recursive_1000_200 | 0.9697 | +0.0438 | 0.5431 | +0.1147 | 1.0000 | 0.0000 |

전체적으로 보면 4주차 실험에서는 대부분의 전략에서 Answer Relevancy가 3주차 baseline보다 상승했다.  
특히 전략 B는 Faithfulness와 Answer Relevancy가 모두 가장 높게 나왔다.

반면 Context Precision은 전략 D를 제외하면 3주차 baseline보다 하락했다.  
즉, 답변 생성 품질은 좋아졌지만, 검색된 chunk의 상위 정렬 품질은 일부 전략에서 약간 떨어졌다고 볼 수 있다.

---

## 6.2 점수 변화 원인 가설

Faithfulness가 상승한 이유는 chunk가 답변에 필요한 문맥을 더 잘 포함했기 때문으로 보인다.

3주차 baseline은 `chunk_size=800`, `chunk_overlap=120`을 사용했는데, 4주차 전략 B에서는 이를 `chunk_size=1000`, `chunk_overlap=200`으로 늘렸다.

게임 추천 도메인에서는 단일 문장보다 여러 문장을 종합해야 하는 경우가 많다.

예를 들어 다음과 같은 질문은 너무 작은 chunk보다 어느 정도 문맥을 포함한 chunk가 더 유리하다.

- 게임의 플레이 스타일
- 최근 리뷰의 전반적인 분위기
- 업데이트 내용 요약
- 장르, 태그, 특징 설명

전략 B는 chunk 크기를 조금 키워서 리뷰나 업데이트처럼 여러 문장을 함께 봐야 하는 질문에서 답변 근거를 더 잘 제공한 것으로 보인다.

Answer Relevancy가 상승한 이유도 비슷하다.  
전략 B는 질문에 필요한 주변 문맥을 더 많이 포함했기 때문에, LLM이 질문에 더 직접적으로 답할 수 있었다고 판단했다.

반대로 Context Precision이 일부 하락한 이유는 top-k 검색 결과 안에 관련 chunk와 약한 노이즈 chunk가 함께 포함되었기 때문으로 보인다.

특히 Steam News 섹션에는 게임 자체 업데이트뿐 아니라 관련 링크, 프로모션, 같은 IP의 다른 게임 뉴스가 섞일 수 있었다.  
예를 들어 Monster Hunter: World 문서에서는 Monster Hunter Stories 3 관련 뉴스가 함께 포함되어 있었고, No Man's Sky 문서에서는 `RELATED LINKS` 성격의 텍스트가 일부 포함되어 있었다.

이런 chunk가 검색 결과에 섞이면 Context Precision이 낮아질 수 있다.

---

## 6.3 가장 적합한 chunking 전략

최종적으로 본 도메인에 가장 적합한 chunking 전략은 전략 B라고 판단했다.

```text
B_recursive_large_1000_200
= 섹션 분리 + RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

전략 B를 선택한 이유는 다음과 같다.

- Faithfulness가 1.0000으로 가장 높았다.
- Answer Relevancy가 0.6669로 가장 높았다.
- Context Precision은 0.9347로 D보다 낮았지만, 충분히 높은 편이었다.
- 정성 평가에서도 답변이 가장 자연스러웠다.
- 리뷰나 업데이트처럼 여러 근거를 종합해야 하는 질문에서 문맥 보존이 좋았다.

전략 D는 Context Precision이 1.0000으로 가장 높았지만, Answer Relevancy가 전략 B보다 낮았다.  
즉, 관련 chunk를 상위에 잘 배치했지만, 최종 답변이 질문에 가장 직접적으로 맞지는 않았다고 해석했다.

전략 C는 Markdown 구조를 보존한다는 장점이 있었지만, H3 단위까지 분리되면서 리뷰와 뉴스가 지나치게 잘게 쪼개졌다.  
이 때문에 최근 리뷰의 전반적인 반응처럼 여러 리뷰를 종합해야 하는 질문에서는 오히려 불리했다.

따라서 현재 Steam 게임 추천 RAG에서는 Markdown H3까지 세밀하게 나누는 방식보다, section metadata를 유지하면서 RecursiveCharacterTextSplitter의 chunk size를 1000 정도로 설정하는 방식이 더 적합하다고 판단했다.

---

## 6.4 5주차 Retrieval 고도화에서 시도할 것

5주차에는 chunking 자체보다 retrieval 품질을 높이는 방향으로 개선할 계획이다.

### 1. Metadata Filtering Retriever

현재 chunk에는 `game_key`, `section`, `source_type`, `relevance_type` metadata가 들어가 있다.  
이를 활용해 질문 의도에 따라 검색 범위를 제한할 수 있다.

예시:

- 리뷰 질문 → `section=review`
- 업데이트 질문 → `section=news`
- 플레이 스타일 질문 → `section=about` 또는 `section=store_summary`
- 특정 게임 질문 → `game_key` filter 적용
- 업데이트/패치 질문 → `relevance_type=valid_update_or_patch` 우선 적용

이를 통해 다른 게임이나 다른 섹션의 chunk가 섞이는 문제를 줄일 수 있다.

### 2. Section-aware Reranking

현재는 top-k 안에 정답 chunk가 들어와도 항상 가장 위에 오지는 않는다.  
따라서 질문 의도에 따라 section 우선순위를 다르게 두는 reranking을 적용해보고 싶다.

예를 들어 gameplay 질문에서는 다음 순서로 우선순위를 둘 수 있다.

```text
about > store_summary > metadata > review > news
```

반대로 업데이트 질문에서는 다음 순서가 더 적합하다.

```text
news > metadata > review > about > store_summary
```

### 3. Hybrid Search

현재는 vector similarity search 중심으로 검색하고 있다.  
하지만 게임명, DLC명, 패치명, 태그명처럼 정확한 키워드가 중요한 경우에는 BM25 기반 keyword search도 필요하다.

예를 들어 `Monster Hunter: World`와 `Monster Hunter Stories 3`는 같은 IP라서 의미적으로 비슷하게 검색될 수 있다.  
Hybrid Search를 적용하면 정확한 게임명 매칭과 의미 기반 검색을 함께 사용할 수 있어 이런 문제를 줄일 수 있을 것으로 기대한다.

### 4. News 전처리 개선

Steam News 섹션에는 관련 링크나 프로모션 문구가 포함될 수 있다.  
따라서 다음 전처리를 추가하고 싶다.

- `RELATED LINKS:` 이후 텍스트 제거
- `Read the rest of the story...` 제거
- 이미지 placeholder 제거
- 프로모션성 뉴스와 실제 업데이트 뉴스 구분
- `relevance_type` metadata를 활용한 우선순위 조정

### 5. Time-aware Retrieval

프로젝트의 핵심 방향은 시간에 따라 변하는 게임 평가를 반영하는 것이다.  
따라서 다음 단계에서는 리뷰 작성일과 업데이트 날짜를 metadata로 활용하고 싶다.

예시:

- 최근 30일 리뷰 우선 검색
- 업데이트 이후 작성된 리뷰만 검색
- 패치 전후 유저 반응 비교
- 오래된 뉴스나 공략의 우선순위 하향

---

## 6.5 요약

4주차 실험 결과, 전략 B가 가장 안정적인 chunking 전략으로 판단되었다.

전략 B는 3주차 baseline 대비 Faithfulness와 Answer Relevancy를 개선했다.  
다만 Context Precision은 일부 하락했는데, 이는 News 섹션의 노이즈와 section filtering 부족이 원인으로 보인다.

따라서 5주차에는 chunking보다는 metadata filtering, section-aware reranking, hybrid search를 통해 retrieval 품질을 개선할 계획이다.

---

## 7. 최종 정리

이번 4주차 과제에서는 Steam 게임 Markdown 문서의 데이터 품질을 진단하고, chunking 전략별 RAGAS 점수를 비교했다.

현재 데이터는 수집 자체가 크게 잘못된 것은 아니지만, News 섹션 안에 업데이트, 판매 순위, 프로모션, 관련 링크 등 서로 다른 성격의 정보가 함께 들어가 있었다.

실험 결과, `B_recursive_large_1000_200` 전략이 가장 안정적이었다.  
이 전략은 3주차 baseline 대비 Faithfulness와 Answer Relevancy를 가장 크게 개선했다.

다음 주차에는 metadata filtering과 reranking을 활용해 질문 의도에 맞는 section을 더 정확히 검색하도록 개선할 예정이다.