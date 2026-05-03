# Week 3 Baseline RAG

## 1. 프로젝트 개요

Steam 게임 추천 AI 프로젝트의 3주차 baseline으로, Steam 게임 문서를 기반으로 질문에 답변하는 Naive RAG를 구현했다.

이번 주차 목표는 완성형 서비스를 만드는 것이 아니라, 다음 RAG 기본 흐름을 재현 가능한 형태로 만드는 것이다.

```text
문서 수집 → 문서 로딩 → Chunk 분할 → Embedding → Vector DB → Retriever → LLM 답변 생성
```

---

## 2. 사용 데이터

이번 baseline에서는 Steam 게임 5개를 대상으로 데이터를 수집했다.

* Hollow Knight
* Monster Hunter: World
* Baldur's Gate 3
* No Man's Sky
* Cyberpunk 2077

각 게임별로 다음 정보를 Markdown 문서로 저장했다.

* Steam Store 설명
* 최근 Steam 리뷰
* Steam News / Update 정보

문서 위치:

```text
data/docs/
```

---

## 3. 폴더 구조

```text
rag-agent/
├── data/
│   ├── docs/        # RAG 입력 문서
│   ├── eval/        # 평가 결과 CSV
│   ├── raw/         # 원본 수집 데이터, GitHub 제외
│   └── chroma_db/   # Chroma DB, GitHub 제외
├── docs/
│   └── week3_baseline_rag.md
├── notebooks/
│   └── week3_baseline_rag.ipynb
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 4. 문서 로딩 및 Chunk 분할

Markdown 문서는 LangChain의 `DirectoryLoader`와 `TextLoader`로 불러왔다.

Chunk 분할에는 `RecursiveCharacterTextSplitter`를 사용했다.

설정값:

| 항목            |   값 |
| ------------- | --: |
| 원본 문서 수       |   5 |
| chunk_size    | 800 |
| chunk_overlap | 120 |
| 생성 chunk 수    | 119 |

게임별 chunk 수:

| 파일                        | chunk 수 |
| ------------------------- | ------: |
| `baldurs_gate_3.md`       |      26 |
| `cyberpunk_2077.md`       |      16 |
| `hollow_knight.md`        |      23 |
| `monster_hunter_world.md` |      28 |
| `no_mans_sky.md`          |      26 |

---

## 5. Embedding / Vector DB / Retriever

사용한 설정은 다음과 같다.

| 항목        | 사용한 기술                                                        |
| --------- | ------------------------------------------------------------- |
| Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | Chroma                                                        |
| Retriever | similarity search                                             |
| 검색 개수     | `k=4`                                                         |
| LLM       | `gpt-5-mini`                                                  |

---

## 6. 검색 개선 내용

초기에는 단순 similarity search만 사용했다. 이 경우 리뷰 질문에서 다른 게임의 review chunk가 함께 검색되는 문제가 있었다.

예를 들어 다음 질문에서 문제가 발생했다.

```text
Cyberpunk 2077 최근 리뷰에서는 어떤 반응이 있나요?
```

초기 검색 결과에는 Cyberpunk 2077뿐 아니라 다른 게임의 review chunk도 섞였다.

이를 해결하기 위해 chunk metadata에 다음 정보를 추가했다.

* `game_key`: 어떤 게임 문서인지 구분
* `section`: 문서 내 영역 구분

사용한 section 값:

* `metadata`
* `store_summary`
* `about`
* `review`
* `news`

이후 질문에서 게임명과 의도를 감지해 Chroma metadata filter를 적용했다.

예시:

```python
{
    "$and": [
        {"section": {"$eq": "review"}},
        {"game_key": {"$eq": "cyberpunk_2077"}}
    ]
}
```

이를 통해 리뷰 질문은 해당 게임의 `review` chunk만, 업데이트 질문은 해당 게임의 `news` chunk만 검색하도록 개선했다.

---

## 7. 평가 질문

평가 질문은 총 10개로 구성했다.

1. Hollow Knight는 어떤 플레이 스타일의 게임인가요?
2. Hollow Knight의 분위기나 월드 구성은 어떤 특징이 있나요?
3. Monster Hunter: World의 핵심 플레이 루프는 무엇인가요?
4. Monster Hunter: World는 협동 플레이 측면에서 어떤 특징이 있나요?
5. Baldur's Gate 3는 어떤 RPG인가요?
6. Baldur's Gate 3에서 선택과 서사는 어떤 역할을 하나요?
7. No Man's Sky는 업데이트와 관련해서 어떤 내용이 있나요?
8. No Man's Sky의 최근 뉴스나 업데이트 방향은 무엇인가요?
9. Cyberpunk 2077 최근 리뷰에서는 어떤 반응이 있나요?
10. Cyberpunk 2077 문서에서 확인되는 주요 특징은 무엇인가요?

평가 결과 저장 위치:

```text
data/eval/week3_baseline_results.csv
```

---

## 8. 현재 Baseline의 아쉬운 점

* 수집 게임 수가 5개로 적다.
* 게임당 최근 리뷰 20개만 사용해 리뷰 대표성이 부족하다.
* Steam News에 대상 게임 외의 관련 뉴스가 섞일 수 있다.
* 아직 BM25 기반 keyword search를 적용하지 않았다.
* 아직 reranker를 적용하지 않았다.
* RAGAS 평가는 아직 완전히 적용하지 않았다.

---

## 9. 다음 개선 방향

다음 주차에서는 다음을 개선할 계획이다.

* 수집 게임 수 확대
* 리뷰 수 확대
* 최근 리뷰와 전체 리뷰 분리
* 업데이트 이후 리뷰만 검색하는 time-aware retrieval
* BM25 + Vector Search 기반 Hybrid Search
* Reranker 적용
* RAGAS 기반 평가 지표 측정

---

## 10. 요약

이번 3주차에서는 Steam 게임 5개에 대해 Store 설명, 최근 리뷰, 뉴스/업데이트 정보를 수집하고 Markdown 문서로 변환했다. 이후 LangChain Loader로 문서를 불러오고, `chunk_size=800`, `chunk_overlap=120` 기준으로 분할해 총 119개 chunk를 생성했다.

Embedding은 multilingual sentence-transformers 모델을 사용했고, Vector DB는 Chroma를 사용했다. 초기 검색에서는 리뷰 질문에 다른 게임 리뷰가 섞이는 문제가 있었지만, `game_key`와 `section` metadata를 추가해 검색 품질을 개선했다.

## RAGAS 평가 결과

가능한 범위에서 RAGAS 평가를 수행했다.  
평가 대상은 전체 10개 질문 중 3개 질문으로 제한했다.

| Metric | Score |
|---|---:|
| Faithfulness | 0.9259 |
| Answer Relevancy | 0.4284 |
| Context Precision | 1.0000 |
| Context Recall | 1.0000 |

### 해석

- Faithfulness는 0.9259로 높게 측정되었다. 답변이 검색된 context에 비교적 잘 근거하고 있음을 의미한다.
- Context Precision과 Context Recall은 모두 1.0000으로 측정되었다. 평가에 사용한 3개 질문에서는 검색된 chunk가 reference 답변에 필요한 정보를 잘 포함하고 있었다.
- Answer Relevancy는 0.4284로 낮게 측정되었다. 답변이 질문에 직접적으로 답하기보다 부가 설명이 섞였거나, 질문 대비 답변 초점이 흐려졌을 가능성이 있다.
- 다음 개선에서는 prompt를 더 간결하게 조정하고, 질문에 직접 답하는 형식으로 답변을 제한할 계획이다.
