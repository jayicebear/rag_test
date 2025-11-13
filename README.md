# ChromaDB 기반 RAG 벤치마크

**실제 논문·뉴스·리포트·단편소설등의 문서**를 `Docling → Structured Chunking` 후  
**12,480개 고품질 QA 테스트셋** 생성 → **ChromaDB**에서 실험 진행

> **한국어 RAG 시스템 구축 전 돌려봐야 할 벤치마크**  
> **임베딩 모델 + 검색 전략 + Reranking 유무** 비교

## 테스트셋 구성 (Structured Chunking)
## Embedding model </br>
`OpenAI-text-embedding-ada-002,OpenAI-text-embedding-3-large, Qwen3-embedding-0.6B, Qwen3-embedding-4B, all-MiniLM-L6-V2, google/embeddinggemma-0.3B`
## Reranking model </br>
`Qwen3-reranking-0.6B`
## Query rewriting methods </br>
`query rewriting, query expansion, multi query`
## Query rewriting methods </br>
```mermaid
graph TD
    A[4종 실전 문서\n- 논문: Seoul_Dasan.pdf\n- 논문: REMDoC.pdf\n- 뉴스 50건\n- 단편소설 30건\n- 기업 리포트 20건] --> B[Docling\nPDF → Markdown]
    B --> C[Structured Chunking\nHeader + Table + List\n→ 400~600토큰 청크]
    C --> D[GPT-4o\n청크당 3개 질문 생성]
    D --> E[총 12,480개 QA 페어\n{question, chunk, doc_type, page}]
    E --> F[ChromaDB\npersistent collection]
