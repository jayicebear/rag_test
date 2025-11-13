# ChromaDB 기반 RAG 벤치마크

**실제 논문·뉴스·리포트·단편소설 등의 문서**를 `Docling → Structured Chunking` 후  
**12,480개 고품질 QA 테스트셋** 생성 → **ChromaDB**에서 실험 진행

> **한국어 RAG 시스템 구축 전 돌려봐야 할 벤치마크**  
> **임베딩 모델 + 검색 전략 + Reranking 유무** 비교

## 테스트셋 구성 (Structured Chunking)
## Embedding model </br>
`OpenAI-text-embedding-ada-002,OpenAI-text-embedding-3-large, Qwen3-embedding-0.6B, Qwen3-embedding-4B, all-MiniLM-L6-V2, google/embeddinggemma-0.3B`
## Reranking model </br>
`Qwen3-reranking-0.6B`
## Query rewriting methods </br>
`Query rewriting, Query expansion, Multi query, Agentic search(majority vote, ensemble)`
