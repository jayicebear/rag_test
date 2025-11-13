import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from reranker import QwenReranker
# -------------------------
# 데이터 수집
def collect_data(dataset):
    all_contexts = []
    context_to_docid = {}
    for i, item in enumerate(dataset):
        for ctx in item["reference_contexts"]:
            doc_id = len(all_contexts)
            all_contexts.append(
                Document(page_content=ctx, metadata={"query_idx": i, "doc_id": doc_id})
            )
            context_to_docid[doc_id] = i
    tokenized_corpus = [doc.page_content.split() for doc in all_contexts]
    return tokenized_corpus, all_contexts, context_to_docid

# -------------------------
# Chroma 시작 (LangChain 래퍼)
def chroma_start(all_contexts, embeddings, db_path):
    vector_store = Chroma.from_documents(
        documents=all_contexts,           # Document 리스트
        embedding=embeddings,             # ✅ LangChain Embeddings 객체여야 함
        persist_directory=db_path
    )
    # vector_store.persist()  # 원하시면 디스크에 강제 flush
    return vector_store

# -------------------------
# BM25 평가
def evaluate_bm25(dataset, tokenized_corpus, context_to_docid, ks=[1,3,5]):
    bm25 = BM25Okapi(tokenized_corpus)
    recalls = {k: [] for k in ks}
    mrrs = []

    for q_idx, item in enumerate(dataset):
        query = item["user_input"]
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)           # 길이 = 문서 수
        ranked = np.argsort(scores)[::-1]                   # doc_id 내림차순
        gold_ids = [doc_id for doc_id, qi in context_to_docid.items() if qi == q_idx]

        for k in ks:
            top_k_ids = ranked[:k]
            hit = any(doc_id in top_k_ids for doc_id in gold_ids)
            recalls[k].append(1 if hit else 0)

        rr = 0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in gold_ids:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    results = {f"Recall@{k}": round(float(np.mean(recalls[k])),3) for k in ks}
    results["MRR"] = round(float(np.mean(mrrs)),3)
    return results

# -------------------------
# Hybrid 평가 (BM25 + Chroma similarity)
def evaluate_hybrid(dataset, tokenized_corpus, context_to_docid, vector_store, ks=[1,3,5], alpha=0.7):
    """
    alpha: 1.0이면 BM25만, 0.0이면 Chroma만. 보통 0.3~0.7 범위에서 튜닝.
    """
    bm25 = BM25Okapi(tokenized_corpus)
    recalls = {k: [] for k in ks}
    mrrs = []
    n_docs = len(tokenized_corpus)

    for q_idx, item in enumerate(dataset):
        query = item["user_input"]
        tokenized_query = query.split()

        # 1) BM25 점수 (0~1 정규화)
        bm25_scores = np.array(bm25.get_scores(tokenized_query))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # 2) Chroma 점수 (cosine distance -> similarity로 변환, 0~1 정규화)
        chroma_results = vector_store.similarity_search_with_score(query, k=n_docs)
        # chroma_results: List[(Document, distance)]
        doc_id_to_chroma = {}
        raw_sims = []
        for doc, dist in chroma_results:
            sim = 1.0 - dist  # distance→similarity
            doc_id_to_chroma[doc.metadata["doc_id"]] = sim
            raw_sims.append(sim)

        # 정규화 준비 (전 문서 대상으로)
        if raw_sims:
            max_sim = max(raw_sims)
            if max_sim <= 0:
                max_sim = 1.0
        else:
            max_sim = 1.0
        chroma_scores = np.zeros(n_docs, dtype=float)
        for i in range(n_docs):
            sim = doc_id_to_chroma.get(i, 0.0)
            chroma_scores[i] = sim / max_sim

        # 3) 가중 결합
        final_scores = alpha * bm25_scores + (1 - alpha) * chroma_scores

        # 4) 순위/지표
        ranked = np.argsort(final_scores)[::-1]
        gold_ids = [doc_id for doc_id, qi in context_to_docid.items() if qi == q_idx]

        for k in ks:
            top_k_ids = ranked[:k]
            hit = any(doc_id in top_k_ids for doc_id in gold_ids)
            recalls[k].append(1 if hit else 0)

        rr = 0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in gold_ids:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    results = {f"Recall@{k}": round(float(np.mean(recalls[k])),3) for k in ks}
    results["MRR"] = round(float(np.mean(mrrs)), 3)
    return results

def evaluate_hybrid_reranker(
    dataset, 
    tokenized_corpus, 
    context_to_docid, 
    vector_store,
    all_contexts,  # 전체 문서 필요
    ks=[1,3,5], 
    alpha=0.5,
    top_k_candidates=20
):
    """
    Hybrid 결과를 Reranker로 재정렬
    """
    # Reranker 초기화
    print("Loading Reranker model...")
    reranker = QwenReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
    if reranker.tokenizer.pad_token is None:
        reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
    bm25 = BM25Okapi(tokenized_corpus)
    recalls = {k: [] for k in ks}
    mrrs = []
    n_docs = len(tokenized_corpus)

    for q_idx, item in enumerate(dataset):
        query = item["user_input"]
        tokenized_query = query.split()

        # 1) BM25 점수
        bm25_scores = np.array(bm25.get_scores(tokenized_query))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # 2) Chroma 점수
        chroma_results = vector_store.similarity_search_with_score(query, k=n_docs)
        doc_id_to_chroma = {}
        raw_sims = []
        for doc, dist in chroma_results:
            sim = 1.0 - dist
            doc_id_to_chroma[doc.metadata["doc_id"]] = sim
            raw_sims.append(sim)

        if raw_sims:
            max_sim = max(raw_sims)
            if max_sim <= 0:
                max_sim = 1.0
        else:
            max_sim = 1.0
        
        chroma_scores = np.zeros(n_docs, dtype=float)
        for i in range(n_docs):
            sim = doc_id_to_chroma.get(i, 0.0)
            chroma_scores[i] = sim / max_sim

        # 3) Hybrid 점수로 Top-K 후보 선택
        hybrid_scores = alpha * bm25_scores + (1 - alpha) * chroma_scores
        top_k_indices = np.argsort(hybrid_scores)[::-1][:top_k_candidates]
        
        # 4) Reranking
        candidate_docs = [all_contexts[idx].page_content for idx in top_k_indices]
        rerank_scores = reranker.rank(query, candidate_docs)
        
        # 5) 최종 순위
        reranked_indices = np.argsort(rerank_scores)[::-1]
        final_ranked = [top_k_indices[i] for i in reranked_indices]
        
        # 6) 평가
        gold_ids = [doc_id for doc_id, qi in context_to_docid.items() if qi == q_idx]

        for k in ks:
            top_k_ids = final_ranked[:k]
            hit = any(doc_id in top_k_ids for doc_id in gold_ids)
            recalls[k].append(1 if hit else 0)

        rr = 0
        for rank, doc_id in enumerate(final_ranked, start=1):
            if doc_id in gold_ids:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    results = {f"Recall@{k}": round(float(np.mean(recalls[k])),3) for k in ks}
    results["MRR"] = round(float(np.mean(mrrs)), 3)
    return results



# -------------------------
if __name__ == "__main__":
    import json
    import os
    file_name = 'safe_sql'
    with open(f"./Test_data/{file_name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    os.environ['OPENAI_API_KEY'] = ""

    # 수집
    tokenized_corpus, all_contexts, context_to_docid = collect_data(data)

    # Embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

    # Chroma 만들기
    db_path = "./bm25_chroma_db"
    vector_store = chroma_start(all_contexts, embeddings, db_path)

    # BM25
    print("\n=== BM25 Evaluation ===")
    bm25_results = evaluate_bm25(data, tokenized_corpus, context_to_docid, ks=[1,3,5])
    print("BM25 Results:", bm25_results)

    # Hybrid
    print("\n=== Hybrid Evaluation ===")
    hybrid_results = evaluate_hybrid(
        data, tokenized_corpus, context_to_docid, vector_store,
        ks=[1,3,5], alpha=0.5
    )
    print("Hybrid Results:", hybrid_results)
    
    # Hybrid with reranker (새로 추가!)
    print("\n=== Hybrid + Reranker Evaluation ===")
    hybrid_with_reranker = evaluate_hybrid_reranker(
        data, 
        tokenized_corpus, 
        context_to_docid, 
        vector_store,
        all_contexts,  # 추가 파라미터
        ks=[1,3,5], 
        alpha=0.5,
        top_k_candidates=20
    )
    print("Hybrid_with_reranker_results:", hybrid_with_reranker)
    
    # 결과 비교
    print("\n=== Comparison ===")
    print(f"BM25:                {bm25_results}")
    print(f"Hybrid:              {hybrid_results}")
    print(f"Hybrid + Reranker:   {hybrid_with_reranker}")
