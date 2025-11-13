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
# Chroma 시작 (LangChain)
def chroma_start(all_contexts, embeddings, db_path):
    vector_store = Chroma.from_documents(
        documents=all_contexts,
        embedding=embeddings,
        persist_directory=db_path
    )
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
        scores = bm25.get_scores(tokenized_query)
        ranked = np.argsort(scores)[::-1]
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
# BM25 + Reranker 평가 (새로 추가!)
def evaluate_bm25_reranker(
    dataset, 
    tokenized_corpus, 
    context_to_docid,
    all_contexts,
    reranker,
    ks=[1,3,5],
    top_k_candidates=20
):
    """
    BM25 결과를 Reranker로 재정렬
    
    Args:
        dataset: 평가 데이터셋
        tokenized_corpus: 토큰화된 문서 리스트
        context_to_docid: doc_id -> query_idx 매핑
        all_contexts: 전체 Document 리스트
        reranker: QwenReranker 인스턴스
        ks: Recall@k 계산할 k 값들
        top_k_candidates: Reranking 전에 먼저 선택할 후보 수
    """
    bm25 = BM25Okapi(tokenized_corpus)
    recalls = {k: [] for k in ks}
    mrrs = []

    for q_idx, item in enumerate(dataset):
        query = item["user_input"]
        tokenized_query = query.split()

        # 1) BM25 점수로 Top-K 후보 선택
        bm25_scores = bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(bm25_scores)[::-1][:top_k_candidates]
        
        # 2) Reranking: Top-K 후보들만 reranker로 재평가
        candidate_docs = [all_contexts[idx].page_content for idx in top_k_indices]
        rerank_scores = reranker.rank(query, candidate_docs)
        
        # 3) Reranker 점수로 최종 순위 결정
        reranked_indices = np.argsort(rerank_scores)[::-1]
        final_ranked = [top_k_indices[i] for i in reranked_indices]
        
        # 4) 평가 지표 계산
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

# -------------------------
# Hybrid + Reranker 평가
def evaluate_hybrid_reranker(
    dataset, 
    tokenized_corpus, 
    context_to_docid, 
    vector_store,
    all_contexts,
    reranker,
    ks=[1,3,5], 
    alpha=0.5,
    top_k_candidates=20
):
    """
    Hybrid 결과를 Reranker로 재정렬
    """
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

    file_name = 'safe_sql'
    with open(f"./Test_data/{file_name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

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

    # Reranker 초기화 (한 번만!)
    print("\n=== Loading Reranker Model ===")
    reranker = QwenReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
    print("Reranker loaded successfully!")

    # BM25 + Reranker (새로 추가!)
    print("\n=== BM25 + Reranker Evaluation ===")
    bm25_reranker_results = evaluate_bm25_reranker(
        data,
        tokenized_corpus,
        context_to_docid,
        all_contexts,
        reranker,
        ks=[1,3,5],
        top_k_candidates=20
    )
    print("BM25 + Reranker Results:", bm25_reranker_results)

    # Hybrid
    print("\n=== Hybrid Evaluation ===")
    hybrid_results = evaluate_hybrid(
        data, tokenized_corpus, context_to_docid, vector_store,
        ks=[1,3,5], alpha=0.5
    )
    print("Hybrid Results:", hybrid_results)

    # Hybrid + Reranker
    print("\n=== Hybrid + Reranker Evaluation ===")
    hybrid_reranker_results = evaluate_hybrid_reranker(
        data, 
        tokenized_corpus, 
        context_to_docid, 
        vector_store,
        all_contexts,
        reranker,
        ks=[1,3,5], 
        alpha=0.5,
        top_k_candidates=20
    )
    print("Hybrid + Reranker Results:", hybrid_reranker_results)

    # 최종 결과 비교
    print("\n" + "="*70)
    print("=== Final Results Comparison ===")
    print("="*70)
    print(f"{'Method':<25} {'Recall@1':<12} {'Recall@3':<12} {'Recall@5':<12} {'MRR':<10}")
    print("-"*70)
    print(f"{'BM25':<25} {bm25_results['Recall@1']:<12} {bm25_results['Recall@3']:<12} {bm25_results['Recall@5']:<12} {bm25_results['MRR']:<10}")
    print(f"{'BM25 + Reranker':<25} {bm25_reranker_results['Recall@1']:<12} {bm25_reranker_results['Recall@3']:<12} {bm25_reranker_results['Recall@5']:<12} {bm25_reranker_results['MRR']:<10}")
    print(f"{'Hybrid':<25} {hybrid_results['Recall@1']:<12} {hybrid_results['Recall@3']:<12} {hybrid_results['Recall@5']:<12} {hybrid_results['MRR']:<10}")
    print(f"{'Hybrid + Reranker':<25} {hybrid_reranker_results['Recall@1']:<12} {hybrid_reranker_results['Recall@3']:<12} {hybrid_reranker_results['Recall@5']:<12} {hybrid_reranker_results['MRR']:<10}")
    print("="*70)