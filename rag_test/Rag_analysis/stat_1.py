import json
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ChromaDB 및 임베딩 관련
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['OPENAI_API_KEY'] = ""

# Recall @ 1: 상위 1개 search result 중 정답이 몇개인가?
#검색된 문서: [문서1]
#정답 문서: [문서3, 문서7]
#교집합: 없음 (문서1은 정답이 아님)
#Recall@1 = 0/2 = 0.0
# Recall @ 3: 상위 3개 search result 중 정답이 몇개인가?
# Recall @ 5: 상위 5개 search result 중 정답이 몇개인가?
# MRR(Mean Reciprocal Rank): 첫번째 정답이 몇 번째 순위에 나타나는가? 

# 기존 코드
def load_data():
    file_name = 'safe_sql'
    with open(f"./Test_data/{file_name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    user_queries = []
    reference_contexts = []
    reference_answers = []

    for item in data:
        user_queries.append(item['user_input'])
        reference_contexts.append(item['reference_contexts'])
        reference_answers.append(item['reference'])
    print(f"총 {len(user_queries)}개의 테스트 데이터 로드")
    return file_name, user_queries, reference_contexts, reference_answers
import shutil

from sentence_transformers import SentenceTransformer

class RetrievalEvaluator:
    def __init__(self, reference_contexts: List[List[str]]):
        """
        reference_contexts: 각 질문별 정답 컨텍스트들
        """
        #self.embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        self.embeddings = SentenceTransformerEmbeddings("Qwen/Qwen3-Embedding-0.6B")
        # 모든 reference contexts를 하나의 문서 컬렉션으로 만들기
        all_contexts = []
        self.context_to_query_map = {}  # 어떤 컨텍스트가 어떤 쿼리의 정답인지 매핑
        
        for query_idx, contexts in enumerate(reference_contexts):
            for context in contexts:
                doc_id = len(all_contexts)
                all_contexts.append(Document(
                    page_content=context,
                    metadata={"query_idx": query_idx, "doc_id": doc_id}
                ))
                self.context_to_query_map[doc_id] = query_idx
        
        # ChromaDB 생성
        db_path = "./test_chroma_db"
        
        if os.path.exists(db_path) and os.listdir(db_path):
            print("기존 ChromaDB를 로드합니다.")
            try:
                self.vector_store = Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"기존 DB 로드 실패: {e}. 새로 생성합니다.")
                self._create_new_db(db_path, all_contexts)
        else:
            print("새로운 ChromaDB를 생성합니다.")
            self._create_new_db(db_path, all_contexts)
        
        self.all_contexts = all_contexts
        print(f"총 {len(all_contexts)}개의 컨텍스트 문서를 처리했습니다.")
    
    def _create_new_db(self, db_path, all_contexts):
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        self.vector_store = Chroma.from_documents(
            documents=all_contexts,
            embedding=self.embeddings,
            persist_directory=db_path
        )
    
    def evaluate_retrieval_method(self, queries: List[str], method_name: str, 
                                 retrieval_func, top_k: int = 5) -> Dict:
        """
        검색 방법 평가
        """
        results = {
            "method": method_name,
            "recall_at_1": [],
            "recall_at_3": [],
            "recall_at_5": [],
            "mrr": [],  # Mean Reciprocal Rank
            "detailed_results": []
        }
        
        for query_idx, query in enumerate(queries):
            # 검색 수행
            retrieved_docs = retrieval_func(query, top_k)
            
            # 정답 문서 ID들 찾기
            correct_doc_ids = [
                doc.metadata["doc_id"] for doc in self.all_contexts 
                if doc.metadata["query_idx"] == query_idx
            ]
            
            # 검색된 문서 ID들
            retrieved_doc_ids = [
                doc.metadata["doc_id"] for doc in retrieved_docs
            ]
            
            # Recall 계산
            recall_1 = self._calculate_recall(correct_doc_ids, retrieved_doc_ids[:1])
            recall_3 = self._calculate_recall(correct_doc_ids, retrieved_doc_ids[:3])
            recall_5 = self._calculate_recall(correct_doc_ids, retrieved_doc_ids[:5])
            
            # MRR 계산
            mrr = self._calculate_mrr(correct_doc_ids, retrieved_doc_ids)
            
            results["recall_at_1"].append(recall_1)
            results["recall_at_3"].append(recall_3)
            results["recall_at_5"].append(recall_5)
            results["mrr"].append(mrr)
            
            results["detailed_results"].append({
                "query": query,
                "correct_docs": len(correct_doc_ids),
                "recall_1": recall_1,
                "recall_3": recall_3,
                "recall_5": recall_5,
                "mrr": mrr
            })
        
        # 평균 계산
        results["avg_recall_at_1"] = np.mean(results["recall_at_1"])
        results["avg_recall_at_3"] = np.mean(results["recall_at_3"])
        results["avg_recall_at_5"] = np.mean(results["recall_at_5"])
        results["avg_mrr"] = np.mean(results["mrr"])
        
        return results
    
    def _calculate_recall(self, correct_ids: List[int], retrieved_ids: List[int]) -> float:
        """Recall@K 계산"""
        if not correct_ids:
            return 0.0
        
        correct_retrieved = len(set(correct_ids) & set(retrieved_ids))
        return correct_retrieved / len(correct_ids)
    
    def _calculate_mrr(self, correct_ids: List[int], retrieved_ids: List[int]) -> float:
        """Mean Reciprocal Rank 계산"""
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in correct_ids:
                return 1.0 / rank
        return 0.0
    
# 평가자 초기화



# 다양한 검색 방법들 정의
def vector_search(query: str, top_k: int = 5):
    """기본 벡터 유사도 검색"""
    return evaluator.vector_store.similarity_search(query, k=top_k)

def mmr_search(query: str, top_k: int = 5):
    """MMR(Maximal Marginal Relevance) 검색"""
    return evaluator.vector_store.max_marginal_relevance_search(query, k=top_k)

# LLM 추가 (쿼리 재작성용)
query_rewriter = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def query_rewriting(query: str, top_k: int = 5):
    """쿼리 재작성 후 검색"""
    
    # 1. 쿼리 재작성
    rewrite_prompt = f"""
Please rewrite the following question into a more specific and search-friendly form.  
Keep the original intent of the question, but make the keywords clearer and add terms that will help with searching.

Original question: {query}

Rewritten question:"""
    
    try:
        rewritten_query = query_rewriter.invoke(rewrite_prompt).content.strip()
        print(f"원본: {query}")
        print(f"재작성: {rewritten_query}")
    except:
        # LLM 호출 실패시 원본 쿼리 사용
        rewritten_query = query
    
    # 2. 재작성된 쿼리로 검색
    #return evaluator.vector_store.similarity_search(rewritten_query, k=top_k)
    return evaluator.vector_store.max_marginal_relevance_search(rewritten_query, k=top_k)
def query_expansion(query: str, top_k: int = 5):
    """쿼리 확장 후 검색"""
    
    # 쿼리 확장 (동의어, 관련 용어 추가)
    expansion_prompt = f"""
Please expand the following question by adding synonyms, similar expressions, and related keywords.  
Expand it into 1 query only.  

Original question: {query}

Expanded search query (original + related keywords):"""
    
    try:
        expanded_query = query_rewriter.invoke(expansion_prompt).content.strip()
        print(f"원본: {query}")
        print(f"확장: {expanded_query}")
    except:
        expanded_query = query
    
    #return evaluator.vector_store.similarity_search(expanded_query, k=top_k)
    return evaluator.vector_store.max_marginal_relevance_search(expanded_query, k=top_k)

def multi_query_search(query: str, top_k: int = 5):
    """다중 쿼리 생성 후 검색"""
    
    # 여러 관점의 쿼리 생성
    multi_query_prompt = f"""
Please generate 3 different search queries for the following question from different perspectives.  
Each query should aim to find the same information as the original question, but be expressed in a different way.  

Original question: {query}

Query 1:  
Query 2:  
Query 3:"""
    
    try:
        multi_queries = query_rewriter.invoke(multi_query_prompt).content.strip()
        # 파싱 (간단하게)
        queries = [line.split(":", 1)[1].strip() for line in multi_queries.split("\n") if ":" in line]
        
        if len(queries) < 3:
            queries = [query]  # 파싱 실패시 원본 사용
            
    except:
        queries = [query]
    
    # 각 쿼리로 검색하여 결과 합치기
    all_results = []
    for q in queries[:3]:  # 최대 3개 쿼리만 사용
        #results = evaluator.vector_store.similarity_search(q, k=top_k//len(queries) + 1)
        results = evaluator.vector_store.max_marginal_relevance_search(q, k=top_k//len(queries) + 1)

        all_results.extend(results)
    
    # 중복 제거 및 상위 k개 반환
    seen_ids = set()
    unique_results = []
    for doc in all_results:
        doc_id = doc.metadata.get("doc_id", str(doc))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_results.append(doc)
        if len(unique_results) >= top_k:
            break
    
    return unique_results[:top_k]

# 하이브리드 검색 (기존 + query rewriting )
# def hybrid_rewrite_search(query: str, top_k: int = 5, alpha: float = 0.7):
#     """원본 쿼리와 재작성 쿼리 결과를 가중 평균"""
    
#     # 원본 쿼리 검색
#     original_results = evaluator.vector_store.similarity_search_with_score(query, k=top_k)
    
#     # 재작성 쿼리 검색
#     rewrite_prompt = f'''Please rewrite the following question into a more specific and search-friendly form.  
#                         Keep the original intent of the question, but make the keywords clearer and add terms that will help with searching.
#                         User question:{query} 
#                         Rewrite:'''
#     try:
#         rewritten_query = query_rewriter.invoke(rewrite_prompt).content.strip()
#         rewritten_results = evaluator.vector_store.similarity_search_with_score(rewritten_query, k=top_k)
#     except:
#         rewritten_results = original_results
    
#     # 점수 조합
#     combined_scores = {}
    
#     # 원본 결과 (가중치 alpha)
#     for doc, score in original_results:
#         doc_id = doc.metadata["doc_id"]
#         combined_scores[doc_id] = combined_scores.get(doc_id, 0) + alpha * score
    
#     # 재작성 결과 (가중치 1-alpha)
#     for doc, score in rewritten_results:
#         doc_id = doc.metadata["doc_id"]
#         combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1-alpha) * score
    
#     # 점수 순으로 정렬
#     sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
#     # 문서 객체 찾아서 반환
#     result_docs = []
#     for doc_id, score in sorted_docs[:top_k]:
#         for doc in evaluator.all_contexts:
#             if doc.metadata["doc_id"] == doc_id:
#                 result_docs.append(doc)
#                 break
    
#     return result_docs



import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SentenceTransformerEmbeddings():
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()
    
    def encode(self, text: str) -> List[float]:
        return self.embed_query(text)
    
def agentic_search(query: str, top_k: int = 5):
    """
    코사인 유사도 기반 Agentic Search
    - 직접 임베딩 계산으로 투명한 점수 산출
    - 6가지 검색 방법 조합
    """
    
    # 모든 문서 가져오기
    try:
        all_docs = evaluator.vector_store._collection.get(include=['documents', 'metadatas', 'embeddings'])
        
        # None 체크를 명시적으로 처리
        docs_list = all_docs['documents']
        metas_list = all_docs['metadatas'] if all_docs['metadatas'] is not None else []
        embeddings_list = all_docs['embeddings'] if all_docs['embeddings'] is not None else []

        # 문서들을 Document 객체로 변환
        documents = []
        for i, (content, metadata) in enumerate(zip(docs_list, metas_list)):
            doc = type('Document', (), {
                'page_content': content,
                'metadata': metadata if metadata else {'doc_id': str(i)}
            })()
            documents.append(doc)
        
        doc_embeddings = embeddings_list
        print('문서로딩 성공')
        
    except Exception as e:
        print(f"문서 로딩 실패, 기존 방식 사용: {e}")
        # 기존 방식으로 폴백
        temp_docs = evaluator.vector_store.similarity_search(" ", k=1000)  # 모든 문서 가져오기
        documents = temp_docs
        doc_embeddings = []
        for doc in documents:
            doc_embedding = evaluator.embeddings.encode(doc.page_content)
            doc_embeddings.append(doc_embedding)
        
    # 쿼리 임베딩 생성
    query_embedding = evaluator.embeddings.encode(query)
    
    def get_top_k_similar(query_emb, docs, doc_embs, k):
        """코사인 유사도로 상위 k개 문서 반환"""
        similarities = cosine_similarity([query_emb], doc_embs)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        return [(docs[i], similarities[i]) for i in top_indices]
    
    # 1. 기본 벡터 유사도 검색 - 점수 포함으로 변경
    vector_results = get_top_k_similar(query_embedding, documents, doc_embeddings, top_k)
    
    # 2. MMR 검색 - 점수 포함으로 변경
    mmr_results = []
    selected_indices = set()
    remaining_indices = list(range(len(documents)))
    
    # 첫 번째는 가장 유사한 문서
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    first_idx = np.argmax(similarities)
    mmr_results.append((documents[first_idx], similarities[first_idx]))
    selected_indices.add(first_idx)
    remaining_indices.remove(first_idx)
    
    # 나머지는 쿼리 유사도와 기존 선택된 문서와의 차이를 고려
    lambda_param = 0.5  # 유사도와 다양성의 균형
    
    for _ in range(min(top_k - 1, len(remaining_indices))):
        best_score = -1
        best_idx = -1
        
        for idx in remaining_indices:
            # 쿼리 유사도
            query_sim = similarities[idx]
            
            # 기존 선택된 문서들과의 최대 유사도
            max_selected_sim = 0
            for selected_idx in selected_indices:
                doc_sim = cosine_similarity([doc_embeddings[idx]], [doc_embeddings[selected_idx]])[0][0]
                max_selected_sim = max(max_selected_sim, doc_sim)
            
            # MMR 점수 계산
            mmr_score = lambda_param * query_sim - (1 - lambda_param) * max_selected_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx != -1:
            mmr_results.append((documents[best_idx], similarities[best_idx]))
            selected_indices.add(best_idx)
            remaining_indices.remove(best_idx)
    
    # 3. 쿼리 재작성
    rewrite_prompt = f'''Please rewrite the following question into a more specific and search-friendly form.
Keep the original intent of the question, but make the keywords clearer and add terms that will help with searching.

User question: {query} 

Rewrite:'''
    
    try:
        rewritten_query = query_rewriter.invoke(rewrite_prompt).content.strip()
        print(f"원본: {query}")
        print(f"rewrite_prompt 재작성: {rewritten_query}")
        
        rewritten_embedding = evaluator.embeddings.encode(rewritten_query)
        rewritten_results = get_top_k_similar(rewritten_embedding, documents, doc_embeddings, top_k)
    except Exception as e:
        print(f"쿼리 재작성 실패: {e}")
        rewritten_results = vector_results
    
    # 4. 쿼리 확장 검색
    try:
        expansion_prompt = f'''Please expand the following question by adding synonyms, similar expressions, and related keywords.
Original question: {query}
Expanded query:'''
        expanded_query = query_rewriter.invoke(expansion_prompt).content.strip()
        print(f"원본: {query}")
        print(f"expansion_prompt 재작성: {expanded_query}")
        
        expanded_embedding = evaluator.embeddings.encode(expanded_query)
        expansion_results = get_top_k_similar(expanded_embedding, documents, doc_embeddings, top_k)
    except:
        expansion_results = vector_results
    
    # 5. 다중 쿼리 검색
    try:
        multi_query_prompt = f'''Please generate 3 different search queries for the following question from different perspectives.  
Each query should aim to find the same information as the original question, but be expressed in a different way.  

Original question: {query}

Query 1:  
Query 2:  
Query 3:
        '''
        multi_response = query_rewriter.invoke(multi_query_prompt).content.strip()
        lines = multi_response.split('\n')
        queries = [line.split(':', 1)[1].strip() for line in lines if ':' in line]
        print(f"원본: {query}")
        print(f"multi_query_prompt 재작성: {multi_response}")
        if len(queries) < 3:
            queries = [query]
        
        # 각 쿼리로 검색
        multi_results = []
        for q in queries[:3]:
            q_embedding = evaluator.embeddings.encode(q)
            q_results = get_top_k_similar(q_embedding, documents, doc_embeddings, 2)  # 각각 2개씩
            multi_results.extend(q_results)
    except:
        multi_results = vector_results
    
    # 점수 조합
    combined_scores = {}
    doc_content_to_object = {}
    
    # 5개 방법에 대한 가중치 설정
    weights = {
        'vector': 0.25,      # 기본 벡터 검색
        'mmr': 0.25,         # MMR 검색
        'rewritten': 0.25,   # 재작성
        'expansion': 0.25,   # 확장
        'multi': 0.25        # 다중 쿼리
    }
    
    # 모든 결과가 이제 (doc, similarity) 형태로 통일됨
    
    # 기본 벡터 검색 결과
    for doc, similarity in vector_results:
        doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weights['vector'] * similarity
        doc_content_to_object[doc_id] = doc
    
    # MMR 검색 결과
    for doc, similarity in mmr_results:
        doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weights['mmr'] * similarity
        doc_content_to_object[doc_id] = doc
    
    # 재작성 검색 결과
    for doc, similarity in rewritten_results:
        doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weights['rewritten'] * similarity
        doc_content_to_object[doc_id] = doc
    
    # 확장 검색 결과
    for doc, similarity in expansion_results:
        doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weights['expansion'] * similarity
        doc_content_to_object[doc_id] = doc
    
    # 다중 쿼리 결과
    for doc, similarity in multi_results:
        doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weights['multi'] * similarity
        doc_content_to_object[doc_id] = doc
    
    # 최종 점수순 정렬 및 반환 (높은 유사도 순)
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    result_docs = []
    for doc_id, score in sorted_docs[:top_k]:
        result_docs.append(doc_content_to_object[doc_id])
    
    return result_docs


from raptor.raptor import RetrievalAugmentation

def Raptor_search(query: str, top_k: int = 5):
    """RAPTOR 검색 - 기존 doc_id 체계 사용"""
    file_name = '자동차보험'
    RA = RetrievalAugmentation(tree=f"/home/ljm/prac/raptor_data/{file_name}")
    
    try:
        # RAPTOR로 검색
        context, layer_info = RA.retrieve(
            question=query,
            start_layer=0,
            num_layers=1,
            top_k=top_k,
            return_layer_information=True
        )
        
        retrieved_docs = []
        for info in layer_info:
            node_index = info['node_index']
            
            # 노드에서 실제 텍스트 가져오기
            node = RA.tree.all_nodes.get(node_index)
            if node:
                # 이 텍스트와 가장 유사한 기존 문서를 vector_store에서 찾기
                
                
                # similar_docs = evaluator.vector_store.similarity_search(
                #     node.text, k=5
                # )
                similar_docs = evaluator.vector_store.max_marginal_relevance_search(
                    node.text, k=5
                )
                if similar_docs:
                    # 기존 문서의 metadata 그대로 사용
                    retrieved_docs.append(similar_docs[0])
        
        return retrieved_docs[:top_k]
    
    except Exception as e:
        print(f"RAPTOR 검색 오류: {e}")
        return []

from langchain.retrievers import BM25Retriever


def evaluate(file_name, user_queries, reference_contexts, reference_answers,evaluator):
    methods_to_test = [
        ("Vector Similarity", vector_search),
        ("MMR Search", mmr_search),
        ("Query Rewriting", query_rewriting),
        ("Query Expansion", query_expansion),
        ("Multi-Query Search", multi_query_search),
        ("agentic_search", agentic_search),
        #("Raptor",Raptor_search)
    ]
    # 나머지 평가 코드는 동일
    all_results = []

    for method_name, method_func in methods_to_test:
        print(f"\n{method_name} 평가 중...")
        result = evaluator.evaluate_retrieval_method(
            user_queries[:100], method_name, method_func, top_k=5  # 테스트를 위해 처음 100개만
        )
        print(result)
        all_results.append(result)
        
        print(f"Recall@1: {result['avg_recall_at_1']:.3f}")
        print(f"Recall@3: {result['avg_recall_at_3']:.3f}")
        print(f"Recall@5: {result['avg_recall_at_5']:.3f}")
        print(f"MRR: {result['avg_mrr']:.3f}")

    # 결과 비교표 생성
    comparison_df = pd.DataFrame([
        {
            "Method": result["method"],
            "Recall@1": f"{result['avg_recall_at_1']:.3f}",
            "Recall@3": f"{result['avg_recall_at_3']:.3f}", 
            "Recall@5": f"{result['avg_recall_at_5']:.3f}",
            "MRR": f"{result['avg_mrr']:.3f}"
        }
        for result in all_results
    ])

    print("\n=== 검색 방법별 성능 비교 ===")
    print(comparison_df.to_string(index=False))

    # # 상세 결과 저장
    # with open("./Test_results/retrieval_evaluation_results.json", "w", encoding="utf-8") as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=2)




    comparison_df.to_json(f"./Test_results/all-MiniLM-L6-v2_safe_sql_{file_name}.json", orient="records", force_ascii=False)
    print(f"\n상세 결과가 저장되었습니다.")


if __name__ == "__main__":
    file_name, user_queries, reference_contexts, reference_answers = load_data()
    evaluator = RetrievalEvaluator(reference_contexts)
    evaluate(file_name, user_queries, reference_contexts, reference_answers,evaluator)

