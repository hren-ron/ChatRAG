from src.evaluator.metrics import RetrievalMetrics
from src.retriever.test_vector_search import retriever


def aggregate(results):
    n = len(results)

    return {
        "recall@1": sum(
            r["metrics"]["recall@1"]
            for r in results
        ) / n,

        "recall@3": sum(
            r["metrics"]["recall@3"]
            for r in results
        ) / n,

        "recall@5": sum(
            r["metrics"]["recall@5"]
            for r in results
        ) / n,

        "recall@10": sum(
            r["metrics"]["recall@10"]
            for r in results
        ) / n,

        "precision@5": sum(
            r["metrics"]["precision@5"]
            for r in results
        ) / n,

        "mrr": sum(
            r["metrics"]["mrr"]
            for r in results
        ) / n,
    }


class Evaluator:

    def __init__(self, retriever):
        self.retriever = retriever

    def evaluate_query(self, item, top_k: int=10):

        query = item["query"]

        expected_documents = item.get("expected_documents", [])

        expected_chunk_ids = item.get("expected_chunks", [])

        results = retriever.retrieve(query=query, top_k=top_k)

        retrieved_chunks = [result["chunk"] for result in results]

        # =========================
        # Document-level metrics
        # =========================

        document_metrics = {
            "recall@1":
                RetrievalMetrics.document_recall_at_k(
                    expected_documents,
                    retrieved_chunks,
                    1,
                ),

            "recall@3":
                RetrievalMetrics.document_recall_at_k(
                    expected_documents,
                    retrieved_chunks,
                    3,
                ),

            "recall@5":
                RetrievalMetrics.document_recall_at_k(
                    expected_documents,
                    retrieved_chunks,
                    5,
                ),

            "recall@10":
                RetrievalMetrics.document_recall_at_k(
                    expected_documents,
                    retrieved_chunks,
                    10,
                ),

            "precision@5":
                RetrievalMetrics.document_precision_at_k(
                    expected_documents,
                    retrieved_chunks,
                    5,
                ),

            "mrr":
                RetrievalMetrics.document_reciprocal_rank(
                    expected_documents,
                    retrieved_chunks,
                ),
        }

        # =========================
        # Chunk-level metrics
        # =========================

        chunk_metrics = {
            "recall@1":
                RetrievalMetrics.chunk_recall_at_k(
                    expected_chunk_ids,
                    retrieved_chunks,
                    1,
                ),

            "recall@3":
                RetrievalMetrics.chunk_recall_at_k(
                    expected_chunk_ids,
                    retrieved_chunks,
                    3,
                ),

            "recall@5":
                RetrievalMetrics.chunk_recall_at_k(
                    expected_chunk_ids,
                    retrieved_chunks,
                    5,
                ),

            "recall@10":
                RetrievalMetrics.chunk_recall_at_k(
                    expected_chunk_ids,
                    retrieved_chunks,
                    10,
                ),

            "precision@5":
                RetrievalMetrics.chunk_precision_at_k(
                    expected_chunk_ids,
                    retrieved_chunks,
                    5,
                ),

            "mrr":
                RetrievalMetrics.chunk_reciprocal_rank(
                    expected_chunk_ids,
                    retrieved_chunks,
                ),
        }

        # =========================
        # Serializable results
        # =========================

        retrieved_results = []

        for rank, result in enumerate(results, start=1):
            chunk = result["chunk"]

            retrieved_results.append({
                "rank": rank,
                "score": float(result["score"]),
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "chapter": chunk.chapter,
                "title": chunk.title,
                "level": chunk.level,
                "parent_path": chunk.parent_path,
                "token_count": chunk.token_count,
            })

        return {
            "id": item["id"],
            "query": query,

            "expected_documents":
                expected_documents,

            "expected_chunk_ids":
                expected_chunk_ids,

            "category":
                item.get("category"),

            "difficulty":
                item.get("difficulty"),

            "document_metrics":
                document_metrics,

            "chunk_metrics":
                chunk_metrics,

            "results":
                retrieved_results,
        }

    def evaluate(self, dataset, top_k=10):

        results = []

        for index, item in enumerate(dataset, start=1):

            id = item["id"]
            query = item["query"]
            print(f"[{index}/{len(dataset)}] id={id} query={query}")

            result = self.evaluate_query(item, top_k=top_k)

            results.append(result)
        return results

