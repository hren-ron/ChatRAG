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

        expected = item["expected_documents"]

        results = retriever.retrieve(query=query, top_k=top_k)

        chunks = [result["chunk"] for result in results]

        metrics = {
            "recall@1":
                RetrievalMetrics.recall_at_k(
                    expected,
                    chunks,
                    1,
                ),

            "recall@3":
                RetrievalMetrics.recall_at_k(
                    expected,
                    chunks,
                    3,
                ),

            "recall@5":
                RetrievalMetrics.recall_at_k(
                    expected,
                    chunks,
                    5,
                ),

            "recall@10":
                RetrievalMetrics.recall_at_k(
                    expected,
                    chunks,
                    10,
                ),

            "precision@5":
                RetrievalMetrics.precision_at_k(
                    expected,
                    chunks,
                    5,
                ),

            "mrr":
                RetrievalMetrics.reciprocal_rank(
                    expected,
                    chunks,
                ),
        }

        # 转换成可以保存到 JSON 的普通字典
        retrieved_results = []

        for result in results:
            chunk = result["chunk"]

            retrieved_results.append({
                "score": float(result["score"]),
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "chapter": chunk.chapter,
                "title": chunk.title,
                "level": chunk.level,
                "token_count": chunk.token_count,
            })

        return {
            "id": item["id"],
            "query": query,
            "expected_documents": expected,
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "metrics": metrics,
            "results": retrieved_results,
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

