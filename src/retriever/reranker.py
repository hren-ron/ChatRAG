from pathlib import Path
from typing import List, Any

from FlagEmbedding import FlagReranker

class Reranker:
     """使用 BGE Reranker 对候选 Chunk 进行精排。"""
     def __init__(
             self,
             model_path: str,
             use_fp16: bool = False,
             devices: list[str] | None = None,
             normalize: bool = False
     ):
         model_dir = Path(model_path)

         if not model_dir.is_dir():
             raise FileNotFoundError(f"Reranker model directory not found: {model_dir}")

         self.normalize = normalize

         kwargs = {"use_fp16": use_fp16}
         if devices is not None:
             kwargs["devices"] = devices

         self.model = FlagReranker(str(model_dir), **kwargs)

     def rerank(self, query: str, candidates: List[dict[str, Any]], top_k: int=5):
         """对候选 Chunk 重新打分并返回 Top-K。"""

         if top_k <= 0 or not candidates:
             return []

         # 构造 Query-Chunk 文本对
         pairs = [[query, result["chunk"].content] for result in candidates]

         # 批量计算相关性分数
         scores = self.model.compute_score(pairs, normalize = self.normalize)

         # 兼容只有一个候选时返回单个数值的情况
         if isinstance(scores, (int, float)):
             scores = [scores]

         reranked_results = []

         for result, score in zip(candidates, scores):
             reranked_results.append({
                 "chunk": result["chunk"],
                 "fusion_score": result.get("score"),
                 "score": float(score),
             })

         reranked_results.sort(key=lambda item:item["score"], reverse=True)

         return reranked_results[:top_k]


if __name__ == "__main__":

    reranker = Reranker(
        model_path=(
            r"D:\PythonCode\ChatRAG\Models"
            r"\bge-reranker-v2-m3"
        ),
        use_fp16=True,
        devices=["cuda:0"],
        normalize=True,
    )

    query = "动力蓄电池健康状态如何评估？"

    candidates = [
        {
            "chunk": type("Chunk", (), {
                "chunk_id": "A",
                "content": "动力蓄电池健康状态应依据相关指标和评价方法进行评估。",
            })(),
            "score": 0.01,
        },
        {
            "chunk": type("Chunk", (), {
                "chunk_id": "B",
                "content": "车辆空调系统的制冷性能测试方法。",
            })(),
            "score": 0.02,
        },
    ]

    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=2,
    )

    for rank, result in enumerate(results, start=1):
        print(
            rank,
            result["chunk"].chunk_id,
            result["score"],
            result["chunk"].content,
        )

