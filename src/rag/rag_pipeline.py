import time
from typing import Any

from config import BGE_M3_PATH, FAISS_PATH, RERANKER_PATH
from src.models.bge_m3_embedder import BGEEmbedder
from src.rag.context_build import ContextBuilder
from src.rag.generator import Generator
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.reranker import Reranker
from src.retriever.vector_retriever import VectorRetriever
from src.vector_store.vector_store import VectorStore
from collections.abc import Iterator


class RAGPipeline:
    """串联检索、上下文构造和回答生成。"""

    def __init__(
            self,
            retriever: Any,
            context_builder: ContextBuilder,
            generator: Generator,
            top_k: int = 10,
            fusion_k: int = 10,
            candidate_k: int = 25
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if fusion_k <= 0:
            raise ValueError("fusion_k must be positive")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator

        self.top_k = top_k
        self.fusion_k = fusion_k
        self.candidate_k = candidate_k

    def query(self, question: str):
        """执行完整 RAG 流程，返回回答和来源。"""

        if not question or not question.strip():
            raise ValueError("question must not be empty")


        results = self.retriever.retrieve(
            query=question,
            top_k=self.top_k,
            fusion_k=self.fusion_k,
            candidate_k=self.candidate_k
        )

        context_result = self.context_builder.build(results)

        # 3. 生成回答
        answer = self.generator.generate(
            question=question,
            context=context_result["context"],
        )

        # 4. 组装统一结果
        return {
            "question": question,
            "answer": answer,
            "sources": context_result["sources"],
            "retrieval": {
                "chunk_count": context_result["chunk_count"],
                "context_chars": context_result["context_chars"],
            },
        }

    def query_stream(self, question: str) -> Iterator[dict[str, Any]]:
        """流式执行 RAG，依次输出来源、文本片段和结束事件。"""

        if not question or not question.strip():
            raise ValueError("question must not be empty")

        pipeline_start = time.perf_counter()

        retrieval_start = time.perf_counter()

        # 1. 检索相关 Chunk
        results = self.retriever.retrieve(
            query=question,
            top_k=self.top_k,
            fusion_k=self.fusion_k,
            candidate_k=self.candidate_k,
        )

        retrieval_seconds = time.perf_counter() - retrieval_start


        # 2. 构造上下文和来源
        context_start = time.perf_counter()

        context_result = self.context_builder.build(results)

        context_seconds = time.perf_counter() - context_start

        context = context_result["context"]
        sources = context_result["sources"]

        retrieval = {
            "chunk_count": context_result["chunk_count"],
            "context_chars": context_result["context_chars"],
        }

        # 3. 先发送来源，让前端可以提前展示引用
        yield {
            "type": "sources",
            "sources": sources,
            "retrieval": retrieval,
            "timings": {
                "retrieval_seconds": round(retrieval_seconds, 3),
                "context_seconds": round(context_seconds, 3),
            }
        }

        # 4. 逐段生成回答
        full_answer = []

        generation_start = time.perf_counter()
        first_token_seconds = None

        for content in self.generator.generate_stream(
                question=question,
                context=context,
        ):

            if content and first_token_seconds is None:
                first_token_seconds = time.perf_counter() - generation_start

            full_answer.append(content)

            yield {
                "type": "token",
                "content": content,
            }
        generation_seconds = time.perf_counter() - generation_start
        pipeline_seconds = time.perf_counter() - pipeline_start

        timings = {
            "retrieval_seconds": round(retrieval_seconds, 3),
            "context_seconds": round(context_seconds, 3),
            "first_token_seconds": (
                round(first_token_seconds, 3)
                if first_token_seconds is not None
                else None
            ),
            "generation_seconds": round(generation_seconds, 3),
            "pipeline_seconds": round(pipeline_seconds, 3),
        }

        # 5. 输出性能指标
        yield {
            "type":"metrics",
            "timings": timings,
        }

        # 6. 发送结束事件
        yield {
            "type": "done",
            "question": question,
            "answer": "".join(full_answer),
            "sources": sources,
            "retrieval": retrieval,
            "timings": timings
        }




if __name__ == "__main__":
    embedder = BGEEmbedder(model_path=BGE_M3_PATH)

    print("Loading FAISS...")

    vector_store = VectorStore.load(FAISS_PATH)

    print(f"Vector Count: {vector_store.size}")

    print(f"Embedding  Dimension: {vector_store.faiss_store.dimension}")

    # 向量检索
    vector_retriever = VectorRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # BM25检索
    bm25_retriever = BM25Retriever(
        chunks=vector_store.chunks
    )

    reranker = Reranker(
        model_path=RERANKER_PATH,
        use_fp16=True,
        devices=["cuda:0"],
        normalize=True,
    )

    hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever, reranker)

    generator = Generator()
    context_builder = ContextBuilder()

    # retriever 使用你现有的初始化代码
    pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        context_builder=context_builder,
        generator=generator,
    )

    # result = pipeline.query(
    #     "车辆动力电池健康状态如何评估？"
    # )
    #
    # print(result["answer"])
    # print(result["sources"])

    for event in pipeline.query_stream(
            "车辆动力电池健康状态如何评估？"
    ):
        if event["type"] == "sources":
            print("\n参考来源：", event["sources"])

        elif event["type"] == "token":
            print(event["content"], end="", flush=True)

        elif event["type"] == "done":
            print("\n回答生成完成")