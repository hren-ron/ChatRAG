from pathlib import Path

from src.vector_store.vector_store import VectorStore
from src.retriever.bm25_retriever import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAISS_PATH = (
    PROJECT_ROOT
    / "data"
    / "faiss"
)


def main():

    # =========================
    # 加载已有 Chunk
    # =========================

    store = VectorStore.load(
        FAISS_PATH
    )

    chunks = store.chunks

    print("=" * 60)
    print(f"Chunk count: {len(chunks)}")
    print("=" * 60)

    # =========================
    # 创建 BM25
    # =========================

    retriever = BM25Retriever(
        chunks
    )

    # =========================
    # Query
    # =========================

    query = (
        "动力蓄电池健康状态如何评价"
    )

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    # =========================
    # 输出
    # =========================

    print()
    print("=" * 60)
    print("BM25 Results")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        chunk = result["chunk"]

        print()
        print(f"Rank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Document ID: {chunk.document_id}")
        print(f"Document: {chunk.document}")
        print(f"Chapter: {chunk.chapter}")
        print(f"Title: {chunk.title}")
        print(
            f"Page: "
            f"{chunk.page_start}-{chunk.page_end}"
        )
        print(
            f"Content: "
            f"{chunk.content[:300]}"
        )


if __name__ == "__main__":
    main()