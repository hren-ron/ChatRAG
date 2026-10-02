
from pathlib import Path

import numpy as np

from src.embedding.embedding_pipeline import EmbeddingPipeline
from src.vector_store.vector_store import VectorStore


class VectorStoreBuilder:

    def __init__(
            self,
            chunk_pipeline,
            embedder,
            output_dir,
            batch_size=32
    ):
        self.chunk_pipeline = chunk_pipeline
        self.embedding_pipeline = EmbeddingPipeline(embedder=embedder, batch_size=batch_size)
        self.output_dir = Path(output_dir)


    def build(self):
        # ==================================
        # 1. 获取 Retrieval Chunks
        # ==================================

        print("=" * 60)
        print("Step 1: Loading Retrieval Chunks")
        print("=" * 60)

        chunks = self.chunk_pipeline.run()

        print(f"Retrieval Chunks: {len(chunks)}")

        if not chunks:
            raise ValueError(f"not have retrieval chunks")

        # ==================================
        # 2. Embedding
        # ==================================

        print("=" * 60)
        print("Step 2: Generating Embeddings")
        print("=" * 60)

        chunks, embeddings = self.embedding_pipeline.run(chunks=chunks)
        embeddings = np.asarray(embeddings, dtype="float32")

        print(f"Embedding shapes: {embeddings.shape}")

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings size mismatch: "
                f"{len(chunks)} != {len(embeddings)}"
            )

        # ==================================
        # 3. 创建 VectorStore
        # ==================================

        print("=" * 60)
        print("Step 3: Building FAISS Index")
        print("=" * 60)

        dimension = embeddings.shape[1]

        vector_store = VectorStore(dimension=dimension)
        vector_store.add(chunks=chunks, embeddings=embeddings)

        print(f"FAISS vectors: {vector_store.size}")

        # ==================================
        # 4. 持久化
        # ==================================

        print("=" * 60)
        print("Step 4: Persisting Vector Store")
        print("=" * 60)

        vector_store.save(str(self.output_dir))

        print("=" * 60)
        print("Build completed")
        print("=" * 60)

        return vector_store
