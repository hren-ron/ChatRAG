
import numpy as np


class EmbeddingPipeline:

    def __init__(
        self,
        embedder,
        batch_size: int = 32,
        embedding_model: str="BAAI/bge-m3",
        embedding_model_version=None
    ):
        self.embedder = embedder
        self.batch_size = batch_size

        self.embedding_model = embedding_model
        self.embedding_model_version = embedding_model_version

    def run(self, chunks):
        """
        对 Retrieval Chunk 进行 Embedding。

        Returns
        -------
        chunks:
            原始 Chunk

        embeddings:
            shape = (N, dimension)
        """

        if not chunks:
            raise ValueError("Not found Retrieval Chunk")

        texts = [chunk.content for chunk in chunks]

        print(f"Start Embedding：{len(texts)} 个 chunks")

        embeddings = self.embedder.encode(
            texts,
            batch_size=self.batch_size,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding number is mismatch："
                f"chunks={len(chunks)}, "
                f"embeddings={len(embeddings)}"
            )

        print(f"Embedding finished：shape={embeddings.shape}")

        dimension = embeddings.shape[1]

        # 给 Chunk 补充 Embedding 元数据
        for chunk in chunks:
            chunk.embedding_model = self.embedding_model
            chunk.embedding_model_version = self.embedding_model_version
            chunk.embedding_dimension = dimension

        return chunks, embeddings