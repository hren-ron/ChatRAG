from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)

    @property
    def size(self):
        return self.index.ntotal

    def add(self, embeddings: np.ndarray):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"embedding dimension mismatch: expected={self.dimension}, got={embeddings.shape[1]}")

        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):

        query_embedding = np.asarray(query_embedding, dtype="float32")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"embedding dimension mismatch: expected={self.dimension}, got={query_embedding.shape[1]}")

        if self.size == 0:
            return (
                np.empty((1, 0), dtype=np.float32),
                np.empty((1, 0), dtype=np.int64),
            )

        top_k = min(top_k, self.size)

        scores, indices = self.index.search(query_embedding, top_k)

        return scores, indices

    def save(self, path: str):

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))


    @classmethod
    def load(cls, path: str):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")

        index = faiss.read_index(str(path))

        store = cls(index.d)

        store.index = index

        return store
