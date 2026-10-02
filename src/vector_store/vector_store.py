from pathlib import Path

from src.vector_store.faiss_store import FaissStore
from src.vector_store.metadata_store import MetadataStore


class VectorStore:

    def __init__(self, dimension: int):

        self.faiss_store = FaissStore(dimension)
        self.metadata_store = MetadataStore()

    def add(self, chunks, embeddings):

        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks and embeddings size mismatch: {len(chunks) != len(embeddings)}")

        self.faiss_store.add(embeddings)

        self.metadata_store.build(chunks)

    def search(self, query_embedding, top_k=5):

        scores, indices = self.faiss_store.search(query_embedding, top_k)

        results = []

        for score, vector_id in zip(scores[0], indices[0]):
            if vector_id < 0:
                continue

            chunk_id = self.metadata_store.get_chunk_id(int(vector_id))

            results.append(
                {
                    "vector_id": int(vector_id),
                    "chunk_id":chunk_id,
                    "score": float(score)
                }
            )
        return results

    def save(self, directory: str):

        directory = Path(directory)

        directory.parent.mkdir(parents=True, exist_ok=True)

        self.faiss_store.save(str(directory / "index.faiss"))

        self.metadata_store.save(str(directory / "metadata.json"))