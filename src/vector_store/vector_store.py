import json
from pathlib import Path

from src.chunker.models import Chunk
from src.vector_store.faiss_store import FaissStore


class VectorStore:

    def __init__(self, dimension: int, chunks=None):

        self.faiss_store = FaissStore(dimension)
        self.chunks = chunks or []

    @property
    def size(self):
        return self.faiss_store.size

    def add(self, chunks, embeddings):

        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks and embeddings size mismatch: {len(chunks) != len(embeddings)}")

        self.faiss_store.add(embeddings)

        self.chunks = chunks

    def search(self, query_embedding, top_k=5):

        scores, indices = self.faiss_store.search(query_embedding, top_k)

        results = []

        for score, vector_id in zip(scores[0], indices[0]):
            index = int(vector_id)

            if index < 0:
                continue

            if index > len(self.chunks):
                continue


            results.append(
                {
                    "score": float(score),
                    "chunk": self.chunks[index]
                }
            )
        return results

    def save(self, directory: str):

        directory = Path(directory)

        directory.parent.mkdir(parents=True, exist_ok=True)

        # 1. 保存 FAISS
        self.faiss_store.save(str(directory / "index.faiss"))

        chunks_path = (directory / "chunks.json")

        datas = [chunk.model_dump() for chunk in self.chunks]

        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump(datas, f, ensure_ascii=False, indent=2)

        print(f"VectorStore has been saved")

    @classmethod
    def load(cls, directory):

        directory = Path(directory)

        index_path = (directory / "index.faiss")

        chunks_path = (directory / "chunks.json")

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not exists")

        if not chunks_path.exists():
            raise FileNotFoundError(f"chunks json not exists")

        # 1. 加载 FAISS
        faiss_store = FaissStore.load(str(index_path))

        with chunks_path.open("r", encoding="utf-8") as f:
            data = json.load(chunks_path)

        chunks = [Chunk.model_validate(item) for item in data]

        store = cls(dimension=faiss_store.dimension, chunks=chunks)

        store.faiss_store = faiss_store
        if store.size != len(store.chunks):
            raise ValueError(f"the number of FAISS and Chunks is mismatch: vectors={store.size}, chunks={len(store.chunks)}")

        return store
