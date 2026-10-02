


class VectorRetriever:

    def __init__(self, embedder, vector_store, chunks):

        self.embedder = embedder
        self.vector_store = vector_store

        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def retrieve(self, query: str, top_k: int=5):

        query_embedding = self.embedder.encode([query], batch_size=1, show_progress_bar=False)

        results = self.vector_store.search(query_embedding, top_k=top_k)

        retrieved_chunks = []
        for result in results:
            chunk_id = result["chunk_id"]

            chunk = self.chunk_by_id.get(chunk_id)

            if chunk is None:
                continue

            retrieved_chunks.append(
                {
                    "score":result["score"],
                    "chunk": chunk
                }
            )
        return retrieved_chunks