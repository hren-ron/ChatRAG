
class VectorRetriever:

    def __init__(self, embedder, vector_store):

        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int=5):

        query_embedding = self.embedder.encode([query], batch_size=1, show_progress_bar=False)

        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results

