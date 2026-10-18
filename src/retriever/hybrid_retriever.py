from src.retriever.rank_fusion import reciprocal_rank_fusion


class HybridRetriever:

    def __init__(self, bm25_retriever, vector_retriever):
        self.bm25_retriever = bm25_retriever

        self.vector_retriever = vector_retriever

    def retrieve(self, query: str, top_k: int=5, candidate_k: int=25):

        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)

        vector_results = self.vector_retriever.retrieve(query, top_k=candidate_k)

        # 3. RRF 排名融合
        results = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            top_k=top_k,
        )

        return results