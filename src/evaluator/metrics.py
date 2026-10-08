from typing import List


class RetrievalMetrics:

    # =========================
    # Document-level
    # =========================

    @staticmethod
    def document_recall_at_k(
            expected_documents: List[str],
            retrieved_chunks: list,
            k: int,
    ) -> float:
        """
        文档级 Recall@K

        判断 Top-K Retrieval Chunks 中，
        有多少不同的 expected documents 被召回。
        """
        expected = set(expected_documents)

        retrieved = {
            chunk.document
            for chunk in retrieved_chunks[:k]
            if chunk.document
        }

        if not expected:
            return 0.0

        return len(expected & retrieved) / len(expected)

    @staticmethod
    def document_precision_at_k(
            expected_documents: List[str],
            retrieved_chunks: list,
            k: int,
    ) -> float:
        """
        文档级 Precision@K

        注意：
        这里按照“去重后的 document 数量”计算，
        而不是直接用 K 作为分母。

        因此这是 Unique Document Precision@K。
        """
        expected = set(expected_documents)

        retrieved = {
            chunk.document
            for chunk in retrieved_chunks[:k]
            if chunk.document
        }

        if not retrieved:
            return 0.0

        return len(expected & retrieved) / len(retrieved)

    @staticmethod
    def document_reciprocal_rank(
            expected_documents: List[str],
            retrieved_chunks: list,
    ) -> float:
        """
        文档级 MRR。

        找到第一个属于 expected_documents 的 chunk，
        根据其 rank 计算 reciprocal rank。
        """
        expected = set(expected_documents)

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if chunk.document in expected:
                return 1.0 / rank

        return 0.0

    # =========================
    # Chunk-level
    # =========================

    @staticmethod
    def chunk_recall_at_k(
            expected_chunk_ids: List[str],
            retrieved_chunks: list,
            k: int,
    ) -> float:
        """
        Chunk-level Recall@K。

        expected_chunk_ids:
            人工标注的答案相关 Chunk。

        retrieved_chunks:
            Retriever 返回的 Chunk。
        """
        expected = set(expected_chunk_ids)

        retrieved = {
            chunk.chunk_id
            for chunk in retrieved_chunks[:k]
        }

        if not expected:
            return 0.0

        return len(expected & retrieved) / len(expected)

    @staticmethod
    def chunk_precision_at_k(
            expected_chunk_ids: List[str],
            retrieved_chunks: list,
            k: int,
    ) -> float:
        """
        Chunk-level Precision@K。

        这里按照实际返回的 unique chunk 数量计算。
        """
        expected = set(expected_chunk_ids)

        retrieved = {
            chunk.chunk_id
            for chunk in retrieved_chunks[:k]
        }

        if not retrieved:
            return 0.0

        return len(expected & retrieved) / len(retrieved)

    @staticmethod
    def chunk_reciprocal_rank(
            expected_chunk_ids: List[str],
            retrieved_chunks: list,
    ) -> float:
        """
        Chunk-level MRR。

        找到第一个 expected chunk 后：
            RR = 1 / rank
        """
        expected = set(expected_chunk_ids)

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if chunk.chunk_id in expected:
                return 1.0 / rank

        return 0.0