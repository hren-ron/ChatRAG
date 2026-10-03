

class RetrievalMetrics:

    @staticmethod
    def recall_at_k(expected_documents, retrieved_chunks, k):

        expected = set(expected_documents)

        retrieved = {chunk.document for chunk in retrieved_chunks[:k]}

        if not retrieved:
            return 0.0

        return len(expected & retrieved) / len(expected)

    @staticmethod
    def precision_at_k(expected_documents, retrieved_chunks, k):

        expected = set(expected_documents)

        retrieved = {chunk.document for chunk in retrieved_chunks}

        if not retrieved:
            return 0.0

        return len(expected & retrieved) / len(retrieved)

    @staticmethod
    def reciprocal_rank(expected_documents, retrieved_chunks):
        expected = set(expected_documents)

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if chunk.document in expected:
                return 1.0 / rank
        return 0.0