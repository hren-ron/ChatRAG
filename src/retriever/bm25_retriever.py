from typing import List

from rank_bm25 import BM25Okapi

from src.chunker.models import Chunk


class BM25Retriever:

    def __init__(self, chunks: List[Chunk]):
        """
        使用 Chunk.content 建立 BM25 索引。

        参数：
            chunks:
                SecondaryChunker 之后的 Chunk 列表
        """

        self.chunks = [chunk for chunk in chunks if not chunk.is_deleted]

        corpus = [self._tokenize(chunk.content) for chunk in chunks]

        self.bm25 = BM25Okapi(corpus)

    def _tokenize(self, text: str):
        """
        第一版采用字符级切分。

        后续可以替换成：
        jieba / pkuseg / 专业词典
        """

        return list(text)


    def retrieve(self, query: str, top_k: int = 10):

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)

        results = []

        for index in ranked_indices[:top_k]: 
            chunk = self.chunks[index]

            results.append({
                "chunk": chunk,
                "score": float(scores[index])
            })

        return results