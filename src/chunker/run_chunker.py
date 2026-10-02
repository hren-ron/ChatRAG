from src.chunker.chunk_loader import ChunkLoader
from src.chunker.secondary_chunker import SecondaryChunker


class ChunkPipeline:

    def __init__(
            self,
            input_dir: str,
            tokenizer,
            max_tokens: int = 600,
            min_tokens: int = 100,
            overlap_tokens: int = 80
    ):

        self.loader = ChunkLoader(input_dir)

        self.chunker = SecondaryChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            overlap_tokens=overlap_tokens
        )

    def run(self):
        # 1. 加载 Structured Chunk
        structured_chunks = self.loader.load()

        # 2. 二次切分
        retrieval_chunks = []

        for chunk in structured_chunks:
            chunks = self.chunker.split(chunk)
            retrieval_chunks.extend(chunks)

        return retrieval_chunks
