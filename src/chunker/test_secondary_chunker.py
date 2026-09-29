from transformers import AutoTokenizer

from src.chunker.chunk_loader import ChunkLoader
from src.chunker.secondary_chunker import SecondaryChunker


loader = ChunkLoader(
    input_dir="../../data/output"
)

structured_chunks = loader.load()


tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-m3"
)

chunker = SecondaryChunker(
    tokenizer=tokenizer,
    max_tokens=600,
    min_tokens=100,
    overlap_tokens=80,
)


retrieval_chunks = []

for chunk in structured_chunks:
    chunks = chunker.split(chunk)
    retrieval_chunks.extend(chunks)


print(
    f"Structured Chunks: {len(structured_chunks)}"
)

print(
    f"Retrieval Chunks: {len(retrieval_chunks)}"
)


for chunk in retrieval_chunks[:5]:
    print("=" * 80)
    print("chunk_id:", chunk.chunk_id)
    print("chapter:", chunk.chapter)
    print("title:", chunk.title)
    print("level:", chunk.level)
    print("parent_path:", chunk.parent_path)
    print("token_count:", chunk.token_count)
    print(chunk.content)