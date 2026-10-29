from config import BGE_M3_PATH, TXT_PATH
from src.chunker.run_chunker import ChunkPipeline
from src.models.bge_m3_embedder import BGEEmbedder

embedder = BGEEmbedder(BGE_M3_PATH)


pipeline = ChunkPipeline(
    input_dir=TXT_PATH,
    tokenizer=embedder.tokenizer,
    max_tokens=600,
    overlap_tokens=80,
)

chunks = pipeline.run()

print("最终 Retrieval Chunk 数量:", len(chunks))