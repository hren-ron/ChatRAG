
from config import OUTPUT_PATH, BGE_M3_PATH
from src.chunker.run_chunker import ChunkPipeline
from src.models.bge_m3_embedder import BGEEmbedder


embedder = BGEEmbedder(BGE_M3_PATH)


chunker_pipeline = ChunkPipeline(
    input_dir=OUTPUT_PATH,
    tokenizer=embedder.tokenizer,
    max_tokens=600,
    overlap_tokens=80,
)

chunks = chunker_pipeline.run()

texts = [chunk.content for chunk in chunks]
print(len(chunks))

embeddings = embedder.encode(texts=texts, batch_size=32)
print(embeddings.shape)