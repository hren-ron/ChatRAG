from pathlib import Path

from config import TXT_PATH, BGE_M3_PATH, FAISS_PATH
from src.build.build_vector_store import VectorStoreBuilder
from src.chunker.run_chunker import ChunkPipeline
from src.models.bge_m3_embedder import BGEEmbedder

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==============================
# 路径
# ==============================
CHUNK_INPUT_DIR = TXT_PATH

MODEL_PATH = BGE_M3_PATH

VECTORSTORE_DIR = FAISS_PATH

# ==============================
# BGE-M3
# ==============================

embedder = BGEEmbedder(
    model_path=MODEL_PATH
)

# ==============================
# Chunk Pipeline
# ==============================

chunk_pipeline = ChunkPipeline(
    input_dir=str(CHUNK_INPUT_DIR),
    tokenizer=embedder.tokenizer,
    max_tokens=600,
    min_tokens=100,
    overlap_tokens=80,
)

# ==============================
# Vector Store Builder
# ==============================

builder = VectorStoreBuilder(
    chunk_pipeline=chunk_pipeline,
    embedder=embedder,
    output_dir=VECTORSTORE_DIR,
    batch_size=32,
)

# ==============================
# Build
# ==============================

vector_store = builder.build()

print(
    f"Vector store size: "
    f"{vector_store.size}"
)