from config import FAISS_PATH, BGE_M3_PATH
from src.models.bge_m3_embedder import BGEEmbedder
from src.retriever.vector_retriever import VectorRetriever
from src.vector_store.vector_store import VectorStore


query = "车辆外摆值如何测量？"

vector_store = VectorStore.load(FAISS_PATH)

print("向量数量:", vector_store.size)
print("Chunk 数量:", len(vector_store.chunks))
print("向量维度:", vector_store.faiss_store.dimension)

embedder = BGEEmbedder(model_path=BGE_M3_PATH)

retriever = VectorRetriever(embedder=embedder, vector_store=vector_store)

results = retriever.retrieve(
    query,
    top_k=5,
)

for i, result in enumerate(results, 1):

    chunk = result["chunk"]

    print("=" * 80)
    print(f"Rank: {i}")
    print(f"Score: {result['score']:.4f}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Title: {chunk.title}")
    print(f"Parent Path: {chunk.parent_path}")
    print()
    print(chunk.content)