from config import BGE_M3_PATH
from src.models.bge_m3_embedder import BGEEmbedder

MODEL_PATH = r"D:\PythonCode\Models\bge-m3"

print(BGE_M3_PATH)
print(type(BGE_M3_PATH))
print("exists:", BGE_M3_PATH.exists())
print("is_dir:", BGE_M3_PATH.is_dir())

embedder = BGEEmbedder(BGE_M3_PATH)

texts = [
    "车辆的制动性能应满足相关技术要求",
    "汽车制动系统应符合规定",
]

embeddings = embedder.encode(texts)

print(embeddings.shape)
print(embedder.dimension)