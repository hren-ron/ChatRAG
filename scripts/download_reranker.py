from modelscope import snapshot_download

model_dir = snapshot_download(
    "BAAI/bge-reranker-v2-m3",
    local_dir=r"D:\PythonCode\ChatRAG\Models\bge-reranker-v2-m3",
)

print("模型下载位置：", model_dir)