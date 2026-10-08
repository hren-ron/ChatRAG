from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# pdf数据集目录
PDF_PATH = PROJECT_ROOT / "data" / "pdf"
# pdf解析成txt的目录
TXT_PATH = PROJECT_ROOT / "data" / "txt"

# model目录
MODEL_DIR = PROJECT_ROOT / "models"
# bge-m3模型路径
BGE_M3_PATH = MODEL_DIR / "bge-m3"


# FAISS 目录
FAISS_PATH = PROJECT_ROOT / "data" / "faiss"

# 评估数据集
DOCUMENT_EVALUATION_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_questions.json"
CHUNK_EVALUATION_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_questions_top_50.json"

# 评估结果
DOCUMENT_RESULT_PATH = PROJECT_ROOT / "data" / "eval" / "bge_document_level_result.json"
CHUNK_RESULT_PATH = PROJECT_ROOT / "data" / "eval" / "bge_chunk_level_result.json"