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
EVALUATION_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_questions.json"

# FAISS评估结果
FAISS_RESULT_PATH = PROJECT_ROOT / "data" / "eval" / "faiss_bge_m3_result.json"
