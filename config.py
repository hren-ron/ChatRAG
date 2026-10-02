from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 数据集目录
PDF_PATH = PROJECT_ROOT / "data" / "pdf"
# 数据集解析成txt的目录
OUTPUT_PATH = PROJECT_ROOT / "data" / "output"

# model目录
MODEL_DIR = PROJECT_ROOT.parent / "Models"
# bge-m3模型路径
BGE_M3_PATH = MODEL_DIR / "bge-m3"