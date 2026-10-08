import json
from collections import defaultdict
from pathlib import Path

from config import BGE_M3_PATH, FAISS_PATH, CHUNK_EVALUATION_PATH, \
    DOCUMENT_EVALUATION_PATH, DOCUMENT_RESULT_PATH, CHUNK_RESULT_PATH
from src.evaluator.evaluator import Evaluator
from src.models.bge_m3_embedder import BGEEmbedder
from src.retriever.vector_retriever import VectorRetriever
from src.vector_store.vector_store import VectorStore


def load_dataset(path: Path):

    if not path.exists():
        raise FileNotFoundError("test dataset not exists")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def calculate_average(results, metric_group):
    """
    metric_group:
        document_metrics
        chunk_metrics
    """

    if not results:
        return {}

    metric_names = results[0][metric_group].keys()

    summary = {}

    for metric_name in metric_names:

        values = [
            result[metric_group][metric_name]
            for result in results
        ]

        summary[metric_name] = (
            sum(values) / len(values)
        )

    return summary


def aggregate_by_field(results, metric_group, field):

    groups = defaultdict(list)

    for result in results:

        value = result.get(field)

        if value is None:
            continue

        groups[value].append(result)
    return {
        value: calculate_average(items, metric_group=metric_group)
        for value, items in groups.items()
    }

def save_results(path: Path, overall, by_difficulty, by_category, results):

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "overall": overall,
        "by_difficulty": by_difficulty,
        "by_category": by_category,
        "queries": results
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_metrics(title, metrics):
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)

    for name, value in metrics.items():
        print(
            f"{name:<15}: {value:.4f}"
        )

def main():

    print("=" * 60)
    print("BGE-M3 + FAISS Retrieval Evaluation")
    print("=" * 60)

    # --------------------------------------------------
    # 1. 加载评测数据
    # --------------------------------------------------

    document_dataset = load_dataset(DOCUMENT_EVALUATION_PATH)
    chunk_dataset = load_dataset(CHUNK_EVALUATION_PATH)

    print(f"Document Level Evaluation Dataset: {len(document_dataset)}")
    print(f"Chunk Level Evaluation Dataset: {len(chunk_dataset)}")

    # --------------------------------------------------
    # 2. 加载 BGE-M3
    # --------------------------------------------------

    print()
    print("Loading BGE-M3...")

    embedder = BGEEmbedder(model_path=BGE_M3_PATH)

    # --------------------------------------------------
    # 3. 加载 FAISS
    # --------------------------------------------------

    print("Loading FAISS...")

    vector_store = VectorStore.load(FAISS_PATH)

    print(f"Vector Count: {vector_store.size}")

    print(f"Embedding  Dimension: {vector_store.faiss_store.dimension}")

    # --------------------------------------------------
    # 4. 创建 Retriever
    # --------------------------------------------------

    retriever = VectorRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # --------------------------------------------------
    # 5. 创建 Evaluator
    # --------------------------------------------------

    evaluator = Evaluator(
        retriever=retriever
    )

    # --------------------------------------------------
    # 6. 执行评测
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Start Document Level Evaluation")
    print("=" * 60)

    document_results = evaluator.evaluate(
        document_dataset,
        top_k=10,
    )

    chunk_results = evaluator.evaluate(chunk_dataset, top_k=10)

    # --------------------------------------------------
    # 7. 计算整体指标
    # --------------------------------------------------

    document_overall = calculate_average(document_results, "document_metrics")
    chunk_overall = calculate_average(chunk_results, "chunk_metrics")

    # --------------------------------------------------
    # 8. 按 difficulty 分组
    # --------------------------------------------------

    document_by_difficulty = aggregate_by_field(
        document_results,
        "document_metrics",
        "difficulty",
    )

    chunk_by_difficulty = aggregate_by_field(
        chunk_results,
        "chunk_metrics",
        "difficulty",
    )

    # --------------------------------------------------
    # 9. 按 category 分组
    # --------------------------------------------------

    document_by_category = aggregate_by_field(
        document_results,
        "document_metrics",
        "category",
    )

    chunk_by_category = aggregate_by_field(
        chunk_results,
        "chunk_metrics",
        "category",
    )

    # --------------------------------------------------
    # 10. 打印整体结果
    # --------------------------------------------------

    print_metrics(
        "Document Level Overall",
        document_overall,
    )

    print_metrics(
        "Chunk Level Overall",
        chunk_overall,
    )

    # --------------------------------------------------
    # 11. 打印 difficulty
    # --------------------------------------------------

    for difficulty, metrics in document_by_difficulty.items():
        print_metrics(
            f"Document Level Difficulty: {difficulty}",
            metrics,
        )

    for difficulty, metrics in chunk_by_difficulty.items():
        print_metrics(
            f"Chunk Level Difficulty: {difficulty}",
            metrics,
        )

    # --------------------------------------------------
    # 12. 打印 category
    # --------------------------------------------------

    for category, metrics in document_by_category.items():
        print_metrics(
            f"Document Level Category: {category}",
            metrics,
        )

    for category, metrics in chunk_by_category.items():
        print_metrics(
            f"chunk Level Category: {category}",
            metrics,
        )

    # --------------------------------------------------
    # 13. 保存结果
    # --------------------------------------------------

    save_results(
        DOCUMENT_RESULT_PATH,
        document_overall,
        document_by_difficulty,
        document_by_category,
        document_results,
    )

    save_results(
        CHUNK_RESULT_PATH,
        chunk_overall,
        chunk_by_difficulty,
        chunk_by_category,
        chunk_results,
    )

    print()
    print("=" * 60)
    print("Evaluation Completed")
    print("=" * 60)

    print(
        f"Result saved to: {DOCUMENT_RESULT_PATH} and {CHUNK_RESULT_PATH}"
    )

if __name__ == "__main__":
    main()

