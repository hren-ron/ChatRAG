import json
from collections import defaultdict
from pathlib import Path

from config import EVALUATION_PATH, BGE_M3_PATH, FAISS_PATH, FAISS_RESULT_PATH
from src.evaluator.evaluator import Evaluator
from src.models.bge_m3_embedder import BGEEmbedder
from src.retriever.vector_retriever import VectorRetriever
from src.vector_store.vector_store import VectorStore


def load_dataset(path: Path):

    if not path.exists():
        raise FileNotFoundError("test dataset not exists")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def aggregate_results(results):
    if not results:
        return {}

    metric_names = [
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "precision@5",
        "mrr",
    ]

    return {
        metric: sum(item["metrics"][metric] for item in results) / len(results)
        for metric in metric_names
    }

def aggregate_by_field(results, field):

    groups = defaultdict(list)

    for result in results:

        value = result.get(field)

        if value is None:
            continue

        groups[value].append(result)
    return {
        value: aggregate_results(items)
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

    dataset = load_dataset(EVALUATION_PATH)

    print(f"Evaluation Dataset: {len(dataset)}")

    # --------------------------------------------------
    # 2. 加载 BGE-M3
    # --------------------------------------------------

    print()
    print("Loading BGE-M3...")

    embedder = BGEEmbedder(
        model_path=BGE_M3_PATH
    )

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
    print("Start Evaluation")
    print("=" * 60)

    results = evaluator.evaluate(
        dataset,
        top_k=10,
    )

    # --------------------------------------------------
    # 7. 计算整体指标
    # --------------------------------------------------

    overall = aggregate_results(
        results
    )

    # --------------------------------------------------
    # 8. 按 difficulty 分组
    # --------------------------------------------------

    by_difficulty = aggregate_by_field(
        results,
        "difficulty",
    )

    # --------------------------------------------------
    # 9. 按 category 分组
    # --------------------------------------------------

    by_category = aggregate_by_field(
        results,
        "category",
    )

    # --------------------------------------------------
    # 10. 打印整体结果
    # --------------------------------------------------

    print_metrics(
        "Overall",
        overall,
    )

    # --------------------------------------------------
    # 11. 打印 difficulty
    # --------------------------------------------------

    for difficulty, metrics in by_difficulty.items():
        print_metrics(
            f"Difficulty: {difficulty}",
            metrics,
        )

    # --------------------------------------------------
    # 12. 打印 category
    # --------------------------------------------------

    for category, metrics in by_category.items():
        print_metrics(
            f"Category: {category}",
            metrics,
        )

    # --------------------------------------------------
    # 13. 保存结果
    # --------------------------------------------------

    save_results(
        FAISS_RESULT_PATH,
        overall,
        by_difficulty,
        by_category,
        results,
    )

    print()
    print("=" * 60)
    print("Evaluation Completed")
    print("=" * 60)

    print(
        f"Result saved to: {FAISS_RESULT_PATH}"
    )

if __name__ == "__main__":
    main()

