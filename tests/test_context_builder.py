
from types import SimpleNamespace

from src.rag.context_build import ContextBuilder


def make_result(
    chunk_id: str,
    content: str,
    score: float = 0.9,
):
    chunk = SimpleNamespace(
        chunk_id=chunk_id,
        document="GB/T TEST-2026",
        chapter="第 4 章",
        title="健康状态评估",
        content=content,
    )

    return {
        "chunk": chunk,
        "score": score,
        "fusion_score": 0.03,
    }


def test_build_context():
    builder = ContextBuilder(
        max_chunks=2,
        max_chars=12000,
    )

    results = [
        make_result("chunk_001", "第一条标准内容"),
        make_result("chunk_002", "第二条标准内容"),
        make_result("chunk_003", "第三条标准内容"),
    ]

    output = builder.build(results)

    assert output["chunk_count"] == 2
    assert len(output["sources"]) == 2
    assert "第一条标准内容" in output["context"]
    assert "第二条标准内容" in output["context"]
    assert "第三条标准内容" not in output["context"]


def test_duplicate_chunks_are_removed():
    builder = ContextBuilder()

    results = [
        make_result("chunk_001", "标准正文"),
        make_result("chunk_001", "重复正文"),
    ]

    output = builder.build(results)

    assert output["chunk_count"] == 1
    assert len(output["sources"]) == 1


def test_empty_results():
    builder = ContextBuilder()

    output = builder.build([])

    assert output["context"] == ""
    assert output["sources"] == []
    assert output["chunk_count"] == 0
