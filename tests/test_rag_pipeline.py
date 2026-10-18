
from unittest.mock import MagicMock

import pytest

from src.rag.rag_pipeline import RAGPipeline


def test_query():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {"chunk": object(), "score": 0.95}
    ]

    context_builder = MagicMock()
    context_builder.build.return_value = {
        "context": "[1]\n标准名称：测试标准\n测试内容",
        "sources": [
            {
                "ref": 1,
                "chunk_id": "chunk-001",
                "document": "测试标准",
                "chapter": "第一章",
                "title": "总则",
            }
        ],
        "chunk_count": 1,
        "context_chars": 35,
    }

    generator = MagicMock()
    generator.generate.return_value = "根据测试标准，结论如下。"

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
    )

    result = pipeline.query("测试问题")

    retriever.retrieve.assert_called_once_with(
        query="测试问题",
        top_k=10,
        fusion_k=10,
        candidate_k=25,
    )

    context_builder.build.assert_called_once_with(
        retriever.retrieve.return_value
    )

    generator.generate.assert_called_once_with(
        question="测试问题",
        context="[1]\n标准名称：测试标准\n测试内容",
    )

    assert result["answer"] == "根据测试标准，结论如下。"
    assert result["sources"][0]["chunk_id"] == "chunk-001"
    assert result["retrieval"]["chunk_count"] == 1


def test_query_rejects_empty_question():
    pipeline = RAGPipeline(
        retriever=MagicMock(),
        context_builder=MagicMock(),
        generator=MagicMock(),
    )

    with pytest.raises(ValueError, match="question"):
        pipeline.query("   ")


def test_query_stream():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {"chunk": object(), "score": 0.95}
    ]

    context_builder = MagicMock()
    context_builder.build.return_value = {
        "context": "[1]\n标准名称：测试标准\n测试内容",
        "sources": [
            {
                "ref": 1,
                "chunk_id": "chunk-001",
                "document": "测试标准",
                "chapter": "第一章",
                "title": "总则",
            }
        ],
        "chunk_count": 1,
        "context_chars": 35,
    }

    generator = MagicMock()
    generator.generate_stream.return_value = iter([
        "动力电池",
        "健康状态",
        "评估方法。",
    ])

    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
    )

    events = list(pipeline.query_stream("如何评估动力电池健康状态？"))

    # 第一个事件应当是来源
    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["chunk_id"] == "chunk-001"

    # 中间事件是模型输出的文本片段
    token_events = [
        event for event in events
        if event["type"] == "token"
    ]
    assert [event["content"] for event in token_events] == [
        "动力电池",
        "健康状态",
        "评估方法。",
    ]

    # 最后一个事件包含完整回答
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "动力电池健康状态评估方法。"

    generator.generate_stream.assert_called_once_with(
        question="如何评估动力电池健康状态？",
        context="[1]\n标准名称：测试标准\n测试内容",
    )

