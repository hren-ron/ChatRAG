

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.rag.generator import Generator


def make_chunk(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content)
            )
        ]
    )


def test_generate_stream():
    client = MagicMock()
    client.chat.completions.create.return_value = [
        make_chunk("动力电池"),
        make_chunk("健康状态"),
        make_chunk("可以通过相关指标评估。"),
    ]

    generator = Generator(
        model="qwen3.8-27b",
        client=client,
    )

    result = list(
        generator.generate_stream(
            question="如何评估动力电池健康状态？",
            context="[1] 标准内容：健康状态评估指标……",
        )
    )

    assert result == [
        "动力电池",
        "健康状态",
        "可以通过相关指标评估。",
    ]

    call_kwargs = (
        client.chat.completions.create.call_args.kwargs
    )
    assert call_kwargs["stream"] is True
    assert call_kwargs["model"] == "qwen3.8-27b"


def test_generate():
    client = MagicMock()
    client.chat.completions.create.return_value = [
        make_chunk("动力电池"),
        make_chunk("健康状态"),
        make_chunk("评估方法。"),
    ]

    generator = Generator(
        model="qwen3.8-27b",
        client=client,
    )

    answer = generator.generate(
        question="如何评估动力电池健康状态？",
        context="[1] 标准内容：健康状态评估指标……",
    )

    assert answer == "动力电池健康状态评估方法。"


def test_generate_stream_skips_empty_content():
    client = MagicMock()
    client.chat.completions.create.return_value = [
        make_chunk(None),
        make_chunk("有效内容"),
        make_chunk(""),
    ]

    generator = Generator(client=client)

    result = list(
        generator.generate_stream(
            question="测试问题",
            context="测试上下文",
        )
    )

    assert result == ["有效内容"]
