from typing import Any


class ContextBuilder:
    """将检索结果整理为 LLM 可使用的上下文。"""

    def __init__(
            self,
            max_chunks: int=5,
            max_chars: int=12000
    ):
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")

        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chunks = max_chunks
        self.max_chars = max_chars

    def build(self, results: list[dict[str, Any]]):
        """构建上下文，保留检索排序和来源信息。"""

        selected = []

        seen_chunk_ids = set()

        used_chars = 0

        for result in results:

            chunk = result.get("chunk")

            if chunk is None:
                continue

            chunk_id = chunk.chunk_id

            if chunk_id in seen_chunk_ids:
                continue

            content = (chunk.content or "").strip()
            if not content:
                continue

            document = getattr(chunk, "document", "") or "未知文档"
            chapter = getattr(chunk, "chapter", "") or "未提供章节"
            title = getattr(chunk, "title", "") or "未提供标题"

            # 暂不截断单个 Chunk，避免破坏条款语义。
            header = (
                f"[{len(selected) + 1}]\n"
                f"标准名称：{document}\n"
                f"章节：{chapter}\n"
                f"标题：{title}\n"
                f"Chunk ID：{chunk_id}\n"
            )

            block = f"{header}\n{content}"
            separator = "\n\n"

            extra_chars = len(block)
            if selected:
                extra_chars += len(separator)

            # 不允许超过字符预算
            if used_chars + extra_chars > self.max_chars:
                continue

            selected.append({
                "ref": len(selected) + 1,
                "chunk_id": chunk_id,
                "document": document,
                "chapter": chapter,
                "title": title,
                "content": content,
                "score": result.get("score"),
                "fusion_score": result.get("fusion_score"),
                "text": block
            })

            seen_chunk_ids.add(chunk_id)
            used_chars += extra_chars

            if len(selected) >= self.max_chunks:
                break

        context = "\n\n".join(item["text"] for item in selected)

        sources = [{
            "ref": item["ref"],
            "chunk_id": item["chunk_id"],
            "document": item["document"],
            "chapter": item["chapter"],
            "title": item["title"],
        } for item in selected]

        return {
            "context": context,
            "sources": sources,
            "chunk_count": len(selected),
            "context_chars": len(context)
        }

