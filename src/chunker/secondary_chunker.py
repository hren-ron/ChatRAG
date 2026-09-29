"""
Secondary Chunker

负责：

    Structured Chunk
        ↓
    Retrieval Chunk

职责：
    1. 保留原始 Chunk 的文档和章节信息
    2. 根据标题层级维护上下文
    3. 按段落进行二次切分
    4. 超长段落按照句子继续切分
    5. 控制 Chunk 最大 Token 数
    6. 为 Retrieval Chunk 生成新的 chunk_id
    7. 计算 token_count
    8. 计算新的 content_hash
    9. 构建 parent_path

不负责：
    - PDF 解析
    - TXT 加载
    - Embedding
    - Vector DB
    - BM25
    - Reranker
"""
import hashlib
import re
from typing import List

from src.chunker.models import Chunk


class SecondaryChunker:
    """
    将 Structured Chunk 转换成 Retrieval Chunk。

    示例：

        Structured Chunk:

            4 技术要求
            4.1 制动系统
            4.1.1 制动性能

            车辆的制动性能应满足……
            制动系统应具有……

        ↓

        Retrieval Chunk:

            4 技术要求
            4.1 制动系统
            4.1.1 制动性能

            车辆的制动性能应满足……
            制动系统应具有……
    """

    STRATEGY = "hierarchical_recursive_v1"

    def __init__(self, tokenizer, max_tokens: int=800, min_tokens: int=100, overlap_tokens: int=80):
        """
        :param tokenizer:
            HuggingFace Tokenizer。

        :param max_tokens:
            一个 Retrieval Chunk 最大 Token 数。

        :param min_tokens:
            一个 Chunk 尽量不要小于该 Token 数。
            注意：这是软限制，不会为了满足最小长度强行拼接不同章节。

        :param overlap_tokens:
            相邻 Chunk 的正文 overlap Token 数。
        """

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        if min_tokens < 0:
            raise ValueError("min_tokens must be greater than or equal to 0")

        if min_tokens > max_tokens:
            raise ValueError("min_tokens cannot be greater than max_tokens")

        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be greater than or equal to 0")

        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

    # ============================================================
    # Public API
    # ============================================================

    def split(self, chunk: Chunk) -> List[Chunk]:

        """
        将一个 Structured Chunk 切分成多个 Retrieval Chunk。

        :param chunk:
            ChunkLoader 生成的 Structured Chunk。

        :return:
            List[Chunk]
        """

        if not chunk.content.strip():
            return []

        lines = self._normalize_lines(chunk.content)

        if not lines:
            return []

        # --------------------------------------------------------
        # 第一步：
        # 根据标题构建结构化段落
        # --------------------------------------------------------
        sections = self._build_sections(lines)

        # --------------------------------------------------------
        # 第二步：
        # 将 section 转换成 Retrieval Chunk
        # --------------------------------------------------------

        retrieval_texts = []

        for section in sections:

            heading_context = section["heading_context"]
            body = section["body"]

            if not body:
                continue

            pieces = self._split_body(body=body, heading_context=heading_context)

            for piece in pieces:
                retrieval_texts.append(
                    {
                        "heading_context": heading_context,
                        "body": piece
                    }
                )
        # --------------------------------------------------------
        # 第三步：
        # 生成新的 Chunk
        # --------------------------------------------------------

        result = []

        for index, item in enumerate(retrieval_texts, start=1):

            content = self._build_content(heading_context=item["heading_context"], body=item["body"])

            token_count = self._count_tokens(content)

            if token_count <= self.max_tokens:
                new_chunk = self._create_chunk(source_chunk=chunk, heading_context=item["heading_context"], body=item["body"], index=index)

                result.append(new_chunk)
            else:
                # 最终保险：
                # 如果因为 overlap / decode 等原因超过 max_tokens，
                # 再进行一次 token 级切分。
                heading_content = self._build_content(
                    heading_context=item["heading_context"],
                    body="",
                )

                heading_tokens = self._count_tokens(heading_content)

                body_max_tokens = self.max_tokens - heading_tokens

                if body_max_tokens <= 0:
                    raise ValueError(
                        f"heading context exceeds max_tokens: "
                        f"{heading_tokens} > {self.max_tokens}"
                    )

                pieces = self._split_by_tokens(content, max_tokens=body_max_tokens)

                for piece_index, piece in enumerate(pieces,start=1):
                    new_chunk = self._create_chunk_from_content(
                        source_chunk=chunk,
                        content=piece,
                        index=f"{index:03d}_{piece_index:02d}",
                    )

                    result.append(new_chunk)

        return result

    # ============================================================
    # Section Building
    # ============================================================
    def _build_sections(self, lines: List[str]):

        """
        根据标题层级，将文本组织成 Section。

        不进行真正的 Chunk 切分。

        例如：

            4 技术要求
            4.1 制动系统
            4.1.1 制动性能
            车辆……

        会形成：

            {
                heading_context: [
                    "4 技术要求",
                    "4.1 制动系统",
                    "4.1.1 制动性能"
                ],
                body: [
                    "车辆……"
                ]
            }
        """
        sections = []
        heading_stack = {}

        current_body = []
        current_heading_context = []

        def flush_body():

            if not current_body:
                return

            sections.append(
                {
                    "heading_context": list(current_heading_context),
                    "body": list(current_body)
                }
            )

        for line in lines:

            level = self._detect_heading_level(line)

            # ----------------------------------------------------
            # 普通正文
            # ----------------------------------------------------
            if level == 0:
                current_body.append(line)
                continue

            # ----------------------------------------------------
            # 遇到新标题
            # ----------------------------------------------------

            flush_body()

            current_body.clear()

            # ----------------------------------------------------
            # 删除当前级别及以下的旧标题
            # ----------------------------------------------------
            for old_level in list(heading_stack.keys()):

                if old_level >= level:
                    del heading_stack[old_level]

            # ----------------------------------------------------
            # 保存当前标题
            # ----------------------------------------------------
            heading_stack[level] = line

            # ----------------------------------------------------
            # 构建当前完整标题路径
            # ----------------------------------------------------
            current_heading_context = [
                heading_stack[key]
                for key in sorted(
                    heading_stack.keys()
                )
            ]
        # --------------------------------------------------------
        # 最后一个正文
        # --------------------------------------------------------

        flush_body()

        return sections

    # ============================================================
    # Text Processing
    # ============================================================
    @staticmethod
    def _normalize_lines(text: str) -> List[str]:
        """
        清理文本行。

        不进行复杂文本清洗。
        PDF 清洗应该已经在 PDF Parser 阶段完成。
        """

        lines = []

        for line in text.splitlines():
            line = line.strip()

            # 保留空行，用于后续识别段落
            if not line:
                lines.append("")
                continue
            lines.append(line)
        return lines

    # ============================================================
    # Heading Processing
    # ============================================================

    @staticmethod
    def _detect_heading_level(text: str) -> int:
        """
        判断一行是否为标题。

        返回：

            0 -> 普通正文
            1 -> 一级标题
            2 -> 二级标题
            3 -> 三级标题
            4 -> 四级标题
        """

        text = text.strip()

        # --------------------------------------------------------
        # 4.1.1.1 xxx
        # --------------------------------------------------------

        if re.match(
                r"^\d+\.\d+\.\d+\.\d+\s+.+$",
                text,
        ):
            return 4

        # --------------------------------------------------------
        # 4.1.1 xxx
        # --------------------------------------------------------

        if re.match(
                r"^\d+\.\d+\.\d+\s+.+$",
                text,
        ):
            return 3

        # --------------------------------------------------------
        # 4.1 xxx
        # --------------------------------------------------------

        if re.match(
                r"^\d+\.\d+\s+.+$",
                text,
        ):
            return 2

        # --------------------------------------------------------
        # 4 xxx
        # --------------------------------------------------------

        if re.match(
                r"^\d+\s+.+$",
                text,
        ):
            return 1

        # --------------------------------------------------------
        # 附录 A
        # 附录 A（规范性）
        # --------------------------------------------------------

        if re.match(
                r"^附录\s*[A-Z](?:\s*[（(].*[）)])?$",
                text,
        ):
            return 1

        return 0

    # ============================================================
    # Body Splitting
    # ============================================================

    def _split_body(
            self,
            body: List[str],
            heading_context: List[str],
    ) -> List[str]:

        """
        对正文进行二次切分。

        优先级：

            段落
              ↓
            句子
              ↓
            Token
        """

        # 计算标题上下文占用的 token
        heading_content = self._build_content(
            heading_context=heading_context,
            body="",
        )

        heading_tokens = self._count_tokens(heading_content)

        # 正文真正允许使用的 token 数
        body_max_tokens = self.max_tokens - heading_tokens

        if body_max_tokens <= 0:
            raise ValueError(
                f"heading context already exceeds max_tokens: "
                f"{heading_tokens} > {self.max_tokens}"
            )

        paragraphs = self._build_paragraphs(body)

        chunks = []

        current_sentences = []
        current_tokens = 0

        for paragraph in paragraphs:

            paragraph_tokens = self._count_tokens(paragraph)

            # ----------------------------------------------------
            # 当前段落本身已经超过最大 Token
            # ----------------------------------------------------
            if paragraph_tokens > body_max_tokens:

                # 先保存当前 Chunk
                if current_sentences:
                    chunks.append("\n".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0

                # 对超长段落继续按照句子切分
                long_chunks = self._split_long_text(paragraph, max_tokens=body_max_tokens)
                chunks.extend(long_chunks)
                continue

            # ----------------------------------------------------
            # 当前 Chunk 可以继续放入这个段落
            # ----------------------------------------------------

            if current_tokens + paragraph_tokens <= body_max_tokens:
                current_sentences.append(paragraph)

                current_tokens += paragraph_tokens
                continue

            # ----------------------------------------------------
            # 放不下了
            # ----------------------------------------------------
            if current_sentences:

                chunks.append("\n".join(current_sentences))

            # ----------------------------------------------------
            # Overlap
            # ----------------------------------------------------
            overlap = self._build_overlap(current_sentences)

            current_sentences = overlap
            current_tokens = self._count_tokens("\n".join(current_sentences))

            # ----------------------------------------------------
            # 加入当前段落
            # ----------------------------------------------------

            if current_tokens + paragraph_tokens <= self.max_tokens:
                current_sentences.append(paragraph)
                current_tokens += paragraph_tokens
            else:
                chunks.append(paragraph)

                current_tokens = 0
                current_sentences = []

        if current_sentences:
            chunks.append("\n".join(current_sentences))
        return chunks

    # ============================================================
    # Long Text Splitting
    # ============================================================
    def _split_long_text(self, text: str, max_tokens: int):
        """
        一个段落超过 max_tokens 时：

            段落
              ↓
            句子
              ↓
            Token
        """

        sentences = self._split_sentences(text)

        chunks = []

        current = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # ----------------------------------------------------
            # 单句本身超过 max_tokens
            # ----------------------------------------------------
            if sentence_tokens > max_tokens:
                if current:
                    chunks.append("".join(current))
                    current = []
                    current_tokens = 0

                chunks.extend(self._split_by_tokens(sentence, max_tokens=max_tokens))
                continue

            # ----------------------------------------------------
            # 当前 Chunk 可以继续放
            # ----------------------------------------------------
            if current_tokens + sentence_tokens <= max_tokens:
                current.append(sentence)
                current_tokens += sentence_tokens
                continue

            # ----------------------------------------------------
            # 当前 Chunk 满了
            # ----------------------------------------------------
            if current:
                chunks.append("".join(current))

            overlap = self._build_overlap(current)

            current = overlap

            current_tokens = self._count_tokens("".join(current))

            if current_tokens + sentence_tokens <= max_tokens:
                current.append(sentence)
                current_tokens += sentence_tokens
            else:
                chunks.append(sentence)
                current = []
                current_tokens = 0

        if current:
            chunks.append("".join(current))
        return chunks




    # ============================================================
    # Token Splitting
    # ============================================================

    def _split_by_tokens(self, text: str, max_tokens: int):
        """
        一个句子本身超过 max_tokens 时，
        直接按照 Token 切分。
        """

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []

        start = 0
        while start < len(token_ids):
            end = min(start + max_tokens, len(token_ids))

            piece_ids = token_ids[start:end]

            piece = self.tokenizer.decode(piece_ids, skip_special_tokens=True).strip()

            if piece:
                chunks.append(piece)

            if end >= len(token_ids):
                break

            start = max(end - self.overlap_tokens, start + 1)
        return chunks

    # ============================================================
    # Overlap
    # ============================================================
    def _build_overlap(self, texts: List[str]):
        """
        从上一 Chunk 的尾部提取 overlap。

        注意：
        overlap 只作用于正文，不作用于标题。
        """

        if self.overlap_tokens <= 0 or not texts:
            return []

        result = []
        token_count = 0
        for text in reversed(texts):
            text_tokens = self._count_tokens(text)

            if token_count + text_tokens > self.overlap_tokens:
                break
            result.insert(0, text)
            token_count += text_tokens

        return result

    # ============================================================
    # Chunk Creation
    # ============================================================
    def _create_chunk(self, source_chunk: Chunk, heading_context: List[str], body: str, index: int):
        """
        根据 Structured Chunk 创建 Retrieval Chunk。
        :param source_chunk:
        :param heading_context:
        :param body:
        :param index:
        :return:
        """
        # --------------------------------------------------------
        # 构建最终 content
        # --------------------------------------------------------

        content_parts = []
        if heading_context:
            content_parts.extend(heading_context)

        if body.strip():
            content_parts.append(body.strip())

        content = "\n\n".join(content_parts)

        # --------------------------------------------------------
        # Hash
        # --------------------------------------------------------

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        chunk_id = (f"{source_chunk.chunk_id}_{index:03d}")

        # --------------------------------------------------------
        # Token Count
        # --------------------------------------------------------
        token_count = self._count_tokens(content)

        # --------------------------------------------------------
        # parent_path
        #
        # 不包含当前最后一级标题。
        #
        # 例如：
        #
        # 4 技术要求
        # 4.1 制动系统
        # 4.1.1 制动性能
        #
        # parent_path：
        #
        # 4 技术要求 > 4.1 制动系统
        # --------------------------------------------------------
        parent_path = self._build_parent_path(heading_context)

        # --------------------------------------------------------
        # chapter / title
        # --------------------------------------------------------

        chapter = source_chunk.chapter
        title = source_chunk.title
        level = source_chunk.level

        if heading_context:
            current_heading = (heading_context[-1])

            current_level = (self._detect_heading_level(current_heading))

            if current_level > 0:
                level = current_level

                number = (self._extract_heading_number(current_heading))

                if number:
                    chapter = number

                title = (self._extract_heading_title(current_heading))

        return Chunk(
            chunk_id=chunk_id,

            document_id=source_chunk.document_id,
            document=source_chunk.document,
            source=source_chunk.source,
            version=source_chunk.version,

            chapter=chapter,
            title=title,
            level=level,
            parent_path=parent_path,

            content=content,
            content_hash=content_hash,

            is_deleted=source_chunk.is_deleted,

            chunk_strategy=self.STRATEGY,
            chunk_index=index,
            token_count=token_count,

            # Embedding 阶段还没有开始
            embedding_model=None,
            embedding_model_version=None,
            embedding_dimension=None,

            page_start=source_chunk.page_start,
            page_end=source_chunk.page_end,
        )

    def _create_chunk_from_content(
            self,
            source_chunk: Chunk,
            content: str,
            index,
    ) -> Chunk:
        """
        根据已经完成最终切分的 content 创建 Retrieval Chunk。

        用于：
            候选 Chunk 在最终构造后仍然超过 max_tokens，
            再经过 token-level split 后，直接生成新的 Chunk。
        """

        content = content.strip()

        if not content:
            raise ValueError("content cannot be empty")

        # --------------------------------------------------------
        # Token 数量
        # --------------------------------------------------------

        token_count = self._count_tokens(content)

        # --------------------------------------------------------
        # 最终安全检查
        # --------------------------------------------------------

        if token_count > self.max_tokens:
            raise ValueError(
                f"content token count {token_count} "
                f"exceeds max_tokens {self.max_tokens}"
            )

        # --------------------------------------------------------
        # content hash
        # --------------------------------------------------------

        content_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        # --------------------------------------------------------
        # chunk_id
        # --------------------------------------------------------

        chunk_id = (
            f"{source_chunk.chunk_id}_{index}"
        )

        # --------------------------------------------------------
        # 创建 Chunk
        # --------------------------------------------------------

        return Chunk(
            chunk_id=chunk_id,

            document_id=source_chunk.document_id,
            document=source_chunk.document,
            source=source_chunk.source,
            version=source_chunk.version,

            chapter=source_chunk.chapter,
            title=source_chunk.title,
            level=source_chunk.level,
            parent_path=source_chunk.parent_path,

            content=content,
            content_hash=content_hash,

            is_deleted=source_chunk.is_deleted,

            chunk_strategy=self.STRATEGY,
            chunk_index=None,
            token_count=token_count,

            embedding_model=None,
            embedding_model_version=None,
            embedding_dimension=None,

            page_start=source_chunk.page_start,
            page_end=source_chunk.page_end,
        )
    @staticmethod
    def _split_sentences(text: str):
        """
        中文技术文档句子切分。

        支持：

            。
            ！
            ？
            ；
        """

        sentences = re.split(r"(?<=[。！？；])", text)

        return [sentence.strip() for sentence in sentences if sentence.strip()]



    # ============================================================
    # Token Count
    # ============================================================

    def _count_tokens(
            self,
            text: str,
    ) -> int:
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    @staticmethod
    def _build_paragraphs(lines: List[str]):
        """
        将连续文本行合并成段落。

        当前 PDF Parser 已经完成文本清洗，
        因此这里主要根据行结构合并。
        """

        paragraphs = []

        current = []
        for line in lines:

            line = line.strip()

            if not line:
                if current:
                    paragraphs.append("".join(current))
                    current = []
                continue

            current.append(line)

        if current:
            paragraphs.append("".join(current))
        return paragraphs

    @staticmethod
    def _build_parent_path(heading_context: List[str]):
        """
        构建父级标题路径。

        例如：

            [
                "4 技术要求",
                "4.1 制动系统",
                "4.1.1 制动性能"
            ]

        得到：

            "4 技术要求 > 4.1 制动系统"

        当前标题本身不放进去。
        """

        if len(heading_context) <= 1:
            return None

        return " > ".join(heading_context[:-1])

    @staticmethod
    def _extract_heading_number(text: str) -> str:
        """
        提取标题编号。

        例如：

            4 技术要求
                -> 4

            4.1 制动系统
                -> 4.1

            4.1.1 制动性能
                -> 4.1.1

            附录 A
                -> 附录 A
        """

        text = text.strip()

        match = re.match(
            r"^(\d+(?:\.\d+)*)\s+",
            text,
        )

        if match:
            return match.group(1)

        match = re.match(
            r"^(附录\s*[A-Z])",
            text,
        )

        if match:
            return match.group(1)

        return ""

    @staticmethod
    def _extract_heading_title(
            heading: str,
    ) -> str:
        """
        从：

            4.1 制动系统

        提取：

            制动系统
        """

        heading = heading.strip()

        match = re.match(
            r"^(?:\d+(?:\.\d+)*|附录\s*[A-Z])\s+(.+)$",
            heading,
        )

        if match:
            return match.group(1).strip()

        return heading

    @staticmethod
    def _build_content(
            heading_context: list[str],
            body,
    ) -> str:
        """
        构造最终 Retrieval Chunk 内容。
        """

        content_parts = []

        # --------------------------------------------------------
        # 标题上下文
        # --------------------------------------------------------

        for heading in heading_context:
            heading = heading.strip()

            if heading:
                content_parts.append(heading)

        # --------------------------------------------------------
        # 正文
        # --------------------------------------------------------

        if isinstance(body, list):
            body = "".join(line.strip() for line in body if line.strip())

        body = body.strip()

        if body:
            content_parts.append(body)

        # --------------------------------------------------------
        # 最终内容
        # --------------------------------------------------------

        return "\n\n".join(content_parts)
