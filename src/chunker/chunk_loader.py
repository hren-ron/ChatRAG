import hashlib
import re

from pathlib import Path

from src.chunker.models import Chunk


class ChunkLoader:
    """
        加载 ChapterSplitter 生成的 TXT Chunk。

        输入结构：

            data/output/
            ├── GB+1589-2026/
            │   ├── 001_1_范围.txt
            │   ├── 002_2_规范性引用文件.txt
            │   ├── 003_3_术语和定义.txt
            │   └── ...
            │
            └── GB+7258-2017/
                ├── ...
                └── ...

        输出：

            List[Chunk]
    """

    def __init__(self, input_dir: str):

        self.input_dir = Path(input_dir)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"chunk dir not exists: {self.input_dir}")

        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"not a dir: {self.input_dir}")


    def load(self):
        """
        加载所有chunks
        :return:
        """

        chunks = []

        document_dirs = sorted(path for path in self.input_dir.iterdir() if path.is_dir())

        for document_dir in document_dirs:

            document_chunks = self._load_document(document_dir)
            chunks.extend(document_chunks)
        return chunks

    def _load_document(self, document_dir: Path):
        """
        加载单个文档下的所有 Chunk。
        """

        document = document_dir.name

        # 文档ID
        document_id = self._build_document_id(document)

        # 从文档中解析出版本
        version = self._parse_version(document)

        chunks = []

        txt_files = sorted(document_dir.glob("*.txt"))

        for txt_file in txt_files:
            chunk = self._load_file(document, document_id=document_id, version=version, txt_file=txt_file)

            if chunk is not None:
                chunks.append(chunk)
        return chunks


    def _load_file(self, document: str, document_id: str, version: str, txt_file: Path):
        """
        加载一个txt文件
        :param document:
        :param document_id:
        :param version:
        :param txt_file:
        :return:
        """

        content = txt_file.read_text(encoding="utf-8").strip()

        if not content:
            return None

        # 从文件名解析章节
        chapter, title, level = self._parse_filename(txt_file.name)

        # 当前版本的 Chunk ID
        chunk_id = self._build_chunk_id(document_id=document_id, filename=txt_file.name)

        # 内容hash
        content_hash = self._build_content_hash(content)

        # 当前阶段没有真正构建 parent_path。
        #
        # 因为 ChapterSplitter 已经保证了：
        #   level 1 / level 2 -> 一个 TXT
        #
        # 这里先保存当前章节。
        #
        # 后续如果需要完整：
        #   4 技术要求
        #   4.1 制动系统
        #   4.1.1 制动性能
        #
        # 可以在 SecondaryChunker 阶段构建。

        parent_path = [title]

        return Chunk(
            chunk_id=chunk_id,

            document_id=document_id,
            document=document,
            source=f"{document}.pdf",
            version=version,

            chapter=chapter,
            title=title,
            level=level,
            parent_path=" > ".join(parent_path),

            content_hash=content_hash,
            content=content,

            is_deleted=False
        )

    @staticmethod
    def _build_chunk_id(document_id: str, filename: str):
        """
        构建稳定 Chunk ID。

        例如：

            gb_1589_2026_004_3.1_汽车
        """

        stem = Path(filename).stem

        match = re.match(
            r"^(\d+)_(.+?)_(.+)$",
            stem
        )

        if match:
            chunk_index = match.group(1)
            chapter = match.group(2)
            return f"{document_id}_{chunk_index}_{chapter}"

        return f"{document_id}_{stem}"


    @staticmethod
    def _build_document_id(document: str):
        """
        将文档名称转换成稳定的 document_id。

        例如：

            GB+1589-2026
                ↓
            gb_1589_2026
        """

        document_id = document.lower()

        document_id = re.sub(r"[^a-z0-9]+", "_", document_id)
        return document_id.strip("_")

    @staticmethod
    def _parse_version(document: str):
        """
        从文档名称中提取版本。

        例如：

            GB+1589-2026
                ↓
            2026

            GB+7258-2017
                ↓
            2017
        """

        match = re.search(r"(\d{4})$", document)

        if match:
            return match.group()
        return ""


    @staticmethod
    def _parse_filename(filename: str):

        """
        从文件名解析：

            003_3_术语和定义.txt

        得到：

            chapter = 3
            title   = 术语和定义
            level   = 1

        例如：

            004_3.1_汽车.txt

        得到：

            chapter = 3.1
            title   = 汽车
            level   = 2
        """

        stem = Path(filename).stem

        match = re.match(r"^\d+_(.+?)_(.+)$", stem)

        if not match:
            return "", stem, 0

        chapter = match.group(1)
        title = match.group(2)

        title = ChunkLoader._clean_title(title)

        level = ChunkLoader._detect_level(chapter)

        return chapter, title, level

    @staticmethod
    def _detect_level(chapter: str):
        """
       判断章节级别。

       1       -> level 1
       1.1     -> level 2
       1.1.1   -> level 3
       1.1.1.1 -> level 4
       附录A    -> level 1
       """

        if chapter.startswith("附录"):
            return 1

        if re.fullmatch(r'\d+', chapter):
            return 1

        if re.fullmatch("r\d+\.\d+", chapter):
            return 2

            # 1.1.1
        if re.fullmatch(r"\d+\.\d+\.\d+",chapter):
            return 3

        # 1.1.1.1
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", chapter):
            return 4

        return 0

    @staticmethod
    def _clean_title(title: str) -> str:
        """
        清理标题中的排版符号。

        例如：

            一般要求············
            一般要求............

        转换为：

            一般要求
        """

        # 去除首尾空白
        title = title.strip()

        # 去除连续的 ·
        title = re.sub(
            r"[·•]+$",
            "",
            title
        )

        # 去除连续的 .
        title = re.sub(
            r"\.+$",
            "",
            title
        )

        # 再次去除空白
        title = title.strip()

        return title

    @staticmethod
    def _build_content_hash(content: str):
        """
        使用 SHA256 计算内容 Hash。

        用于后续判断：
        Chunk 内容是否发生变化。
        """

        return hashlib.sha256(content.encode("utf-8")).hexdigest()