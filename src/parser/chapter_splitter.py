import re
from pathlib import Path
from typing import List, Tuple


class ChapterSplitter:
    """
    按最多二级标题进行分块。

    一级标题：
        1 范围
        2 规范性引用文件
        3 术语和定义

    二级标题：
        3.1 汽车
        3.2 挂车

    三级标题：
        3.1.1 乘用车

    四级标题：
        3.1.1.1 小型乘用车

    三级、四级标题不会继续切块。
    """

    # ========================================================
    # 一级标题
    #
    # 1 范围
    # 2 规范性引用文件
    #
    # 注意：
    # 不能匹配 3.1
    # ========================================================

    LEVEL_1_RE = re.compile(
        r"^(\d+)\s+(.+)$"
    )

    LEVEL_1_NO_SPACE_RE = re.compile(
        r"^(\d+)([\u4e00-\u9fff].*)$"
    )

    # ========================================================
    # 二级标题
    #
    # 3.1 汽车
    # 3.2 挂车
    #
    # 不匹配：
    # 3
    # 3.1.1
    # 3.1.1.1
    # ========================================================

    LEVEL_2_RE = re.compile(
        r"^(\d+\.\d+)\s+(.+)$"
    )

    LEVEL_2_NO_SPACE_RE = re.compile(
        r"^(\d+\.\d+)([\u4e00-\u9fff].*)$"
    )

    # ========================================================
    # 附录
    #
    # 附录 A
    # 附录 B（规范性）
    # ========================================================

    APPENDIX_RE = re.compile(
        r"^附录\s+([A-Z])(?:\s*[（(](.*?)[）)])?.*$"
    )

    def __init__(
        self,
        text: str,
        output_dir: str = "chunks"
    ):
        self.text = text
        self.output_dir = Path(output_dir)

    # ========================================================
    # 主函数
    # ========================================================

    def split(self):

        lines = self.text.splitlines()

        chunks = self.detect_chunks(lines)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for index, chunk in enumerate(chunks, start=1):

            self.save_chunk(
                index,
                chunk
            )

        print(
            f"共生成 {len(chunks)} 个分块"
        )

    # ========================================================
    # 检测分块
    # ========================================================

    def detect_chunks(
        self,
        lines: List[str]
    ) -> List[Tuple[str, List[str]]]:

        chunks = []

        current_title = None
        current_content = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # 判断是不是需要切块的标题
            # ------------------------------------------------

            title_type = self.detect_title_type(
                line
            )

            # ------------------------------------------------
            # 一级 / 二级标题
            # ------------------------------------------------

            if title_type in (
                "level1",
                "level2",
                "appendix"
            ):

                # 保存之前的 chunk
                if current_title is not None:

                    chunks.append(
                        (
                            current_title,
                            current_content
                        )
                    )

                # 开始新 chunk
                current_title = line

                current_content = [
                    line
                ]

            # ------------------------------------------------
            # 普通正文
            #
            # 包括：
            #
            # 3.1.1 xxx
            # 3.1.1.1 xxx
            #
            # 它们不会触发切块
            # ------------------------------------------------

            else:

                if current_title is not None:

                    current_content.append(
                        line
                    )

        # ----------------------------------------------------
        # 保存最后一个 chunk
        # ----------------------------------------------------

        if current_title is not None:

            chunks.append(
                (
                    current_title,
                    current_content
                )
            )

        return chunks

    # ========================================================
    # 判断标题类型
    # ========================================================

    def detect_title_type(
        self,
        line: str
    ) -> str | None:

        # ----------------------------------------------------
        # 1. 一级标题
        # ----------------------------------------------------

        if self.is_level1(line):

            return "level1"

        # ----------------------------------------------------
        # 2. 二级标题
        # ----------------------------------------------------

        if self.is_level2(line):

            return "level2"

        # ----------------------------------------------------
        # 3. 附录
        # ----------------------------------------------------

        if self.APPENDIX_RE.match(line):

            return "appendix"

        return None

    # ========================================================
    # 判断一级标题
    # ========================================================

    def is_level1(
        self,
        line: str
    ) -> bool:

        # ----------------------------------------------------
        # 1 范围
        # ----------------------------------------------------

        match = self.LEVEL_1_RE.match(
            line
        )

        if match:

            number = match.group(1)

            return self.valid_section_number(
                number
            )

        # ----------------------------------------------------
        # 1范围
        # ----------------------------------------------------

        match = self.LEVEL_1_NO_SPACE_RE.match(
            line
        )

        if match:

            number = match.group(1)

            return self.valid_section_number(
                number
            )

        return False

    # ========================================================
    # 判断二级标题
    # ========================================================

    def is_level2(
        self,
        line: str
    ) -> bool:

        # ----------------------------------------------------
        # 3.1 汽车
        # ----------------------------------------------------

        match = self.LEVEL_2_RE.match(
            line
        )

        if match:

            return True

        # ----------------------------------------------------
        # 3.1汽车
        # ----------------------------------------------------

        match = self.LEVEL_2_NO_SPACE_RE.match(
            line
        )

        if match:

            return True

        return False

    # ========================================================
    # 一级章节编号是否合法
    # ========================================================

    def valid_section_number(
        self,
        number: str
    ) -> bool:

        try:
            value = int(number)
        except ValueError:
            return False

        # 防止正文中的超长数字被识别为章节
        return 1 <= value <= 99

    # ========================================================
    # 保存 chunk
    # ========================================================

    def save_chunk(
        self,
        index: int,
        chunk: Tuple[str, List[str]]
    ):

        title, lines = chunk

        filename = self.make_filename(
            index,
            title
        )

        output_path = (
            self.output_dir / filename
        )

        content = "\n".join(
            lines
        )

        output_path.write_text(
            content,
            encoding="utf-8"
        )

        print(
            f"[{index:03d}] {title}"
        )

    # ========================================================
    # 生成文件名
    # ========================================================

    def make_filename(
        self,
        index: int,
        title: str
    ) -> str:

        # ----------------------------------------------------
        # 一级标题
        # ----------------------------------------------------

        match = self.LEVEL_1_RE.match(
            title
        )

        if match:

            number = match.group(1)
            name = match.group(2)

            name = self.clean_filename(
                name
            )

            return (
                f"{index:03d}_{number}_{name}.txt"
            )

        # ----------------------------------------------------
        # 二级标题
        # ----------------------------------------------------

        match = self.LEVEL_2_RE.match(
            title
        )

        if match:

            number = match.group(1)
            name = match.group(2)

            name = self.clean_filename(
                name
            )

            return (
                f"{index:03d}_{number}_{name}.txt"
            )

        # ----------------------------------------------------
        # 附录
        # ----------------------------------------------------

        match = self.APPENDIX_RE.match(
            title
        )

        if match:

            letter = match.group(1)

            name = self.clean_filename(
                title
            )

            return (
                f"{index:03d}_附录_{letter}_{name}.txt"
            )

        return (
            f"{index:03d}_unknown.txt"
        )

    # ========================================================
    # 清理文件名
    # ========================================================

    def clean_filename(
        self,
        name: str
    ) -> str:

        name = re.sub(
            r'[\\/:*?"<>|]',
            "_",
            name
        )

        name = name.strip()

        if len(name) > 80:

            name = name[:80]

        return name


# ============================================================
# 使用
# ============================================================

# if __name__ == "__main__":
#
#     splitter = ChapterSplitter(
#         text_path="GB1589-2026.txt",
#         output_dir="chunks"
#     )
#
#     splitter.split()