import re
import fitz

from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ============================================================
# 1. 数据模型
# ============================================================

@dataclass
class LogicalLine:
    """
    PDF 中的一行逻辑文本。

    text : 文本
    page : 页码，从 1 开始
    x0/y0/x1/y1 : 文本区域坐标
    """
    text: str
    page: int

    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0


@dataclass
class Heading:
    """
    标题结构。

    kind:
        main    正文标题
        appendix 附录标题
        figure  图标题
        table   表标题

    number:
        例如：
        4
        4.1
        4.1.1
        A
        A.1
        图 B.1
        表 5.1

    title:
        标题名称

    level:
        1~4
    """
    kind: str
    number: str
    title: str
    level: int

    @property
    def full_text(self) -> str:
        if self.title:
            return f"{self.number} {self.title}"
        return self.number


@dataclass
class DocumentBlock:
    """
    最终用于 RAG 的逻辑文档块。
    """
    document: str

    page_start: int
    page_end: int

    chapter: Optional[str]

    section: Optional[str]

    heading_path: List[str]

    text: str


# ============================================================
# 2. 配置
# ============================================================

@dataclass
class ParserConfig:

    # 同一行判断的 y 坐标误差
    y_tolerance: float = 3.0

    # 如果后一行距离前一行太远，则不认为是同一标题
    max_merge_gap: float = 20.0

    # 页面顶部区域
    header_ratio: float = 0.05

    # 页面底部区域
    footer_ratio: float = 0.90

    # 最大标题层级
    max_heading_level: int = 4

    # 是否规范化空格
    normalize_space: bool = True

    # 是否启用页眉页脚清理
    remove_header_footer: bool = True


# ============================================================
# 3. PDF 提取器
# ============================================================

class PDFExtractor:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def open(self):
        return fitz.open(self.pdf_path)

    @staticmethod
    def extract_words(page):
        """
        返回：

        (
            x0,
            y0,
            x1,
            y1,
            word,
            block_no,
            line_no,
            word_no
        )
        """
        return page.get_text("words")


# ============================================================
# 4. Word -> LogicalLine
# ============================================================

class LineBuilder:

    def __init__(self, config: ParserConfig):
        self.config = config

    def build(self, words, page_number: int) -> List[LogicalLine]:

        if not words:
            return []

        # ----------------------------------------------------
        # 按 block / line 分组
        # ----------------------------------------------------

        grouped = {}

        for word in words:

            x0, y0, x1, y1, text, block_no, line_no, word_no = word

            key = (block_no, line_no)

            grouped.setdefault(key, []).append(word)

        lines = []

        # ----------------------------------------------------
        # 每个 PDF line 组装成 LogicalLine
        # ----------------------------------------------------

        for _, word_list in grouped.items():

            word_list.sort(key=lambda x: x[0])

            text = self._join_words(word_list)

            x0 = min(w[0] for w in word_list)
            y0 = min(w[1] for w in word_list)

            x1 = max(w[2] for w in word_list)
            y1 = max(w[3] for w in word_list)

            lines.append(
                LogicalLine(
                    text=text,
                    page=page_number,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                )
            )

        # ----------------------------------------------------
        # 按 y 坐标排序
        # ----------------------------------------------------

        lines.sort(
            key=lambda line: (
                round(line.y0 / self.config.y_tolerance),
                line.x0,
            )
        )

        return lines

    @staticmethod
    def _join_words(words) -> str:

        result = ""

        for word in words:

            text = word[4]

            if not result:
                result = text
                continue

            previous = result[-1]

            # ------------------------------------------------
            # 中文场景：
            #
            # 例如：
            # "车辆" + "宽度"
            #
            # 不需要加空格
            # ------------------------------------------------

            if LineBuilder._need_space(previous, text[0]):
                result += " "

            result += text

        return result

    @staticmethod
    def _need_space(left: str, right: str) -> bool:

        # 中文字符之间不加空格
        if "\u4e00" <= left <= "\u9fff":
            if "\u4e00" <= right <= "\u9fff":
                return False

        # 中文 + 标点
        if right in "，。；：、）】》〉":
            return False

        if left in "（【《〈":
            return False

        return True


# ============================================================
# 5. 标题解析器
# ============================================================

class HeadingParser:

    # --------------------------------------------------------
    # 正文标题
    #
    # 4
    # 4.1
    # 4.1.1
    # 4.1.1.1
    # --------------------------------------------------------

    MAIN_PATTERN = re.compile(
        r"^(\d+(?:\.\d+){0,3})\s*(.*)$"
    )

    # --------------------------------------------------------
    # 附录
    #
    # A
    # A.1
    # A.1.1
    # A.1.1.1
    # --------------------------------------------------------

    APPENDIX_PATTERN = re.compile(
        r"^([A-Z](?:\.\d+){0,3})\s*(.*)$"
    )

    # --------------------------------------------------------
    # 图
    #
    # 图 B.1
    # 图 B.1.1
    # 图 B.1.1.1
    # 图 B.1.1.1.1
    # --------------------------------------------------------

    FIGURE_PATTERN = re.compile(
        r"^(图\s*[A-Z](?:\.\d+){1,3})\s*(.*)$"
    )

    # --------------------------------------------------------
    # 表
    #
    # 表 5
    # 表 5.1
    # 表 5.1.1
    # 表 5.1.1.1
    # --------------------------------------------------------

    TABLE_PATTERN = re.compile(
        r"^(表\s*\d+(?:\.\d+){0,3})\s*(.*)$"
    )

    def __init__(self, config: ParserConfig):
        self.config = config

    def parse(self, text: str) -> Optional[Heading]:

        text = self.normalize(text)

        if not text:
            return None

        # ----------------------------------------------------
        # 图
        # ----------------------------------------------------

        match = self.FIGURE_PATTERN.match(text)

        if match:

            number = match.group(1)
            title = match.group(2).strip()

            level = self._calc_level(
                number.replace("图", "").strip()
            )

            if 1 <= level <= self.config.max_heading_level:
                return Heading(
                    kind="figure",
                    number=number,
                    title=title,
                    level=level,
                )

        # ----------------------------------------------------
        # 表
        # ----------------------------------------------------

        match = self.TABLE_PATTERN.match(text)

        if match:

            number = match.group(1)
            title = match.group(2).strip()

            level = self._calc_level(
                number.replace("表", "").strip()
            )

            if 1 <= level <= self.config.max_heading_level:
                return Heading(
                    kind="table",
                    number=number,
                    title=title,
                    level=level,
                )

        # ----------------------------------------------------
        # 正文
        # ----------------------------------------------------

        match = self.MAIN_PATTERN.match(text)

        if match:

            number = match.group(1)
            title = match.group(2).strip()

            level = number.count(".") + 1

            if level <= self.config.max_heading_level:

                # 避免：
                #
                # 4.4车辆通过性要求应符合GB1589的规定。
                #
                # 这种普通正文被全部识别成标题。
                #
                # 当前版本采用一个简单启发式：
                # 如果标题过长，更可能是正文。
                #
                if self._looks_like_heading(title):

                    return Heading(
                        kind="main",
                        number=number,
                        title=title,
                        level=level,
                    )

        # ----------------------------------------------------
        # 附录
        # ----------------------------------------------------

        match = self.APPENDIX_PATTERN.match(text)

        if match:

            number = match.group(1)
            title = match.group(2).strip()

            level = number.count(".") + 1

            if level <= self.config.max_heading_level:

                if self._looks_like_heading(title):

                    return Heading(
                        kind="appendix",
                        number=number,
                        title=title,
                        level=level,
                    )

        return None

    @staticmethod
    def normalize(text: str) -> str:

        text = text.strip()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text

    @staticmethod
    def _calc_level(number: str) -> int:

        return number.count(".") + 1

    @staticmethod
    def _looks_like_heading(title: str) -> bool:

        if not title:
            return True

        # ----------------------------------------------------
        # 太长一般不是标题
        #
        # 这个阈值只是初始值。
        # 后续建议加入字体、字号、粗体等信息。
        # ----------------------------------------------------

        if len(title) > 100:
            return False

        # ----------------------------------------------------
        # 明显句号结尾通常更像正文
        # ----------------------------------------------------

        if title.endswith(("。", "；")):
            return False

        return True


# ============================================================
# 6. 标题编号恢复
# ============================================================

class LineMerger:

    # --------------------------------------------------------
    # 例如：
    #
    # 4.7.2.最大允许总质量应不超过55000kg。
    # 4
    #
    # ->
    #
    # 4.7.2.4 最大允许总质量应不超过55000kg。
    # --------------------------------------------------------

    MAIN_INCOMPLETE = re.compile(
        r"^(\d+(?:\.\d+){0,2})\.(.*)$"
    )

    APPENDIX_INCOMPLETE = re.compile(
        r"^([A-Z](?:\.\d+){0,2})\.(.*)$"
    )

    FIGURE_INCOMPLETE = re.compile(
        r"^(图\s*[A-Z](?:\.\d+){0,2})\.(.*)$"
    )

    TABLE_INCOMPLETE = re.compile(
        r"^(表\s*\d+(?:\.\d+){0,2})\.(.*)$"
    )

    PURE_NUMBER = re.compile(
        r"^\d+$"
    )

    def __init__(self, config: ParserConfig):
        self.config = config

    def merge(self, lines: List[LogicalLine]) -> List[LogicalLine]:

        if not lines:
            return []

        result = []

        i = 0

        while i < len(lines):

            current = lines[i]

            # ------------------------------------------------
            # 当前行是否可能是一个不完整标题
            # ------------------------------------------------

            if self._is_incomplete_heading(current.text):

                if i + 1 < len(lines):

                    next_line = lines[i + 1]

                    if self._can_merge(current, next_line):

                        merged = self._complete_heading(
                            current,
                            next_line
                        )

                        if merged:

                            result.append(merged)

                            i += 2
                            continue

            result.append(current)

            i += 1

        return result

    def _is_incomplete_heading(self, text: str) -> bool:

        text = text.strip()

        patterns = [
            self.MAIN_INCOMPLETE,
            self.APPENDIX_INCOMPLETE,
            self.FIGURE_INCOMPLETE,
            self.TABLE_INCOMPLETE,
        ]

        for pattern in patterns:

            if pattern.match(text):
                return True

        return False

    def _can_merge(
        self,
        current: LogicalLine,
        next_line: LogicalLine
    ) -> bool:

        # ----------------------------------------------------
        # 必须是同一页
        # ----------------------------------------------------

        if current.page != next_line.page:
            return False

        # ----------------------------------------------------
        # 下一行必须是纯数字
        #
        # 例如：
        #
        # 4
        # 5
        # 6
        # ----------------------------------------------------

        number = next_line.text.strip()

        if not self.PURE_NUMBER.match(number):
            return False

        # ----------------------------------------------------
        # 最多只能补一位数字
        # ----------------------------------------------------

        if len(number) != 1:
            return False

        # ----------------------------------------------------
        # 垂直距离不能太大
        # ----------------------------------------------------

        gap = next_line.y0 - current.y1

        if gap < 0:
            return False

        if gap > self.config.max_merge_gap:
            return False

        return True

    def _complete_heading(
        self,
        current: LogicalLine,
        next_line: LogicalLine
    ) -> Optional[LogicalLine]:

        text = current.text.strip()
        suffix = next_line.text.strip()

        # ----------------------------------------------------
        # 4.7.2. + 4
        #
        # -> 4.7.2.4
        # ----------------------------------------------------

        match = self.MAIN_INCOMPLETE.match(text)

        if match:

            prefix = match.group(1)
            title = match.group(2).strip()

            level = prefix.count(".") + 2

            if level <= self.config.max_heading_level:

                new_text = (
                    f"{prefix}.{suffix} {title}"
                )

                return LogicalLine(
                    text=new_text,
                    page=current.page,
                    x0=current.x0,
                    y0=current.y0,
                    x1=max(current.x1, next_line.x1),
                    y1=next_line.y1,
                )

        # ----------------------------------------------------
        # A.4.2. + 2
        #
        # -> A.4.2.2
        # ----------------------------------------------------

        match = self.APPENDIX_INCOMPLETE.match(text)

        if match:

            prefix = match.group(1)
            title = match.group(2).strip()

            level = prefix.count(".") + 2

            if level <= self.config.max_heading_level:

                new_text = (
                    f"{prefix}.{suffix} {title}"
                )

                return LogicalLine(
                    text=new_text,
                    page=current.page,
                    x0=current.x0,
                    y0=current.y0,
                    x1=max(current.x1, next_line.x1),
                    y1=next_line.y1,
                )

        # ----------------------------------------------------
        # 图B. + 1
        #
        # -> 图 B.1
        # ----------------------------------------------------

        match = self.FIGURE_INCOMPLETE.match(text)

        if match:

            prefix = match.group(1)
            title = match.group(2).strip()

            level = prefix.replace("图", "").strip().count(".") + 2

            if level <= self.config.max_heading_level:

                new_text = (
                    f"{prefix}{suffix} {title}"
                )

                # 标准化 "图B.1"
                new_text = re.sub(
                    r"^图\s*",
                    "图 ",
                    new_text
                )

                return LogicalLine(
                    text=new_text,
                    page=current.page,
                    x0=current.x0,
                    y0=current.y0,
                    x1=max(current.x1, next_line.x1),
                    y1=next_line.y1,
                )

        # ----------------------------------------------------
        # 表5. + 1
        # ----------------------------------------------------

        match = self.TABLE_INCOMPLETE.match(text)

        if match:

            prefix = match.group(1)
            title = match.group(2).strip()

            level = prefix.replace("表", "").strip().count(".") + 2

            if level <= self.config.max_heading_level:

                new_text = (
                    f"{prefix}{suffix} {title}"
                )

                new_text = re.sub(
                    r"^表\s*",
                    "表 ",
                    new_text
                )

                return LogicalLine(
                    text=new_text,
                    page=current.page,
                    x0=current.x0,
                    y0=current.y0,
                    x1=max(current.x1, next_line.x1),
                    y1=next_line.y1,
                )

        return None


# ============================================================
# 7. 清理页眉、页脚、页码
# ============================================================

class LineCleaner:

    STANDARD_HEADER_PATTERN = re.compile(
        r"^(GB\s*/?\s*T?\s*\d+.*|GB\d+.*)[—\-]\d{4}$",
        re.IGNORECASE
    )

    def __init__(self, config: ParserConfig):
        self.config = config

    def clean_page(
        self,
        lines: List[LogicalLine],
        page_height: float
    ) -> List[LogicalLine]:

        result = []

        for line in lines:

            text = line.text.strip()

            if not text:
                continue

            # ------------------------------------------------
            # 页码
            #
            # 注意：
            #
            # 标题编号恢复已经在前面完成。
            #
            # 所以这里再删除底部纯数字才安全。
            # ------------------------------------------------

            if self._is_page_number(
                line,
                page_height
            ):
                continue

            # ------------------------------------------------
            # 页眉
            # ------------------------------------------------

            if self._is_header(
                line,
                page_height
            ):
                continue

            result.append(line)

        return result

    def _is_page_number(
        self,
        line: LogicalLine,
        page_height: float
    ) -> bool:

        if line.y0 < page_height * self.config.footer_ratio:
            return False

        text = line.text.strip()

        return bool(
            re.fullmatch(r"\d+", text)
        )

    def _is_header(
        self,
        line: LogicalLine,
        page_height: float
    ) -> bool:

        if line.y0 > page_height * self.config.header_ratio:
            return False

        text = line.text.strip()

        if self.STANDARD_HEADER_PATTERN.match(text):
            return True

        return False


# ============================================================
# 8. 文档块构建器
# ============================================================

class DocumentBlockBuilder:

    def __init__(
        self,
        document_name: str,
        heading_parser: HeadingParser
    ):

        self.document_name = document_name
        self.heading_parser = heading_parser

    def build(
        self,
        lines: List[LogicalLine]
    ) -> List[DocumentBlock]:

        blocks = []

        # ----------------------------------------------------
        # 当前标题上下文
        #
        # {
        #     1: "4 车辆外廓尺寸要求",
        #     2: "4.4 车辆通过性要求",
        #     3: "4.4.1 最小离地间隙"
        # }
        # ----------------------------------------------------

        headings: Dict[int, str] = {}

        buffer: List[LogicalLine] = []

        def flush():

            nonlocal buffer

            if not buffer:
                return

            text = self._join_buffer(buffer)

            if not text:
                buffer = []
                return

            heading_path = [
                headings[level]
                for level in sorted(headings)
            ]

            chapter = (
                headings.get(1)
            )

            section = (
                headings.get(
                    max(headings)
                )
                if headings
                else None
            )

            blocks.append(
                DocumentBlock(
                    document=self.document_name,

                    page_start=buffer[0].page,

                    page_end=buffer[-1].page,

                    chapter=chapter,

                    section=section,

                    heading_path=heading_path,

                    text=text,
                )
            )

            buffer = []

        # ----------------------------------------------------
        # 一行一行处理
        # ----------------------------------------------------

        for line in lines:

            heading = self.heading_parser.parse(
                line.text
            )

            # ------------------------------------------------
            # 标题
            # ------------------------------------------------

            if heading:

                # 先把前面的正文保存
                flush()

                # ------------------------------------------------
                # 图、表暂时作为正文内容
                #
                # 因为图/表通常不应该改变正文章节层级。
                # ------------------------------------------------

                if heading.kind in (
                    "figure",
                    "table",
                ):

                    buffer.append(
                        LogicalLine(
                            text=heading.full_text,
                            page=line.page,
                            x0=line.x0,
                            y0=line.y0,
                            x1=line.x1,
                            y1=line.y1,
                        )
                    )

                    continue

                # ------------------------------------------------
                # 更新标题上下文
                # ------------------------------------------------

                headings[heading.level] = (
                    heading.full_text
                )

                # 删除更深层标题
                for level in list(headings):

                    if level > heading.level:
                        del headings[level]

                continue

            # ------------------------------------------------
            # 普通正文
            # ------------------------------------------------

            buffer.append(line)

        # ----------------------------------------------------
        # 最后一块
        # ----------------------------------------------------

        flush()

        return blocks

    @staticmethod
    def _join_buffer(
        lines: List[LogicalLine]
    ) -> str:

        texts = []

        for line in lines:

            text = line.text.strip()

            if text:
                texts.append(text)

        return "\n".join(texts)


# ============================================================
# 9. 总解析器
# ============================================================

class PDFParser:

    def __init__(
        self,
        pdf_path: str,
        config: Optional[ParserConfig] = None
    ):

        self.pdf_path = pdf_path

        self.config = (
            config
            if config is not None
            else ParserConfig()
        )

        self.extractor = PDFExtractor(
            pdf_path
        )

        self.line_builder = LineBuilder(
            self.config
        )

        self.line_merger = LineMerger(
            self.config
        )

        self.cleaner = LineCleaner(
            self.config
        )

        self.heading_parser = HeadingParser(
            self.config
        )

    def parse(self) -> List[DocumentBlock]:

        document_name = self.pdf_path.split("/")[-1]

        all_lines = []

        with self.extractor.open() as doc:

            for page_index in range(
                len(doc)
            ):

                page = doc[page_index]

                page_number = page_index + 1

                # ------------------------------------------------
                # 1. 提取 words
                # ------------------------------------------------

                words = (
                    self.extractor.extract_words(
                        page
                    )
                )

                # ------------------------------------------------
                # 2. words -> LogicalLine
                # ------------------------------------------------

                lines = (
                    self.line_builder.build(
                        words,
                        page_number
                    )
                )

                # ------------------------------------------------
                # 3. 恢复被 PDF 拆开的标题编号
                #
                # 非常重要：
                #
                # 必须先做。
                # ------------------------------------------------

                lines = (
                    self.line_merger.merge(
                        lines
                    )
                )

                # ------------------------------------------------
                # 4. 删除页眉、页脚、页码
                # ------------------------------------------------

                if self.config.remove_header_footer:

                    lines = (
                        self.cleaner.clean_page(
                            lines,
                            page.rect.height
                        )
                    )

                all_lines.extend(lines)

        # --------------------------------------------------------
        # 5. 按逻辑标题构建 DocumentBlock
        # --------------------------------------------------------

        builder = DocumentBlockBuilder(
            document_name=document_name,
            heading_parser=self.heading_parser
        )

        blocks = builder.build(
            all_lines
        )

        return blocks


# ============================================================
# 10. JSON 输出
# ============================================================

def blocks_to_dict(
    blocks: List[DocumentBlock]
):

    result = []

    for block in blocks:

        result.append(
            {
                "document": block.document,

                "page_start": block.page_start,

                "page_end": block.page_end,

                "chapter": block.chapter,

                "section": block.section,

                "heading_path": block.heading_path,

                "text": block.text,
            }
        )

    return result


# ============================================================
# 11. 测试标题识别
# ============================================================

def test_heading_parser():

    config = ParserConfig()

    parser = HeadingParser(config)

    test_cases = [

        # 正文
        "4 车辆外廓尺寸要求",

        "4.1 车辆长度要求",

        "4.4车辆通过性要求",

        "4.4.1 最小离地间隙",

        "4.4.1.1 特殊车辆要求",

        # 附录
        "A 附录",

        "A.1 测量方法",

        "A.4.2.2 以下装置不在车辆宽度测量范围:",

        # 图
        "图 B.1 车辆外摆值示意图",

        "图 B.1.1 测量示意图",

        # 表
        "表 5 车辆尺寸要求",

        "表 5.1 车辆长度",

        # 普通正文
        "车辆长度应符合4.1的规定。",

    ]

    for text in test_cases:

        heading = parser.parse(text)

        print("=" * 70)

        print(
            f"输入: {text}"
        )

        if heading:

            print(
                f"类型: {heading.kind}"
            )

            print(
                f"编号: {heading.number}"
            )

            print(
                f"标题: {heading.title}"
            )

            print(
                f"层级: {heading.level}"
            )

        else:

            print("不是标题")


# ============================================================
# 12. 测试标题编号恢复
# ============================================================

def test_line_merger():

    config = ParserConfig()

    merger = LineMerger(config)

    examples = [

        (
            "4.7.2.最大允许总质量应不超过55000kg。",
            "4",
        ),

        (
            "4.7.2.后悬应符合4.5的要求。",
            "5",
        ),

        (
            "4.7.2.驱动轴的轴荷应符合4.6.1的要求。",
            "6",
        ),

        (
            "A.4.2.以下装置不在车辆宽度测量范围:",
            "2",
        ),

        (
            "A.2.车辆横向平面(X基准平面)",
            "3",
        ),

        (
            "图B.车辆外摆值示意图(汽车)",
            "1",
        ),

        (
            "图B.车辆外摆值示意图(汽车列车)",
            "2",
        ),

        (
            "B.2.上述过程顺时针及逆时针各进行一次。",
            "5",
        ),

        (
            "B.2.汽车或汽车列车起步,由直线行驶过渡到B.1.2所述的圆周内运动,至少车辆尾部进入",
            "3",
        ),
    ]

    for first, second in examples:

        lines = [

            LogicalLine(
                text=first,
                page=1,
                x0=100,
                y0=100,
                x1=500,
                y1=110,
            ),

            LogicalLine(
                text=second,
                page=1,
                x0=100,
                y0=112,
                x1=110,
                y1=122,
            ),
        ]

        result = merger.merge(lines)

        print("=" * 70)

        print("原始:")

        print(first)

        print(second)

        print()

        print("恢复后:")

        for line in result:

            print(line.text)


# ============================================================
# 13. 实际运行
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 修改成你的 PDF
    # --------------------------------------------------------

    PDF_PATH = "../../data/pdf/GB+1589-2026.pdf"

    # --------------------------------------------------------
    # 运行解析
    # --------------------------------------------------------

    parser = PDFParser(
        PDF_PATH
    )

    blocks = parser.parse()

    print(
        f"\n解析完成，共 {len(blocks)} 个 DocumentBlock\n"
    )

    # --------------------------------------------------------
    # 输出前 20 个 block
    # --------------------------------------------------------

    for index, block in enumerate(
        blocks[:40]
    ):

        print("=" * 80)

        print(
            f"Block {index + 1}"
        )

        print(
            f"文档: {block.document}"
        )

        print(
            f"页码: {block.page_start} - "
            f"{block.page_end}"
        )

        print(
            f"Chapter: {block.chapter}"
        )

        print(
            f"Section: {block.section}"
        )

        print(
            f"Heading Path: "
            f"{block.heading_path}"
        )

        print()

        print(block.text[:1000])

        print()