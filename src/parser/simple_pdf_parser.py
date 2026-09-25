import re
import fitz
from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block_no: int
    line_no: int
    word_no: int


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_no: int
    block_no: int = -1
    line_no: int = -1


# ============================================================
# PDF 文本解析器
# ============================================================

class PDFTextParser:

    # --------------------------------------------------------
    # 配置
    # --------------------------------------------------------

    # 页面顶部多少比例认为可能是页眉
    HEADER_RATIO = 0.05

    # 页面底部多少比例认为可能是页脚
    FOOTER_RATIO = 0.90

    # y 坐标相差多少认为属于同一视觉行
    Y_TOLERANCE = 3.0

    # 标题碎片之间允许的最大垂直距离
    HEADING_Y_TOLERANCE = 25.0

    # 标题最多 4 级
    MAX_HEADING_LEVEL = 4

    # 是否删除页眉
    REMOVE_HEADER = True

    # 是否删除页码
    REMOVE_PAGE_NUMBER = True

    # 是否规范空格
    NORMALIZE_SPACE = True

    # --------------------------------------------------------
    # 正则
    # --------------------------------------------------------

    # 普通数字标题：
    #
    # 1
    # 1.2
    # 1.2.3
    # 1.2.3.4
    #
    NUMERIC_HEADING_RE = re.compile(
        r"^\d+(?:\.\d+){0,3}$"
    )

    # 字母型标题：
    #
    # A
    # A.1
    # A.1.2
    # A.1.2.3
    #
    LETTER_HEADING_RE = re.compile(
        r"^[A-Z](?:\.\d+){0,3}$"
    )

    # 完整标题前缀
    #
    # 1.
    # 1.2.
    # B.
    # B.2.
    # B.2.4.
    #
    HEADING_PREFIX_RE = re.compile(
        r"^(?:"
        r"\d+(?:\.\d+){0,3}"
        r"|"
        r"[A-Z](?:\.\d+){0,3}"
        r")\.?$"
    )

    # 图号
    #
    # 图B.1
    # 图B.2
    # 图1
    #
    FIGURE_RE = re.compile(
        r"^图[A-Z]?\d+(?:\.\d+)*$"
    )

    # 标准号
    #
    # GB1589—2026
    # GB/T 1589—2026
    #
    STANDARD_RE = re.compile(
        r"^GB(?:/T)?\s*\d+(?:\.\d+)*\s*[—\-]\s*\d{4}$",
        re.I
    )

    # 页码
    PAGE_NUMBER_RE = re.compile(
        r"^(?:"
        r"\d+"
        r"|"
        r"第\s*\d+\s*页"
        r")$"
    )

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    # ========================================================
    # 对外接口
    # ========================================================

    def parse(self) -> str:

        doc = fitz.open(self.pdf_path)

        all_page_lines = []

        for page_no, page in enumerate(doc):

            words = self.extract_words(page)

            lines = self.build_visual_lines(
                words,
                page_no
            )

            # 页眉处理
            if self.REMOVE_HEADER:
                lines = self.remove_header(
                    lines,
                    page.rect.height
                )

            all_page_lines.append(lines)

        # ----------------------------------------------------
        # 关键：
        # 标题恢复必须在页面内部完成
        # 不要把所有页面 flatten 后再处理
        # ----------------------------------------------------

        result_lines = []

        for lines in all_page_lines:

            # 先恢复标题
            lines = self.recover_headings(lines)

            # 再删除页码
            lines = self.remove_page_numbers(lines)

            # 清洗文本
            lines = self.clean_lines(lines)

            result_lines.extend(lines)

        doc.close()

        # ----------------------------------------------------
        # 最终输出一个 TXT
        # ----------------------------------------------------

        result = []

        for line in result_lines:

            text = line.text.strip()

            if not text:
                continue

            result.append(text)

        return "\n".join(result)

    # ========================================================
    # 1. 提取 words
    # ========================================================

    def extract_words(self, page) -> List[Word]:

        raw_words = page.get_text("words")

        words = []

        for item in raw_words:

            x0, y0, x1, y1, text, block_no, line_no, word_no = item

            if not text.strip():
                continue

            words.append(
                Word(
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    block_no=block_no,
                    line_no=line_no,
                    word_no=word_no
                )
            )

        return words

    # ========================================================
    # 2. 根据 y 坐标构造视觉行
    #
    # 不完全依赖 block_no + line_no
    # ========================================================

    def build_visual_lines(
        self,
        words: List[Word],
        page_no: int
    ) -> List[Line]:

        if not words:
            return []

        # 先按照 y 排序
        words = sorted(
            words,
            key=lambda w: (
                w.y0,
                w.x0
            )
        )

        rows = []

        current_row = []
        current_y = None

        for word in words:

            if current_y is None:

                current_row = [word]
                current_y = word.y0

                continue

            # 和当前行的 y 坐标接近
            if abs(word.y0 - current_y) <= self.Y_TOLERANCE:

                current_row.append(word)

                # 动态更新平均 y
                current_y = sum(
                    w.y0 for w in current_row
                ) / len(current_row)

            else:

                rows.append(current_row)

                current_row = [word]
                current_y = word.y0

        if current_row:
            rows.append(current_row)

        lines = []

        for row in rows:

            # 同一视觉行内部按照 x 排序
            row = sorted(
                row,
                key=lambda w: w.x0
            )

            text = self.join_words(
                [w.text for w in row]
            )

            if not text.strip():
                continue

            lines.append(
                Line(
                    text=text,
                    x0=min(w.x0 for w in row),
                    y0=min(w.y0 for w in row),
                    x1=max(w.x1 for w in row),
                    y1=max(w.y1 for w in row),
                    page_no=page_no,
                    block_no=row[0].block_no,
                    line_no=row[0].line_no
                )
            )

        return lines

    # ========================================================
    # 3. 单词拼接
    # ========================================================

    def join_words(self, words: List[str]) -> str:

        if not words:
            return ""

        result = words[0]

        for word in words[1:]:

            if self.need_space(
                result[-1:] if result else "",
                word[:1]
            ):
                result += " "

            result += word

        return result

    # ========================================================
    # 4. 判断两个字符之间是否需要空格
    # ========================================================

    def need_space(
        self,
        left: str,
        right: str
    ) -> bool:

        if not left or not right:
            return False

        # 中文
        left_cn = self.is_chinese(left)
        right_cn = self.is_chinese(right)

        # 英文
        left_en = left.isascii() and left.isalpha()
        right_en = right.isascii() and right.isalpha()

        # 数字
        left_num = left.isdigit()
        right_num = right.isdigit()

        # ----------------------------------------------------
        # 中文 + 中文
        # ----------------------------------------------------

        if left_cn and right_cn:
            return False

        # ----------------------------------------------------
        # 数字 + 中文
        # 中文 + 数字
        # ----------------------------------------------------

        if left_num and right_cn:
            return True

        if left_cn and right_num:
            return True

        # ----------------------------------------------------
        # 英文 + 中文
        # 中文 + 英文
        # ----------------------------------------------------

        if left_en and right_cn:
            return True

        if left_cn and right_en:
            return True

        # ----------------------------------------------------
        # 数字 + 英文
        # 英文 + 数字
        # ----------------------------------------------------

        if left_num and right_en:
            return True

        if left_en and right_num:
            return True

        return False

    # ========================================================
    # 5. 中文判断
    # ========================================================

    @staticmethod
    def is_chinese(ch: str) -> bool:

        if not ch:
            return False

        code = ord(ch)

        return (
            0x4E00 <= code <= 0x9FFF
            or
            0x3400 <= code <= 0x4DBF
        )

    # ========================================================
    # 6. 删除页眉
    # ========================================================

    def remove_header(
        self,
        lines: List[Line],
        page_height: float
    ) -> List[Line]:

        result = []

        for line in lines:

            # 顶部区域
            if line.y0 <= page_height * self.HEADER_RATIO:

                text = line.text.strip()

                # 标准号
                if self.STANDARD_RE.match(text):
                    continue

            result.append(line)

        return result

    # ========================================================
    # 7. 删除页码
    # ========================================================

    def remove_page_numbers(
        self,
        lines: List[Line]
    ) -> List[Line]:

        result = []

        for line in lines:

            text = line.text.strip()

            # 只删除非常简单的页码
            if self.PAGE_NUMBER_RE.match(text):

                # 注意：
                # 只有孤立数字才删除
                #
                # 例如：
                # 4
                #
                # 但是：
                # B.2.4
                #
                # 不会进入这里

                if (
                    re.fullmatch(r"\d+", text)
                    and len(text) <= 4
                ):
                    continue

                if re.fullmatch(
                    r"第\s*\d+\s*页",
                    text
                ):
                    continue

            result.append(line)

        return result

    # ========================================================
    # 8. 标题恢复
    #
    # 这是整个解析器最重要的部分
    # ========================================================

    def recover_headings(
        self,
        lines: List[Line]
    ) -> List[Line]:

        if not lines:
            return []

        used = set()

        result = []

        i = 0

        while i < len(lines):

            if i in used:
                i += 1
                continue

            line = lines[i]

            # ------------------------------------------------
            # 尝试恢复标题
            # ------------------------------------------------

            recovered = self.try_recover_heading(
                lines,
                i,
                used
            )

            if recovered is not None:

                merged_line, consumed = recovered

                result.append(
                    merged_line
                )

                for idx in consumed:
                    used.add(idx)

                i += 1

                continue

            result.append(line)

            i += 1

        return result

    # ========================================================
    # 9. 尝试恢复标题
    # ========================================================

    def try_recover_heading(
        self,
        lines: List[Line],
        index: int,
        used: set
    ) -> Optional[tuple]:

        current = lines[index]

        text = current.text.strip()

        # ----------------------------------------------------
        # 情况一：
        #
        # 4上述过程中……
        #
        # B.
        # 2.
        #
        # =>
        #
        # B.2.4 上述过程中……
        # ----------------------------------------------------

        m = re.match(
            r"^(\d+)(.*)$",
            text
        )

        if m:

            last_number = m.group(1)
            body = m.group(2).strip()

            # 必须有正文
            if body:

                candidates = []

                # 在附近寻找：
                #
                # B.
                # 2.
                #
                for j in range(
                    max(0, index - 5),
                    min(len(lines), index + 6)
                ):

                    if j == index:
                        continue

                    if j in used:
                        continue

                    other = lines[j]

                    # y 距离不能太远
                    if abs(
                        other.y0 - current.y0
                    ) > self.HEADING_Y_TOLERANCE:
                        continue

                    other_text = other.text.strip()

                    # 必须是标题碎片
                    if not self.is_heading_fragment(
                        other_text
                    ):
                        continue

                    candidates.append(
                        (
                            j,
                            other
                        )
                    )

                if candidates:

                    prefix = self.build_heading_prefix(
                        last_number,
                        candidates
                    )

                    if prefix:

                        consumed = [
                            index
                        ]

                        consumed.extend(
                            j
                            for j, _ in candidates
                        )

                        # ------------------------------------------------
                        # 去重
                        # ------------------------------------------------

                        consumed = sorted(
                            set(consumed)
                        )

                        merged_text = (
                            prefix
                            + " "
                            + body
                        )

                        merged = Line(
                            text=self.normalize_text(
                                merged_text
                            ),
                            x0=min(
                                [current.x0]
                                + [
                                    x.x0
                                    for _, x in candidates
                                ]
                            ),
                            y0=min(
                                [current.y0]
                                + [
                                    x.y0
                                    for _, x in candidates
                                ]
                            ),
                            x1=max(
                                [current.x1]
                                + [
                                    x.x1
                                    for _, x in candidates
                                ]
                            ),
                            y1=max(
                                [current.y1]
                                + [
                                    x.y1
                                    for _, x in candidates
                                ]
                            ),
                            page_no=current.page_no
                        )

                        return merged, consumed

        # ----------------------------------------------------
        # 情况二：
        #
        # 图B.
        #
        # 1车辆外摆值示意图
        #
        # =>
        #
        # 图B.1 车辆外摆值示意图
        # ----------------------------------------------------

        if text.startswith("图"):

            candidates = []

            for j in range(
                max(0, index - 5),
                min(len(lines), index + 6)
            ):

                if j == index:
                    continue

                if j in used:
                    continue

                other = lines[j]

                if abs(
                    other.y0 - current.y0
                ) > self.HEADING_Y_TOLERANCE:
                    continue

                other_text = other.text.strip()

                # 找类似：
                #
                # 1车辆外摆值示意图
                #
                if re.match(
                    r"^\d+",
                    other_text
                ):
                    candidates.append(
                        (
                            j,
                            other
                        )
                    )

            if candidates:

                # 选择距离最近的
                candidates.sort(
                    key=lambda x:
                    abs(
                        x[1].y0 - current.y0
                    )
                )

                j, other = candidates[0]

                m = re.match(
                    r"^(\d+)(.*)$",
                    other.text.strip()
                )

                if m:

                    number = m.group(1)
                    body = m.group(2).strip()

                    prefix = text

                    # 图B. + 1
                    #
                    # => 图B.1
                    #
                    if prefix.endswith("."):
                        prefix += number
                    else:
                        prefix += "." + number

                    merged_text = (
                        prefix
                        + (
                            " " + body
                            if body
                            else ""
                        )
                    )

                    merged = Line(
                        text=self.normalize_text(
                            merged_text
                        ),
                        x0=min(
                            current.x0,
                            other.x0
                        ),
                        y0=min(
                            current.y0,
                            other.y0
                        ),
                        x1=max(
                            current.x1,
                            other.x1
                        ),
                        y1=max(
                            current.y1,
                            other.y1
                        ),
                        page_no=current.page_no
                    )

                    return merged, [
                        index,
                        j
                    ]

        return None

    # ========================================================
    # 10. 判断是否是标题碎片
    # ========================================================

    def is_heading_fragment(
        self,
        text: str
    ) -> bool:

        text = text.strip()

        if not text:
            return False

        # B.
        # 2.
        # 3
        # A
        #
        if re.fullmatch(
            r"[A-Z]\.",
            text
        ):
            return True

        if re.fullmatch(
            r"\d+\.",
            text
        ):
            return True

        if re.fullmatch(
            r"[A-Z]",
            text
        ):
            return True

        return False

    # ========================================================
    # 11. 构造标题编号
    # ========================================================

    def build_heading_prefix(
        self,
        last_number: str,
        candidates: List[tuple]
    ) -> Optional[str]:

        fragments = []

        for idx, line in candidates:

            text = line.text.strip()

            if re.fullmatch(
                r"[A-Z]\.",
                text
            ):
                fragments.append(
                    (
                        "letter",
                        text[:-1]
                    )
                )

            elif re.fullmatch(
                r"\d+\.",
                text
            ):
                fragments.append(
                    (
                        "number",
                        text[:-1]
                    )
                )

        if not fragments:
            return None

        # ----------------------------------------------------
        # 找字母根节点
        # ----------------------------------------------------

        letter = None

        for kind, value in fragments:

            if kind == "letter":
                letter = value
                break

        if letter is None:
            return None

        # ----------------------------------------------------
        # 找数字层级
        # ----------------------------------------------------

        numbers = [
            value
            for kind, value in fragments
            if kind == "number"
        ]

        # ----------------------------------------------------
        # 最多 4 级
        #
        # B.2.4
        #
        # B.2.4.5
        # ----------------------------------------------------

        parts = [letter]

        for number in numbers:

            if len(parts) >= self.MAX_HEADING_LEVEL:
                break

            parts.append(number)

        # 最后一层来自正文开头
        if len(parts) < self.MAX_HEADING_LEVEL:

            parts.append(
                last_number
            )

        return ".".join(parts)

    # ========================================================
    # 12. 文本清洗
    # ========================================================

    def clean_lines(
        self,
        lines: List[Line]
    ) -> List[Line]:

        result = []

        for line in lines:

            text = line.text.strip()

            if not text:
                continue

            # 标准号页眉
            if self.STANDARD_RE.match(text):
                continue

            # 删除多余空白
            text = self.normalize_text(
                text
            )

            if not text:
                continue

            result.append(
                Line(
                    text=text,
                    x0=line.x0,
                    y0=line.y0,
                    x1=line.x1,
                    y1=line.y1,
                    page_no=line.page_no,
                    block_no=line.block_no,
                    line_no=line.line_no
                )
            )

        return result

    # ========================================================
    # 13. 文本标准化
    # ========================================================

    def normalize_text(
        self,
        text: str
    ) -> str:

        if not self.NORMALIZE_SPACE:
            return text.strip()

        # ----------------------------------------------------
        # 首先统一空白
        # ----------------------------------------------------

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        # ----------------------------------------------------
        # 去除中文之间不应该出现的空格
        # ----------------------------------------------------

        text = re.sub(
            r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])",
            r"\1\2",
            text
        )

        # ----------------------------------------------------
        # 数字 / 中文
        #
        # 4上述
        # =>
        # 4 上述
        #
        # 上述4
        # =>
        # 上述 4
        # ----------------------------------------------------

        text = re.sub(
            r"(\d)([\u4e00-\u9fff])",
            r"\1 \2",
            text
        )

        text = re.sub(
            r"([\u4e00-\u9fff])(\d)",
            r"\1 \2",
            text
        )

        # ----------------------------------------------------
        # 英文 / 中文
        # ----------------------------------------------------

        text = re.sub(
            r"([A-Za-z])([\u4e00-\u9fff])",
            r"\1 \2",
            text
        )

        text = re.sub(
            r"([\u4e00-\u9fff])([A-Za-z])",
            r"\1 \2",
            text
        )

        # ----------------------------------------------------
        # 数字 / 英文
        # ----------------------------------------------------

        text = re.sub(
            r"(\d)([A-Za-z])",
            r"\1 \2",
            text
        )

        text = re.sub(
            r"([A-Za-z])(\d)",
            r"\1 \2",
            text
        )

        # ----------------------------------------------------
        # 标题编号特殊处理
        #
        # B.2.4上述
        #
        # =>
        #
        # B.2.4 上述
        # ----------------------------------------------------

        text = re.sub(
            r"^((?:[A-Z]\.)?\d+(?:\.\d+){0,3})([\u4e00-\u9fff])",
            r"\1 \2",
            text
        )

        # ----------------------------------------------------
        # 图B.1车辆
        #
        # =>
        #
        # 图B.1 车辆
        # ----------------------------------------------------

        text = re.sub(
            r"^(图[A-Z]?\d+(?:\.\d+)*)([\u4e00-\u9fff])",
            r"\1 \2",
            text
        )

        # ----------------------------------------------------
        # 多个空格
        # ----------------------------------------------------

        text = re.sub(
            r" {2,}",
            " ",
            text
        )

        return text.strip()


# ============================================================
# 使用
# ============================================================

# if __name__ == "__main__":
#
#     pdf_path = "../../data/pdf/GB+1589-2026.pdf"
#     output_path = "GB1589-2026.txt"
#
#     parser = PDFTextParser(
#         pdf_path
#     )
#
#     text = parser.parse()
#
#     with open(
#         output_path,
#         "w",
#         encoding="utf-8"
#     ) as f:
#
#         f.write(text)
#
#     print(
#         f"解析完成：{output_path}"
#     )