import re
from pathlib import Path
from typing import List, Dict, Tuple

import fitz


class PDFParser:

    """
    面向中文标准/技术文档的 PDF Parser。

    主要功能：
    1. PDF 文本提取
    2. 页眉/页脚检测
    3. 页码删除
    4. 跨行文本合并
    5. 标准编号修复
    6. 中英文文本规范化
    7. 章节结构识别
    """

    def __init__(
            self,
            header_footer_threshold: float=0.5,
            header_lines: int=3,
            footer_lines: int=3
    ):
        self.threshold = header_footer_threshold
        self.header_lines = header_lines
        self.footer_lines = footer_lines

    def parse(self, pdf_path: str | Path) -> List[Dict]:
        """
        解析PDF
        :param pdf_path: pdf路径
        :return:
        [
            {
                "page": 5,
                "chapter": "3 术语和定义",
                "section": "3.1 车辆长度",
                "text": "依据附录A测得..."
            }
        ]
        """

        pdf_path = Path(pdf_path)

        if not pdf_path:
            raise FileNotFoundError(
                f"PDF is not exists: {pdf_path}"
            )

        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(
                f"Not PDF file: {pdf_path}"
            )

        # -----------------------------------------------------
        # Step 1：提取原始页面
        # -----------------------------------------------------
        raw_pages = self.extract_pages(pdf_path)

        # -----------------------------------------------------
        # Step 2：检测页眉 / 页脚
        # -----------------------------------------------------
        # headers, footers = self.detect_headers_footers(raw_pages)

    # =========================================================
    # 2. PDF 提取
    # =========================================================
    def extract_pages(self, path: Path) -> List[Dict]:

        pages = []

        with fitz.open(path) as doc:

            for page_index, page in enumerate(doc):
                page_number = page_index + 1

                lines = self.extract_lines_from_words(page)

                pages.append(
                    {
                        "page": page_number,
                        "lines": lines
                    }
                )
        return pages

    # =========================================================
    # 3. 使用 words 提取文本
    # =========================================================
    @staticmethod
    def extract_lines_from_words(page) -> List[str]:
        """
        使用 PyMuPDF words API 提取文本。

        words 格式：
            x0, y0, x1, y1,
            word,
            block_no,
            line_no,
            word_no
        """

        words = page.get_text("words")

        if not words:
            return []

        # -----------------------------------------------------
        # 按 block + line 分组
        # -----------------------------------------------------

        line_groups = {}

        for word in words:
            x0, y0, x1, y1, text = word[:5]

            block_no = word[5]
            line_no = word[6]

            key = (block_no, line_no)

            if key not in line_groups:
                line_groups[key] = []

            line_groups[key].append(
                {
                    "x0": x0,
                    "x1": x1,
                    "text": text
                }
            )

        # -----------------------------------------------------
        # 按 block / line 顺序排列
        # -----------------------------------------------------

        sorted_lines = []
        for key, words_in_line in line_groups.items():
            words_in_line.sort(
                key=lambda x:x["x0"]
            )

            line_text = (
                PDFParser.join_words(
                    words_in_line
                )
            )

            sorted_lines.append(
                (key[0], key[1], line_text)
            )

        # -----------------------------------------------------
        # block → line
        # -----------------------------------------------------

        sorted_lines.sort(
            key=lambda x:(x[0], x[1])
        )

        return [item[2] for item in sorted_lines]

    # =========================================================
    # 4. words 合并
    # =========================================================
    @staticmethod
    def join_words(words: List[Dict]) -> str:
        """
        根据 x 坐标判断两个 word 是否需要空格。
        中文通常直接连接：车辆长度

        英文：vehicle length

        中英文：长度 vehicle
        """

        if not words:
            return ""

        result = words[0]["text"]
        for i in range(1, len(words)):

            previous = words[i-1]
            current = words[i]

            prev_text = previous["text"]
            curr_text = current["text"]

            gap = current["x0"] - previous["x1"]

            if PDFParser.need_space(prev_text, curr_text, gap):
                result += " "

            result += curr_text
        return result

    # =========================================================
    # 5. 判断是否需要空格
    # =========================================================
    @staticmethod
    def need_space(
        previous: str,
        current: str,
        gap: float
    ) -> bool:
        """
        判断两个 PDF word 之间是否需要空格。
        """

        if not previous or not current:
            return False

        prev_char = previous[-1]
        curr_char = current[0]

        prev_is_chinese = (
            "\u4e00" <= prev_char <= "\u9fff"
        )

        curr_is_chinese = (
            "\u4e00" <= curr_char <= "\u9fff"
        )

        prev_is_alpha = prev_char.isascii() and prev_char.isalpha()
        curr_is_alpha = curr_char.isascii() and curr_char.isalpha()

        # 中文 + 中文
        if prev_is_chinese and curr_is_chinese:
            return False

        # 中文 + 英文
        if prev_is_chinese and curr_is_alpha:
            return True

        # 英文 + 中文
        if prev_is_alpha and curr_is_chinese:
            return True

        # 英文 + 英文
        if prev_is_alpha and curr_is_alpha:
            return True

        # 数字 + 中文
        if prev_char.isdigit() and curr_is_chinese:
            return False

        # 中文 + 数字
        if prev_is_chinese and curr_char.isdigit():
            return False

        # 数字 + 英文
        if prev_char.isdigit() and curr_is_alpha:
            return False

        # 英文 + 数字
        if prev_is_alpha and curr_char.isdigit():
            return False

        # 如果 PDF 中两个 word 之间存在明显间距
        if gap > 3:
            return True

        return False

    # =========================================================
    # 6. 页眉 / 页脚检测
    # =========================================================
    def detect_headers_footers(self, pages:List[Dict]) -> Tuple[set, set]:
        pass

    @staticmethod
    def clean_text(text: str) -> str:

        # 去掉首位空白
        text = text.strip()

        # 全角空格 --> 普通空格
        text = text.replace("\u3000", " ")

        # 连续空格压缩
        text = re.sub(r"[ \t]+", " ", text)

        # 连续空行压缩
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 标准编号处理
        text = re.sub(
            r"GB/T\s*(\d+).\s*(\d+)",
            r"GB/T\1.\2",
            text
        )

        # 中英文之间补空格
        text = re.sub(
            r"([\u4e00-\u9fff])([A-Za-z])",
            r"\1 \2",
            text
        )

        text = re.sub(
            r"([A-Za-z]([\u4e00-\u9fff]))",
            r"\1 \2",
            text
        )

        return text



