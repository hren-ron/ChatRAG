import re
from pathlib import Path

import fitz


class PDFParser:

    def parse(self, pdf_path: str | Path):
        """
        解析PDF, 按页返回文本
        :param pdf_path: pdf路径
        :return:
        [
            {
                "page": 1,
                "text": "第一页文本..."
            },
            {
                "page": 2,
                "text": "第二页文本..."
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

        pages = []

        with fitz.open(pdf_path) as doc:

            for page_index, page in enumerate(doc):
                page_number = page_index + 1

                text = page.get_text("text")

                text = self.clean_text(text)

                # 空白页不保存
                if not text:
                    continue

                print(page_number)
                print(text)

                pages.append(
                    {
                        "page": page_number,
                        "text": text
                    }
                )
        return pages

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



