from pathlib import Path

from simple_pdf_parser import PDFTextParser
from chapter_splitter import ChapterSplitter


class PDFPipeline:
    """
    PDF 文档处理流水线

    Step 1:
        PDF → 完整文本

    Step 2:
        完整文本 → 多个章节 TXT

    不保存中间的完整 TXT 文件。
    """

    def __init__(
        self,
        pdf_path: str,
        output_dir: str
    ):
        # PDF 文件
        self.pdf_path = Path(pdf_path)

        # PDF 文件名，不包含 .pdf
        # 例如：
        # GB+1589-2026.pdf
        # ↓
        # GB+1589-2026
        self.document_name = self.pdf_path.stem

        # 当前文档的输出目录
        #
        # output/
        # └── GB+1589-2026/
        self.document_dir = (
            Path(output_dir) / self.document_name
        )

    def run(self):

        print("=" * 60)
        print("PDF 文档处理 Pipeline")
        print("=" * 60)

        print(f"输入 PDF：{self.pdf_path}")
        print(f"文档名称：{self.document_name}")
        print(f"输出目录：{self.document_dir}")

        # 创建输出目录
        self.document_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ==========================================================
        # Step 1: PDF → 完整文本
        # ==========================================================

        print("\n[1/2] PDF → 完整文本")
        print("-" * 60)

        parser = PDFTextParser(
            pdf_path=str(self.pdf_path)
        )

        text = parser.parse()

        print(f"文本长度：{len(text)}")

        # ==========================================================
        # Step 2: 完整文本 → 多个章节 TXT
        # ==========================================================

        print("\n[2/2] 完整文本 → 多个章节 TXT")
        print("-" * 60)

        splitter = ChapterSplitter(
            text=text,
            output_dir=str(self.document_dir)
        )

        splitter.split()

        # ==========================================================
        # 统计生成的文件
        # ==========================================================

        chunk_files = sorted(
            self.document_dir.glob("*.txt")
        )

        print("\n生成的文件：")

        for chunk_file in chunk_files:
            print(f"  {chunk_file.name}")

        print("\n" + "=" * 60)
        print("Pipeline 执行完成")
        print("=" * 60)

        print(f"输出目录：{self.document_dir}")
        print(f"文件数量：{len(chunk_files)}")


if __name__ == "__main__":

    # 当前文件：
    #
    # ChatRAG/
    # ├── src/
    # │   └── parser/
    # │       └── run_pipeline.py
    # │
    # └── data/
    #     └── pdf/
    #         └── GB+1589-2026.pdf

    current_dir = Path(__file__).resolve().parent

    pdf_path = (
        current_dir
        / "../../data/pdf/GB+1589-2026.pdf"
    ).resolve()

    output_dir = (
        current_dir
        / "../../data/output"
    ).resolve()

    pipeline = PDFPipeline(
        pdf_path=str(pdf_path),
        output_dir=str(output_dir)
    )

    pipeline.run()