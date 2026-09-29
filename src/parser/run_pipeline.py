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


class BatchPDFPipeline:
    """
        批量 PDF 文档处理流水线

        自动遍历 PDF 目录中的所有 PDF 文件，
        每个 PDF 独立执行 PDFPipeline。
    """

    def __init__(self, pdf_dir: str, output_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)

    def run(self):

        # 获取所有pdf文件
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"no find pdf files:{self.pdf_dir}")
            return

        print("=" * 60)
        print("process pdf files")
        print("=" * 60)

        print(f"pfd files dir: {self.pdf_dir}")
        print(f"the number of pdf files: {len(pdf_files)}")

        success = 0
        failed = 0

        for index, pdf_file in enumerate(pdf_files, start=1):
            print("\n")

            print(f"=========== [{index}/{len(pdf_files)}] ===========")

            try:
                pipeline = PDFPipeline(pdf_path=str(pdf_file), output_dir=str(self.output_dir))

                pipeline.run()

                success += 1
            except Exception as e:
                failed += 1
                print(e)
                print(f"process {pdf_file.name} failed")


        print("\n")
        print("=" * 60)
        print(f"process pdf files finished")
        print("=" * 60)

        print(f"total number of pdf files: {len(pdf_files)}")
        print(f"success number: {success}")
        print(f"failed number: {failed}")

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

    # current_dir = Path(__file__).resolve().parent

    # pdf_path = (
    #     current_dir
    #     / "../../data/pdf/GB+1589-2026.pdf"
    # ).resolve()
    #
    # output_dir = (
    #     current_dir
    #     / "../../data/output"
    # ).resolve()
    #
    # pipeline = PDFPipeline(
    #     pdf_path=str(pdf_path),
    #     output_dir=str(output_dir)
    # )
    #
    # pipeline.run()

    current_dir = Path(__file__).resolve().parent

    pdf_dir = (
            current_dir / "../../data/pdf"
    ).resolve()

    output_dir = (
            current_dir / "../../data/output"
    ).resolve()

    pipeline = BatchPDFPipeline(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir)
    )

    pipeline.run()