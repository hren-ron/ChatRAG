from src.parser.pdf_parser import PDFParser

parser = PDFParser()

pages = parser.parse(
    "../../data/pdf/GB+1589-2026.pdf"
)

print(f"解析完成，共 {len(pages)} 个页面")

for page in pages[:10]:

    print("=" * 80)

    print(
        f"PAGE: {page['page']}"
    )

    print(
        f"CHAPTER: {page['chapter']}"
    )

    print(
        f"SECTION: {page['section']}"
    )

    print("-" * 80)

    print(
        page["text"]
    )
