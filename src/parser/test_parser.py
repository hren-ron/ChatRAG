from src.parser.pdf_parser import PDFParser

parser = PDFParser()

pages = parser.parse(
    "../../data/pdf/GB+1589-2026.pdf"
)

print(f"总页数：{len(pages)}")

for page in pages[:3]:
    print("=" * 50)
    print(page['page'])
    print(page['text'])