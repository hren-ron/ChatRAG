from io import StringIO
from pdfminer.high_level import extract_text_to_fp


def extract_pdf_text(pdf_path):
    output = StringIO()

    with open(pdf_path, "rb") as f:
        extract_text_to_fp(
            f,
            output,
            # laparams=laparams,
            output_type="text",
            codec="utf-8"
        )

    return output.getvalue()


text = extract_pdf_text("../data/pdf/GB+1589-2026.pdf")

print(text)