from PyPDF2 import PdfReader
from markitdown import MarkItDown

def get_pdf_content(documents):
    raw_text = ""

    for document in documents:
        pdf_reader = PdfReader(document)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

from markitdown import MarkItDown
md = MarkItDown()
result = md.convert("test.pdf")
with open("test2.txt", "w") as file:
    file.write(result.text_content)


pdf_text = get_pdf_content(["test.pdf"])
with open("test1.txt", "w") as file:
    file.write(pdf_text)