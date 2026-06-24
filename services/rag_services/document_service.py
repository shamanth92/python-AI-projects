import fitz
from pathlib import Path
from schemas.document import Document, Page


def read_document(file_path: str) -> Document:
    path = Path(file_path)
    with fitz.open(file_path) as doc:
        pages = [
            Page(page_number=i + 1, content=page.get_text())
            for i, page in enumerate(doc)
        ]

    return Document(file_name=path.name, pages=pages)
