import fitz  # PyMuPDF
from pathlib import Path
from schemas.document import Document, Page


# Step 1 of the RAG pipeline: read a PDF off disk into a structured Document
# (one Page per PDF page). No chunking or embedding happens here -- that's
# the job of chunking_service and vectorization_service.
def read_document(file_path: str) -> Document:
    path = Path(file_path)
    with fitz.open(file_path) as doc:
        pages = [
            Page(page_number=i + 1, content=page.get_text())
            for i, page in enumerate(doc)
        ]

    return Document(file_name=path.name, pages=pages)
