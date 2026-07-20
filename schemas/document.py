from pydantic import BaseModel


# Represents one page of text extracted from a source PDF.
class Page(BaseModel):
    page_number: int
    content: str


# Represents an entire ingested PDF before it gets split into Chunks.
# This is the output of document_service.read_document() and the input
# to chunking_service.chunk_document().
class Document(BaseModel):
    file_name: str
    pages: list[Page]
