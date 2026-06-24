from schemas.chunk import Chunk
from schemas.document import Document

CHUNK_SIZE = 1000
OVERLAP = 200


def chunk_document(document: Document) -> list[Chunk]:
    chunks = []
    base_name = document.file_name.rsplit(".", 1)[0]

    for page in document.pages:
        text = page.content
        start = 0
        i = 0

        while start < len(text):
            chunks.append(Chunk(
                chunk_id=f"{base_name}_p{page.page_number}_c{i}",
                text=text[start:start + CHUNK_SIZE],
                document_name=document.file_name,
                page_number=page.page_number
            ))

            start += CHUNK_SIZE - OVERLAP
            i += 1

    return chunks
