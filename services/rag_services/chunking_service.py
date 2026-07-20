from schemas.chunk import Chunk
from schemas.document import Document

# CHUNK_SIZE: max characters per chunk. Smaller chunks = more precise retrieval
# but less surrounding context per chunk; bigger chunks = opposite trade-off.
# OVERLAP: characters repeated between consecutive chunks so a sentence that
# straddles a chunk boundary still appears whole in at least one chunk.
CHUNK_SIZE = 1000
OVERLAP = 200


# Step 2 of the RAG pipeline: split each page's text into overlapping,
# fixed-size chunks. This is a naive sliding-window splitter (LangChain's
# RecursiveCharacterTextSplitter, used in the langchain pipeline, does the
# same job but splits on sentence/paragraph boundaries where possible).
def chunk_document(document: Document) -> list[Chunk]:
    chunks = []
    base_name = document.file_name.rsplit(".", 1)[0]

    for page in document.pages:
        text = page.content
        start = 0
        i = 0

        while start < len(text):
            chunks.append(Chunk(
                # id must be unique across the whole collection, hence page + index
                chunk_id=f"{base_name}_p{page.page_number}_c{i}",
                text=text[start:start + CHUNK_SIZE],
                document_name=document.file_name,
                page_number=page.page_number
            ))

            # advance by (chunk size - overlap) so the next chunk overlaps the last
            start += CHUNK_SIZE - OVERLAP
            i += 1

    return chunks
