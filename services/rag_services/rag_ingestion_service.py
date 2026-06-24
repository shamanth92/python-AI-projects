from pathlib import Path
from services.rag_services.document_service import read_document
from services.rag_services.chunking_service import chunk_document
from services.rag_services.vectorization_service import vectorize_chunks
from services.rag_services.vector_store_service import store_embeddings


async def ingest_documents(docs_dir: str = "docs") -> dict:
    pdf_files = list(Path(docs_dir).glob("*.pdf"))
    total_chunks = 0
    total_embeddings = 0

    for pdf_path in pdf_files:
        document = read_document(str(pdf_path))
        chunks = chunk_document(document)
        embeddings = await vectorize_chunks(chunks)
        store_embeddings(embeddings)
        total_chunks += len(chunks)
        total_embeddings += len(embeddings)

    return {
        "documents_processed": len(pdf_files),
        "chunks_created": total_chunks,
        "embeddings_stored": total_embeddings
    }
