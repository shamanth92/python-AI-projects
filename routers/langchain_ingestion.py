from fastapi import APIRouter, HTTPException
from pathlib import Path
import chromadb
from services.langchain_rag_services.rag_ingestion_service import rag_ingestion_service

chroma_client = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "aws-rag-documents"

router = APIRouter(prefix="/langchain/ingestion", tags=["langchain-ingestion"])


@router.post("/")
def ingest(docs_dir: str = "docs"):
    if not Path(docs_dir).exists():
        raise HTTPException(status_code=404, detail=f"Directory '{docs_dir}' not found")

    pdf_files = list(Path(docs_dir).glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(status_code=404, detail=f"No PDF files found in '{docs_dir}'")

    try:
        total_chunks = 0
        for pdf_path in pdf_files:
            result = rag_ingestion_service(str(pdf_path))
            total_chunks += result["chunks_created"]

        return {
            "documents_processed": len(pdf_files),
            "chunks_created": total_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/")
def delete_all():
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    chroma_client.get_or_create_collection(COLLECTION_NAME)
    return {"message": f"Collection '{COLLECTION_NAME}' cleared"}
