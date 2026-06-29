from fastapi import APIRouter, HTTPException
from pathlib import Path
from services.langchain_rag_services.rag_ingestion_service import rag_ingestion_service

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
