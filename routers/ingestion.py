from fastapi import APIRouter, HTTPException
from pathlib import Path
from services.rag_services.rag_ingestion_service import ingest_documents
from services.rag_services.vector_store_service import get_collection, client

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post("/")
async def ingest(docs_dir: str = "docs"):
    if not Path(docs_dir).exists():
        raise HTTPException(status_code=404, detail=f"Directory '{docs_dir}' not found")

    pdf_files = list(Path(docs_dir).glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(status_code=404, detail=f"No PDF files found in '{docs_dir}'")

    try:
        return await ingest_documents(docs_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug")
def debug():
    try:
        collection = get_collection()
        peek = collection.peek()
        return {
            "count": collection.count(),
            "sample": {
                "ids": peek["ids"],
                "documents": peek["documents"],
                "metadatas": peek["metadatas"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/")
def delete_all():
    client.delete_collection("documents")
    client.get_or_create_collection("documents")
    return {"message": "All records deleted"}
