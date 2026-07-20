from fastapi import APIRouter, HTTPException
from pathlib import Path
from services.rag_services.rag_ingestion_service import ingest_documents
from services.rag_services.vector_store_service import get_collection, client

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


# Runs the full basic (non-LangChain) ingestion pipeline over every PDF in
# docs_dir. Directory/file-not-found checks happen here (in the router) so
# they return proper 404s; anything unexpected during ingestion itself
# bubbles up as a 500.
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


# Lets you sanity-check ingestion without a separate vector DB GUI --
# peek() returns a small sample of stored records. Note: raw embedding
# vectors are deliberately excluded from the response since they're not
# JSON-serializable-friendly (1536 floats each) and aren't useful to a human.
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


# Wipes and recreates the collection rather than deleting individual records --
# simpler and avoids issues with unreliable `where` filter deletes.
@router.delete("/")
def delete_all():
    client.delete_collection("documents")
    client.get_or_create_collection("documents")
    return {"message": "All records deleted"}
