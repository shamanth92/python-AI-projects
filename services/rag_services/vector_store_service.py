import chromadb
from schemas.embedding import Embedding

client = chromadb.PersistentClient(path="./chroma_db")


def get_collection():
    return client.get_or_create_collection(name="documents")


def store_embeddings(embeddings: list[Embedding]) -> None:
    get_collection().add(
        ids=[e.chunk.chunk_id for e in embeddings],
        embeddings=[e.embedding for e in embeddings],
        documents=[e.chunk.text for e in embeddings],
        metadatas=[
            {
                "document_name": e.chunk.document_name,
                "page_number": e.chunk.page_number
            }
            for e in embeddings
        ]
    )
