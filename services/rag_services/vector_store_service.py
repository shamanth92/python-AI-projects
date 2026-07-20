import chromadb
from schemas.embedding import Embedding

# PersistentClient writes to disk (./chroma_db) so embeddings survive server
# restarts. This is the vector database for the "basic" (non-LangChain) pipeline;
# the LangChain pipeline uses a separate collection in the same chroma_db folder.
client = chromadb.PersistentClient(path="./chroma_db")


# Fetches (or creates) the collection fresh on every call rather than caching
# it at module level. Important: after a delete_collection() call, a cached
# collection object becomes stale and raises errors -- always go through this
# function instead of holding onto a collection reference.
def get_collection():
    return client.get_or_create_collection(name="documents")


# Step 4 of the RAG pipeline: persist chunk text + embedding vector + metadata
# into ChromaDB. `ids` must be unique per record (see chunking_service for how
# chunk_id is built) or this call will raise a duplicate ID error.
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
