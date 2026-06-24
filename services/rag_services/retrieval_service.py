from services.openai_client import client
from services.rag_services.vectorization_service import EMBEDDING_MODEL
from services.rag_services.vector_store_service import get_collection
from schemas.chunk import Chunk


async def retrieve_chunks(question: str, top_k: int = 5) -> list[Chunk]:
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    question_embedding = response.data[0].embedding

    collection = get_collection()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append(Chunk(
            chunk_id=results["ids"][0][i],
            text=results["documents"][0][i],
            document_name=results["metadatas"][0][i]["document_name"],
            page_number=results["metadatas"][0][i]["page_number"]
        ))

    return chunks
