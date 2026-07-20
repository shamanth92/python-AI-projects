from services.openai_client import client
from schemas.chunk import Chunk
from schemas.embedding import Embedding

# Shared here (and imported by retrieval_service.py) so ingestion and query-time
# embeddings always use the exact same model -- mixing models would produce
# vectors that aren't comparable to each other.
EMBEDDING_MODEL = "text-embedding-3-small"


# Step 3 of the RAG pipeline: turn each chunk's text into a vector embedding
# via the OpenAI embeddings API. Batched into a single API call (all chunk
# texts sent as one `input` list) instead of one call per chunk, for speed.
async def vectorize_chunks(chunks: list[Chunk]) -> list[Embedding]:
    texts = [chunk.text for chunk in chunks]

    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    # response.data is returned in the same order as the input list, so we
    # can zip it back up with the original chunks positionally.
    return [
        Embedding(chunk=chunk, embedding=result.embedding)
        for chunk, result in zip(chunks, response.data)
    ]
