from services.openai_client import client
from schemas.chunk import Chunk
from schemas.embedding import Embedding

EMBEDDING_MODEL = "text-embedding-3-small"


async def vectorize_chunks(chunks: list[Chunk]) -> list[Embedding]:
    texts = [chunk.text for chunk in chunks]

    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return [
        Embedding(chunk=chunk, embedding=result.embedding)
        for chunk, result in zip(chunks, response.data)
    ]
