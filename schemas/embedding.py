from pydantic import BaseModel
from schemas.chunk import Chunk


# Pairs a Chunk with its vector embedding. This is what actually gets written
# to ChromaDB: the embedding is used for similarity search, and the chunk
# (text + metadata) is what gets returned once a match is found.
class Embedding(BaseModel):
    chunk: Chunk
    embedding: list[float]  # 1536 floats for OpenAI's text-embedding-3-small
