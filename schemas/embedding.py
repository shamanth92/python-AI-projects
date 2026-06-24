from pydantic import BaseModel
from schemas.chunk import Chunk

class Embedding(BaseModel):
    chunk: Chunk
    embedding: list[float]
