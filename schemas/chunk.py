from pydantic import BaseModel

class Chunk(BaseModel):
    chunk_id: str
    text: str
    document_name: str
    page_number: int
