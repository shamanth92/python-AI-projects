from pydantic import BaseModel

class Page(BaseModel):
    page_number: int
    content: str

class Document(BaseModel):
    file_name: str
    pages: list[Page]

