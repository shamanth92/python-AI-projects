from pydantic import BaseModel


# A Chunk is a slice of a document's text small enough to embed and retrieve
# individually. RAG never sends whole documents to the LLM -- it stores many
# small chunks, retrieves only the relevant ones for a given question, and
# sends just those as context.
class Chunk(BaseModel):
    chunk_id: str       # unique id, e.g. "<filename>_p<page>_c<index>" -- used as the ChromaDB record id
    text: str           # the actual chunk text that gets embedded and later shown to the LLM
    document_name: str  # source file, kept so answers can cite where they came from
    page_number: int    # source page, same reason as above
