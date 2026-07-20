from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# LangChain equivalent of the basic pipeline's read -> chunk -> embed -> store
# steps, collapsed into one function since LangChain's building blocks each
# handle one of those steps internally:
#   PyMuPDF4LLMLoader           -> reads the PDF (like document_service.read_document)
#   RecursiveCharacterTextSplitter -> chunks text, splitting on paragraph/sentence
#                                     boundaries where possible (unlike the basic
#                                     pipeline's naive fixed-size slicing)
#   OpenAIEmbeddings + Chroma   -> embeds and stores in one call (add_documents
#                                  embeds internally, unlike the basic pipeline
#                                  where vectorization and storage are separate steps)
def rag_ingestion_service(file_path: str):
    loader = PyMuPDF4LLMLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    # Filter out empty chunks (e.g. image-only pages produce blank text) --
    # ChromaDB rejects add_documents() calls containing an empty embedding.
    chunks = [c for c in text_splitter.split_documents(documents) if c.page_content.strip()]

    # A file with no extractable text (e.g. all pages are scanned images)
    # ends up with zero chunks -- add_documents([]) would still try to embed
    # an empty list and raise the same "non-empty list ... got []" error, so
    # skip the store call entirely in that case.
    if not chunks:
        return {
            "file": file_path,
            "chunks_created": 0
        }

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Uses a separate collection name ("aws-rag-documents") from the basic
    # pipeline's "documents" collection, in the same chroma_db folder, so the
    # two pipelines never overwrite each other's data.
    vector_store = Chroma(
        collection_name="aws-rag-documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    vector_store.add_documents(chunks)

    return {
        "file": file_path,
        "chunks_created": len(chunks)
    }
