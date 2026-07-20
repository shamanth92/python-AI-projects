from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def rag_ingestion_service(file_path: str):
    loader = PyMuPDF4LLMLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = [c for c in text_splitter.split_documents(documents) if c.page_content.strip()]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
