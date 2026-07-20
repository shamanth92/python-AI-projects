# AWS AI Practitioner Study RAG

A FastAPI app that turns AWS whitepapers/exam guides (PDFs) into a study assistant using
Retrieval-Augmented Generation (RAG). Ask questions about AWS AI/ML concepts, or generate
practice exam questions grounded in the ingested documents.

The app implements the same RAG pipeline **twice**, side by side, for learning purposes:

- **Basic pipeline** (`/ingestion`, `/chat/rag`) — hand-rolled, no LangChain
- **LangChain pipeline** (`/langchain/ingestion`, `/langchain/chat/rag`) — same steps, built with LangChain

Both use ChromaDB for vector storage (in separate collections) and OpenAI for embeddings + chat.

## Features

- **Ingestion** — reads PDFs from a `docs/` folder, chunks them, embeds the chunks, and stores them in ChromaDB
- **Q&A mode** — ask a question, get an answer grounded in the retrieved chunks
- **Exam mode** — ask about a topic, get AWS-style scenario-based practice questions (with answers,
  explanations, and citations back to the source document/page) generated from the retrieved chunks
- **Delete endpoints** — clear out a collection to re-ingest from scratch

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) — web framework
- [OpenAI Python SDK](https://github.com/openai/openai-python) — embeddings + chat completions
- [LangChain](https://python.langchain.com/) — LCEL-based RAG pipeline (loader, splitter, retriever, chain)
- [ChromaDB](https://www.trychroma.com/) — vector store (persisted locally to `./chroma_db`)
- [PyMuPDF](https://pymupdf.readthedocs.io/) / `pymupdf4llm` — PDF text extraction
- [Pydantic](https://docs.pydantic.dev/) — request/response validation

## Prerequisites

- Python 3.11+
- An OpenAI API key
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

1. Clone the repository and install dependencies:

   ```bash
   uv sync
   ```

2. Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

3. Create a `docs/` folder in the project root and drop in the PDFs you want to ingest
   (e.g. AWS whitepapers, the AI Practitioner exam guide).

## Running the Server

```bash
uv run fastapi dev main.py
```

The API will be available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

## API Endpoints

### Basic (non-LangChain) pipeline

#### `POST /ingestion/`

Reads every PDF in `docs/`, chunks it, embeds the chunks, and stores them in the `documents` ChromaDB collection.

**Response:**
```json
{
  "documents_processed": 3,
  "chunks_created": 245,
  "embeddings_stored": 245
}
```

#### `GET /ingestion/debug`

Peek at a sample of stored records (ids, text, metadata) to sanity-check ingestion.

#### `DELETE /ingestion/`

Clears the `documents` collection so you can re-ingest from scratch.

#### `POST /chat/rag/`

**Request body:**
```json
{
  "message": "What is Amazon Bedrock?"
}
```

**Response:**
```json
{
  "answer": "Amazon Bedrock is a fully managed service that ..."
}
```

### LangChain pipeline

#### `POST /langchain/ingestion/`

Same as above, but built with LangChain (`PyMuPDF4LLMLoader`, `RecursiveCharacterTextSplitter`,
`OpenAIEmbeddings`, `Chroma`). Stores into the `aws-rag-documents` collection.

#### `DELETE /langchain/ingestion/`

Clears the `aws-rag-documents` collection so you can re-ingest from scratch.

#### `POST /langchain/chat/rag/`

Single endpoint for both Q&A and exam-question generation — the `mode` field selects the prompt.

**Q&A mode (default):**
```json
{
  "message": "What is Amazon Bedrock?"
}
```

**Exam mode:**
```json
{
  "message": "Amazon Bedrock",
  "mode": "exam"
}
```

Exam mode returns AWS-style scenario-based multiple choice questions, each with the correct
answer, an explanation, and a citation back to the source document/page.

## Project Structure

```
.
├── main.py                                     # FastAPI app entry point, registers all routers
├── docs/                                        # PDFs to ingest (gitignored)
├── chroma_db/                                   # persisted vector store (gitignored)
├── routers/
│   ├── basicRag/
│   │   ├── ingestion.py                         # POST/GET/DELETE for the basic pipeline
│   │   └── rag_chat.py                          # POST /chat/rag/
│   └── langChainRag/
│       ├── langchain_ingestion.py               # POST/DELETE for the LangChain pipeline
│       └── langchain_rag_chat.py                # POST /langchain/chat/rag/
├── schemas/
│   ├── chat.py                                  # ChatRequest / ChatResponse / ChatMode
│   ├── document.py                              # Document / Page models
│   ├── chunk.py                                 # Chunk model
│   └── embedding.py                             # Embedding model
├── services/
│   ├── openai_client.py                         # shared AsyncOpenAI client
│   ├── rag_services/                            # basic pipeline: read -> chunk -> embed -> store -> retrieve -> chat
│   │   ├── document_service.py
│   │   ├── chunking_service.py
│   │   ├── vectorization_service.py
│   │   ├── vector_store_service.py
│   │   ├── retrieval_service.py
│   │   └── rag_chat_service.py
│   └── langchain_rag_services/                  # LangChain pipeline
│       ├── rag_ingestion_service.py
│       └── rag_retrieval_service.py
└── pyproject.toml
```
