# Python AI Learning

A FastAPI-based AI chatbot application that integrates with OpenAI to provide conversational AI responses, including streaming support.

## Features

- **Chat endpoint** — send a message and receive an AI response
- **Streaming endpoint** — stream AI responses in real-time via Server-Sent Events (SSE)
- **Conversation history** — messages are tracked by conversation ID across requests

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) — web framework
- [OpenAI Python SDK](https://github.com/openai/openai-python) — AI model integration
- [Pydantic](https://docs.pydantic.dev/) — request/response validation
- [Uvicorn](https://www.uvicorn.org/) — ASGI server

## Prerequisites

- Python 3.11+
- An OpenAI API key

## Setup

1. Clone the repository and create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -e .
   ```

3. Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

## API Endpoints

### `POST /chat/`

Send a message and receive a complete AI response.

**Request body:**
```json
{
  "message": "Hello, how are you?",
  "conversation_id": "optional-existing-id"
}
```

**Response:**
```json
{
  "response": "I'm doing well, thanks for asking!",
  "conversation_id": "abc123"
}
```

### `POST /chat/stream`

Stream an AI response chunk by chunk using Server-Sent Events.

Same request body as above; returns a streaming response.

## Project Structure

```
.
├── main.py                        # FastAPI app entry point
├── routers/
│   └── chat.py                    # Route definitions
├── schemas/
│   ├── chat.py                    # ChatRequest / ChatResponse models
│   └── message.py                 # Message model
├── services/
│   ├── chat_service.py            # OpenAI integration
│   └── conversation_service.py    # Conversation history management
└── pyproject.toml
```
