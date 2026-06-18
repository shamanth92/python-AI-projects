from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from schemas.chat import ChatRequest, ChatResponse
from services.chat_service import get_ai_response, stream_ai_response

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer = await get_ai_response(request.message)
    return ChatResponse(answer=answer)

@router.post("/stream")
async def chat_stream(conversation_id: str, request: ChatRequest):
    async def event_generator():
        async for chunk in stream_ai_response(conversation_id, request.message):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")