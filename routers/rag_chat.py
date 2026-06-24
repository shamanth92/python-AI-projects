from fastapi import APIRouter, HTTPException
from schemas.chat import ChatRequest, ChatResponse
from services.rag_services.rag_chat_service import rag_chat

router = APIRouter(prefix="/chat/rag", tags=["rag-chat"])


@router.post("/", response_model=ChatResponse)
async def rag_chat_endpoint(request: ChatRequest):
    try:
        answer = await rag_chat(request.message)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
