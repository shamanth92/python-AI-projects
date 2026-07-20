from fastapi import APIRouter, HTTPException
from schemas.chat import ChatRequest, ChatResponse
from services.langchain_rag_services.rag_retrieval_service import generate_answer

router = APIRouter(prefix="/langchain/chat/rag", tags=["langchain-rag-chat"])


# Single endpoint for both features of the LangChain pipeline: pass
# mode="qa" (default) to answer a question, or mode="exam" to generate
# practice exam questions on a topic. Same retrieval step either way --
# only the prompt used inside generate_answer() changes based on mode.
@router.post("/", response_model=ChatResponse)
async def rag_chat_endpoint(request: ChatRequest):
    try:
        answer = await generate_answer(request.message, request.mode)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
