from pydantic import BaseModel
from enum import Enum

class ChatMode(str, Enum):
    qa = "qa"
    exam = "exam"

class ChatRequest(BaseModel):
    message: str
    mode: ChatMode = ChatMode.qa

class ChatResponse(BaseModel):
    conversation_id: str | None = None
    answer: str