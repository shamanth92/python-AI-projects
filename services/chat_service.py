from services.openai_client import client
from services.conversation_service import get_conversation, append_message

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a friendly, concise AI chatbot. Answer directly in 2-4 sentences."
}

async def get_ai_response(message: str) -> str:
    response = await client.responses.create(model="gpt-5-mini", input=message)
    return response.output_text

async def stream_ai_response(conversation_id: str, message: str):
    history = get_conversation(conversation_id)
    if history is None:
        history = [SYSTEM_PROMPT]
    
    append_message(conversation_id, {"role": "user", "content": message})
    
    messages = get_conversation(conversation_id)
    
    stream = await client.responses.create(
        model="gpt-5-mini",
        input=messages,
        stream=True
    )
    
    full_response = ""
    async for event in stream:
        if event.type == "response.output_text.delta":
            full_response += event.delta
            yield event.delta
    
    append_message(conversation_id, {"role": "assistant", "content": full_response})