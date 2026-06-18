from typing import List, Dict, Optional

conversations: Dict[str, List[Dict[str, str]]] = {}

def get_conversation(conversation_id: str) -> Optional[List[Dict[str, str]]]:
    return conversations.get(conversation_id)

def save_conversation(conversation_id: str, messages: List[Dict[str, str]]):
    conversations[conversation_id] = messages

def append_message(conversation_id: str, message: Dict[str, str]):
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    conversations[conversation_id].append(message)
