from typing import Literal


# Simple shape for a single chat message (system/user/assistant + text),
# matching the OpenAI chat message format.
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
