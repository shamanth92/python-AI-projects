from typing import Literal

class Message:
    role: Literal["system", "user", "assistant"]
    content: str