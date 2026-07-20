import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Loads OPENAI_API_KEY (and any other secrets) from a local .env file.
# .env is gitignored -- never commit real keys.
load_dotenv()

# Shared async client so every service reuses one instance instead of each
# creating its own. Async because FastAPI route handlers are async and we
# don't want to block the event loop on network calls to OpenAI.
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
