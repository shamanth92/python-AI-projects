from fastapi import FastAPI
from pydantic import BaseModel
# from routers import users, utils, dbusers
# from database import engine
# from models.user import Base
from routers.chat import router as chat_router
from routers.ingestion import router as ingestion_router
from routers.rag_chat import router as rag_chat_router
from routers.langchain_ingestion import router as langchain_ingestion_router
from routers.langchain_rag_chat import router as langchain_rag_chat_router

# uv run fastapi dev main.py 

# Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(chat_router)
app.include_router(ingestion_router)
app.include_router(rag_chat_router)
app.include_router(langchain_ingestion_router)
app.include_router(langchain_rag_chat_router)




