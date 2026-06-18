from fastapi import FastAPI
from pydantic import BaseModel
from routers import users, utils, dbusers
from database import engine
from models.user import Base
from routers.chat import (
    router as chat_router
)

# uv run fastapi dev main.py 

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(chat_router)




