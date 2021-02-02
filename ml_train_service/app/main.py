from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.train_model import train

app = FastAPI(openapi_url="/api/train/openapi.json", docs_url="/api/train/docs")

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(train, prefix='/api/train', tags=['training ml model'])




