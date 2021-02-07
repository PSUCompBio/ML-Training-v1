from fastapi import APIRouter, HTTPException
from .run_sagemaker import create_and_evaluate_model
from fastapi import BackgroundTasks
from .schema import *
from .db_manager import get_log

train = APIRouter()

@train.post('/base-model',status_code=201)
async def train_mps():
    result = await create_and_evaluate_model()
    return result

@train.get('/recent-logs', response_model=LogOut)
async def get_trainig_logs(model_name: str):
    try:
        result = await get_log(model_name)
        if result:
            return result
        else:
            return {"message":"No data found"}
    except:
        return {"message": "Can't get result"}