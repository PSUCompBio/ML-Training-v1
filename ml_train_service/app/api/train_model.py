from fastapi import APIRouter, HTTPException
from .run_sagemaker import create_and_evaluate_model
from fastapi import BackgroundTasks


train = APIRouter()

@train.post('/base-model',status_code=201)
async def predict_mps(background_tasks: BackgroundTasks):
    background_tasks.add_task(create_and_evaluate_model)
    return {"message":"Training Process started"}
