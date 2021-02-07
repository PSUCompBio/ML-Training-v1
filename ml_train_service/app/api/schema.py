from pydantic import BaseModel,Field
import datetime

class LogOut(BaseModel):
    id: int
    date_time: datetime.datetime
    model_name:str
    message: str
    training_status: str



