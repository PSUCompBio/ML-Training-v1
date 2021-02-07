from .db import database, training_data


async def insert_log(logs):
    query = training_data.insert().values(**logs)
    return await database.execute(query=query)


async def update_log(training_id,logs):
    query = training_data.update().where(training_data.c.id == training_id).values(**logs)
    return await database.execute(query=query)


async def get_log(model_name):
    query = training_data.select().where(training_data.c.model_name == model_name).order_by(training_data.c.id.desc())
    return await database.fetch_one(query=query)

