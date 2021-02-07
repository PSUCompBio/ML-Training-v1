import os

from sqlalchemy import (Column, DateTime, Integer, MetaData, String, Table,
                        create_engine, ARRAY, FLOAT)


from databases import Database

DATABASE_URI = os.getenv('DATABASE_URI')

engine = create_engine(DATABASE_URI)
metadata = MetaData()


training_data = Table(
    'training_log',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('date_time',DateTime),
    Column('model_name',String(length=150)),
    Column('message',String(length=1000)),
    Column('training_status',String(length=100))
)

database = Database(DATABASE_URI)