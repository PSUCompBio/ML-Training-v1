FROM python:3.6-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY . /app/

CMD uvicorn app.main:app --reload --host  0.0.0.0 --port 8000
