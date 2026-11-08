import os

from celery import Celery

RABBITMQ_USER = os.getenv("RABBITMQ_USER")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_HOST = os.getenv("IP")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT")

broker_url = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}"

app = Celery(
    "media_dub",
    broker=broker_url,
    include=[
        "media_dub.pipeline.stage_translation.tasks",
    ],
)
