import os

from celery import Celery

RABBITMQ_USER = os.getenv("RABBITMQ_USER")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_HOST = os.getenv("IP")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT")

broker_url = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}"

WORKER_TYPE = os.getenv("WORKER_TYPE")

include_tasks = []

match WORKER_TYPE:
    case "voice-over":
        include_tasks = ["media_dub.pipeline.stage_voice_over.tasks"]
    case "translation":
        include_tasks = ["media_dub.pipeline.stage_translation.tasks"]

app = Celery(
    "media_dub",
    broker=broker_url,
    include=include_tasks
)
