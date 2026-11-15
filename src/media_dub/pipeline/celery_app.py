import os
import urllib.parse
from celery import Celery

RABBITMQ_USER = os.getenv("RABBITMQ_USER")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_HOST = os.getenv("IP")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT")

DB_IP = os.getenv("IP")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = urllib.parse.quote_plus(os.getenv("DB_PASSWORD"))
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"db+postgresql://{DB_USER}:{DB_PASSWORD}@{DB_IP}:{DB_PORT}/{DB_NAME}"

broker_url = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}"

WORKER_TYPE = os.getenv("WORKER_TYPE")

include_tasks = []

match WORKER_TYPE:
    case "voice-over":
        include_tasks = ["media_dub.pipeline.stage_voice_over.tasks"]
    case "translation":
        include_tasks = ["media_dub.pipeline.stage_translation.tasks"]
    case "audio-merge":
        include_tasks = ["media_dub.pipeline.stage_merge.tasks"]

app = Celery(
    "media_dub",
    broker=broker_url,
    backend=DATABASE_URL,
    include=include_tasks,
    broker_heartbeat=0,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    task_acks_late=True,
)
