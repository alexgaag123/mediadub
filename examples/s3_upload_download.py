import os

from media_dub.common.s3_sevice import get_s3_client
from media_dub.pipeline.celery_app import app

if __name__ == "__main__":

    s3 = get_s3_client()

    s3.upload_file("./examples/test.txt", "english-text", "test.txt")

    s3.download_file("english-text", "test.txt", "./examples/downaloaded_test.txt")
