import os
import time

import media_dub.common.database as db
from media_dub.common.s3_sevice import (
    generate_temporary_url,
    get_s3_client,
    load_text_from_s3,
    save_text_to_s3,
)
from media_dub.pipeline.celery_app import app
from media_dub.pipeline.stage_voice_over.tasks import voice_over_task

if __name__ == "__main__":
    db.init_db()
    s3 = get_s3_client()


    with db.get_db_session() as session:
        id = db.insert_job(session)
        seg_id = db.insert_segment(session, id, 1234, 6214)
        seg_stage_res = db.insert_segment_stage(session, "Voice over", seg_id)


    s3.upload_file("./examples/sample_audio.wav", "english-segment-voice", f"{seg_id}.wav")

    save_text_to_s3("russian-text", f"{seg_id}.txt", "Привет! Озвучь меня пожалуйста! Спасибо!")

    print("Uploaded everything")

    voice_over_task.apply_async(
        args=[seg_id], queue="voice_over_queue"
    )


    print("Submitted job")

    url = generate_temporary_url("russian-segment-voice", f"{seg_id}.wav")

    print(url)
