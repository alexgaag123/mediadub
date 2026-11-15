import os
import time

import media_dub.common.database as db
from media_dub.common.s3_sevice import (
    generate_temporary_url,
    get_s3_client,
    save_text_to_s3,
)
from media_dub.pipeline.celery_app import app
from celery import signature

if __name__ == "__main__":
    db.init_db()
    s3 = get_s3_client()


    with db.get_db_session() as session:
        id = db.insert_job(session)
        job_stage_res = db.insert_job_stage(session, db.JobStageType.PROCESSING_SEGMENTS, id, db.WorkStatus.IN_PROGRESS)
        seg_id = db.insert_segment(session, id, 10000, 15000)
        seg_stage_res = db.insert_segment_stage(session, "Voice over", seg_id)


    s3.upload_file("./examples/sample_audio.wav", "english-segment-voice", f"{seg_id}.wav")

    save_text_to_s3("russian-text", f"{seg_id}.txt", "Привет! Озвучь меня пожалуйста! Спасибо!")

    print(f"Uploaded everything, seg_id: {seg_id}")

    workflow = signature("media_dub.stage_voice_over",
                         queue="voice_over_queue",
                         kwargs={"segment_id": seg_id},
                         app=app)

    workflow.apply_async()

    print("Submitted job")

    url = generate_temporary_url("russian-segment-voice", f"{seg_id}.wav")

    print(url)
