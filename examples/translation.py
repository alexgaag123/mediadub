import os
import time

import media_dub.common.database as db
from media_dub.common.s3_sevice import generate_temporary_url, save_text_to_s3
from media_dub.pipeline.celery_app import app
from celery import signature

if __name__ == "__main__":
    db.init_db()

    with open('./examples/test.txt', encoding='utf-8') as f:
        text = f.read()

    print("Sending text:")
    print(text)

    with db.get_db_session() as session:
        id = db.insert_job(session)
        seg_id = db.insert_segment(session, id, 123, 214)
        seg_stage_res = db.insert_segment_stage(session, "Translation", seg_id)


    save_text_to_s3("english-text", f"{seg_id}.txt", text)



    workflow = signature("media_dub.stage_translation",
                         queue="translation_queue",
                         kwargs={"segment_id": seg_id},
                         app=app)
    
    workflow.apply_async()

    print("submitted job")

    url = generate_temporary_url("english-text", f"{seg_id}.txt")

    print(url)
