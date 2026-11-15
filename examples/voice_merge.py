import os
import time

import media_dub.common.database as db
from media_dub.common.s3_sevice import (
    generate_temporary_url,
    get_s3_client,
    save_text_to_s3,
)
from media_dub.pipeline.celery_app import app
from celery import signature, chain, chord


def start_dubbing_workflow(segment_ids: list[int]):
    pipelines = [
        chain(
            signature(
                "media_dub.stage_voice_over",
                kwargs={"segment_id": seg_id},
                queue="voice_over_queue",
                app=app
            ),
        )
        for seg_id in segment_ids
    ]

    merge_sig = signature(
        "media_dub.stage_merge",
        queue="merge_queue",
        app=app
    )

    return chord(pipelines)(merge_sig)


phrases = [
    {
        "text": "Hello, let me start",
        "start": 0,
        "finish": 5000 
    },
    {
        "text": "This is just a test",
        "start": 5000,
        "finish": 10000 
    },
    {
        "text": "You can change this",
        "start": 10000,
        "finish": 15000
    }
]


if __name__ == "__main__":
    db.init_db()
    s3 = get_s3_client()

    segment_list = []

    with db.get_db_session() as session:
        id = db.insert_job(session)
        job_stage_res = db.insert_job_stage(session, db.JobStageType.PROCESSING_SEGMENTS, id, db.WorkStatus.IN_PROGRESS)

        for phrase in phrases:
            seg_id = db.insert_segment(session, id, phrase["start"], phrase["finish"])
            seg_stage_res = db.insert_segment_stage(session, "Voice over", seg_id)
            s3.upload_file("./examples/sample_audio.wav", "english-segment-voice", f"{seg_id}.wav")
            save_text_to_s3("russian-text", f"{seg_id}.txt", phrase["text"])

            segment_list.append(seg_id)      

    print("Uploaded and submitted")

    future = start_dubbing_workflow(segment_list)
    
    print(f"Submitted job {id}")

    future.get()

    print(f"Job is finished, trying to get ")

    url = generate_temporary_url("merged-audio", f"{id}.wav")

    print(url)
