import logging
import tempfile

from pydub import AudioSegment
from celery.signals import worker_process_init

import media_dub.common.database as db
from media_dub.common.s3_sevice import get_s3_client, generate_temporary_url
from media_dub.pipeline.celery_app import app


@worker_process_init.connect
def init(**kwargs):
    app.logger = logging.getLogger(__name__)
    db.init_db()

@app.task(name="media_dub.stage_merge", queue="merge_queue", acks_late=True)
def voice_over_task(segment_list: list[int]) -> int:

    with db.get_db_session() as session:
        job_id = db.get_segment_by_id(session, segment_list[0]).job_id
        
        # Todo verify segment list and crash if something wrong
        
        db.update_job_stage_status(session, db.JobStageType.MERGE,
                                   job_id, db.WorkStatus.IN_PROGRESS)
        db.set_job_stage_started_time(session, db.JobStageType.MERGE, job_id)
        
    try:

        s3 = get_s3_client()
        
        with db.get_db_session() as session:
            segments = []
            for seg_id in segment_list:
                segments.append(db.get_segment_by_id(session, seg_id))

            duration_ms = max(segments, key=lambda x: x.end_time).end_time

            final_track = AudioSegment.silent(duration=float(duration_ms), frame_rate=44100)


            for seg in segments:
                with tempfile.NamedTemporaryFile(mode="w+t", delete_on_close=True,
                                                 suffix=".wav") as seg_wav:
                    s3.download_file("russian-segment-voice", f"{seg.id}.wav", seg_wav.name)    

                    audio = AudioSegment.from_wav(seg_wav.name).set_frame_rate(44100)

                    app.logger.info(seg.to_dict())

                    final_track = final_track.overlay(audio, position=float(seg.start_time))


            with tempfile.NamedTemporaryFile(mode="w+t", delete_on_close=True,
                                             suffix=".wav") as final_wav:

                final_track.export(final_wav.name, format="wav")


                s3.upload_file(final_wav.name, "merged-audio", f"{job_id}.wav")

            db.update_job_stage_status(session, db.JobStageType.MERGE,
                                       job_id, db.WorkStatus.FINISHED)
            db.set_job_stage_finished_time(session, db.JobStageType.MERGE, job_id)

        return job_id

    except Exception as e:
        app.logger.error(f"Error in translation: {e}")
        with db.get_db_session() as session:
            db.update_job_stage_status(session, db.JobStageType.MERGE,
                                               job_id, db.WorkStatus.FAILED)

        raise e
