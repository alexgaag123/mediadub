import logging
import tempfile

import media_dub.common.database as db
from celery.signals import worker_process_init
from media_dub.common.s3_sevice import get_s3_client, load_text_from_s3
from media_dub.pipeline.celery_app import app


class VoiceOverWorkerModelOmniVoice:
    def __init__(self):
        import torch
        from omnivoice import OmniVoice

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = OmniVoice.from_pretrained("drbaph/OmniVoice-bf16",
                                               device_map="cuda:0", dtype=torch.float16)

    def voice_over(self, path_to_reference: str, text: str, savepath: str, duration: int) -> str:
        import soundfile as sf

        generated_audio = self.model.generate(text=text, ref_audio=path_to_reference,
                                              duration=duration)
        sf.write(savepath, generated_audio[0], 24000)




@worker_process_init.connect
def init_translation_model(**kwargs):
    app.logger = logging.getLogger(__name__)
    app.logger.info("Load OmniVoice...")
    app.voice_over_model = VoiceOverWorkerModelOmniVoice()

    db.init_db()


@app.task(name="media_dub.stage_voice_over", queue="voice_over_queue", acks_late=True)
def voice_over_task(segment_id: int) -> None:
    try:
        worker_model = getattr(app, "voice_over_model", None)

        if worker_model is None:
            raise RuntimeError("Model was not initialised")

        downloaded_str = load_text_from_s3("russian-text", str(segment_id) + ".txt")

        s3 = get_s3_client()

        with tempfile.NamedTemporaryFile(mode="w+t", delete_on_close=True,
                                         suffix=".wav") as result_wav:

            with tempfile.NamedTemporaryFile(mode="w+t", delete_on_close=True,
                                             suffix=".wav") as reference_wav:
                s3.download_file("english-segment-voice", f"{segment_id}.wav", reference_wav.name)

                app.logger.info(f"Voice over: {downloaded_str}")

                with db.get_db_session() as session:
                    db.set_segment_stage_started_time(session, db.SegmentStageType.VOICE_OVER,
                                                      segment_id)
                    db.update_segment_stage_status(session, db.SegmentStageType.VOICE_OVER,
                                        segment_id, db.WorkStatus.IN_PROGRESS)
                    segment = db.get_segment_by_id(session, segment_id)
                    duration = segment.end_time - segment.start_time

                duration = float(duration) / 1000.0
                app.logger.info(f"Started audio generation for {downloaded_str}")
                worker_model.voice_over(reference_wav.name, downloaded_str,
                                        result_wav.name, duration)


            app.logger.info(f"Voice over was performed successfully: {downloaded_str}")

            s3 = get_s3_client()
            s3.upload_file(result_wav.name, "russian-segment-voice", f"{segment_id}.wav")

        with db.get_db_session() as session:
            db.set_stage_finished_time(session, db.SegmentStageType.VOICE_OVER, segment_id)
            db.update_segment_stage_status(session, db.SegmentStageType.VOICE_OVER,
                                   segment_id, db.WorkStatus.FINISHED)


        return segment_id

    except Exception as e:
        app.logger.error(f"Error in translation: {e}")
        with db.get_db_session() as session:
            db.update_segment_stage_status(session, db.SegmentStageType.VOICE_OVER,
                                   segment_id, db.WorkStatus.FAILED)

        raise e
