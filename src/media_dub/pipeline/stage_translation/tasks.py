import logging

from celery.signals import worker_process_init

import media_dub.common.database as db
from media_dub.common.s3_sevice import load_text_from_s3, save_text_to_s3
from media_dub.pipeline.celery_app import app


class TranslationWorkerModelNLLB200:
    def __init__(self):
        import ctranslate2
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        app.logger.info("Init nllb-200-1.3B-ct2 worker start")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "int8" if self.device == "cuda" else "default"

        model_dir = snapshot_download(
            repo_id="KomorebiAI/nllb-200-1.3B-ct2", allow_patterns=["*.bin", "*.json", "*.txt"]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, src_lang="eng_Latn", fix_mistral_regex=True
        )
        self.translator = ctranslate2.Translator(
            model_dir, device=self.device, compute_type=self.compute_type
        )
        app.logger.info("Init nllb-200-1.3B-ct2 worker finished")

    def translate(self, text: str) -> str:
        app.logger.info(f"Translate {text}")
        encoded_tokens = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        source_tokens = self.tokenizer.convert_ids_to_tokens(encoded_tokens)

        target_prefix = ["rus_Cyrl"]

        results = self.translator.translate_batch(
            [source_tokens], target_prefix=[target_prefix], beam_size=4
        )

        result = results[0]
        hyp = result.hypotheses[0]
        target_tokens = hyp[1:] if (len(hyp) > 0 and hyp[0] == "rus_Cyrl") else hyp

        result = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target_tokens)).strip()

        app.logger.info(f"Translated result {result}")

        return result


@worker_process_init.connect
def init_translation_model(**kwargs):
    app.logger = logging.getLogger(__name__)
    app.logger.info("Load NLLB-200...")
    app.translation_model = TranslationWorkerModelNLLB200()

    db.init_db()


@app.task(name="media_dub.stage_translation", queue="translation_queue", acks_late=True)
def translation_task(segment_id: int) -> None:
    try:
        worker_model = getattr(app, "translation_model", None)

        if worker_model is None:
            raise RuntimeError("Model was not initialised")

        downloaded_str = load_text_from_s3("english-text", str(segment_id) + ".txt")

        app.logger.info(f"Translate: {downloaded_str}")

        with db.get_db_session() as session:
            db.set_segment_stage_started_time(session, db.SegmentStageType.TRANSLATION, segment_id)
            db.update_segment_stage_status(session, db.SegmentStageType.TRANSLATION,
                                   segment_id, db.WorkStatus.IN_PROGRESS)

        translated_text = worker_model.translate(downloaded_str)

        app.logger.info(f"Result: {translated_text}")

        save_text_to_s3("russian-text", str(segment_id) + ".txt", translated_text)

        with db.get_db_session() as session:
            db.set_stage_finished_time(session, db.SegmentStageType.TRANSLATION, segment_id)
            db.update_segment_stage_status(session, db.SegmentStageType.TRANSLATION,
                                   segment_id, db.WorkStatus.FINISHED)


        return translated_text

    except Exception as e:
        app.logger.error(f"Error in translation: {e}")
        with db.get_db_session() as session:
            db.update_segment_stage_status(session, db.SegmentStageType.TRANSLATION,
                                   segment_id, db.WorkStatus.FAILED)

        raise e
