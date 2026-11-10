"""Local 4 GB profile: CPU restore, sequential models, BF16 ASR on CUDA."""
import gc
import json
import os
import sys
from pathlib import Path

import torch
import nemo_stage as stage


def release():
    gc.collect()
    torch.cuda.empty_cache()


def save_json(path, value):
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(value), encoding="utf-8")
    partial.replace(path)


def main():
    directories = [Path(path) for path in sys.argv[1:]]
    from nemo.collections.asr.models import SortformerEncLabelModel, ASRModel

    pending_diarization = [d for d in directories if not (d / "nemotron_w60.json").exists()]
    if pending_diarization:
        diar = SortformerEncLabelModel.from_pretrained(
            "nvidia/Nemotron-3-Diarization", map_location="cpu"
        ).eval().cuda()
        for directory in pending_diarization:
            with stage.tempfile.NamedTemporaryFile(suffix=".wav") as wav:
                stage.to_wav16k(directory / "vocals.wav", wav.name)
                duration = float(stage.subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", wav.name],
                    check=True, capture_output=True, text=True,
                ).stdout)
                turns = stage.nemotron_windows(diar, wav.name, duration)
                save_json(directory / "nemotron_w60.json", turns)
        del diar
        release()

    pending_asr = [d for d in directories if not (d / "words.json").exists()]
    if pending_asr:
        model_v3 = os.environ.get("MEDIADUB_PARAKEET_V3")
        if model_v3:
            asr = ASRModel.restore_from(model_v3, map_location="cpu")
            model_id = "nvidia/parakeet-tdt-0.6b-v3"
        else:
            model_id = "nvidia/parakeet-tdt-0.6b-v2"
            asr = ASRModel.from_pretrained(model_id, map_location="cpu")
        print(f"ASR: {model_id}", flush=True)
        asr = asr.eval().to(device="cuda", dtype=torch.bfloat16)
        # NeMo casts encoder output to FP32 before TDT decoding, outside autocast.
        # Keep the small decoder/joint in FP32 while the large encoder stays BF16.
        asr.decoder.float()
        asr.joint.float()
        asr.change_attention_model("rel_pos_local_attn", [256, 256])
        asr.change_subsampling_conv_chunking_factor(1)
        for directory in pending_asr:
            try:
                words = stage.transcribe(asr, directory)
            except RuntimeError as exc:
                if not any(text in str(exc).lower() for text in ("dtype", "out of memory")):
                    raise
                print(f"GPU ASR unavailable for this input; retrying FP32 on CPU: {exc}", flush=True)
                asr.cpu().float()
                release()
                words = stage.transcribe(asr, directory)
            save_json(directory / "local_profile.json", {
                "asr_model": model_id,
                "asr_device": str(next(asr.parameters()).device),
                "asr_encoder_dtype": str(next(asr.encoder.parameters()).dtype),
                "sequential_models": True,
                "source_commit": "6f23433fdc884c045aebeea636fb014a6f3db773",
            })
            save_json(directory / "words.json", words)
        del asr
        release()


if __name__ == "__main__":
    main()
