"""NeMo stage of the pipeline (run from .envs/nemo, its own torch): for each film folder
  nemotron_w60.json Nemotron 3 Diarization on 60 s windows of vocals — local speakers "w{i}_{spk}"
  words_full.json   Parakeet TDT 0.6B v2 on the whole asr_input.wav — only used to find speech spans
  words.json        final words: WhisperX-style cut & merge — speech spans (Nemotron ∪ full-pass words) merged into
                    ≤30 s chunks regardless of pauses and decoded one by one (per-chunk feature normalization; the model
                    was trained on ≤40 s), then Nemotron speech still without words is re-decoded gap by gap
                    (filler-only gaps dropped).

AVA-AVD dev (37 films, collar 0.25), ASR miss / FA; ToS WER vs subtitles:
  whole file 0.182 / 0.139, 0.197;  τ=30 chunks 0.144 / 0.154, 0.188;  chunks + gap re-decode 0.120 / 0.168, 0.194.

  .envs/nemo/bin/python nemo_stage.py <out_dir> ...     # progress log: <first folder>/../nemo_stage.log
"""
import json, math, subprocess, sys, tempfile
from pathlib import Path
import torch
from tqdm import tqdm

WINDOW = 60  # ponytail: Nemotron's ≤8 speakers almost always suffice for 60 s; pipeline.link links the windows
CHUNK = 30.0  # max ASR chunk (s)
PAD = 0.3  # audio added around a chunk; also how far a word "covers" speech around it
MIN_GAP, GAP_CTX = 0.3, 0.2  # gaps shorter than MIN_GAP are ignored; GAP_CTX s of audio around a gap when re-decoding
FILLERS = {"mm", "mm-hmm", "mhm", "hmm", "hm", "um", "uh", "uh-huh", "ah", "oh", "ha", "haha", "huh", "eh", "ooh", "aah",
           "ugh", "whoa"}  # non-verbal only gaps stay in the background (not dubbed)


def to_wav16k(src, dst, *args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)], check=True)


def load_parakeet():
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").eval()
    m.change_attention_model("rel_pos_local_attn", [256, 256])  # full attention on 12+ min doesn't fit in 12 GB
    m.change_subsampling_conv_chunking_factor(1)
    return m


def parakeet_words(m, wav):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        r = m.transcribe([str(wav)], timestamps=True, batch_size=1, verbose=False)[0]
    return [{"w": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3), "p": None} for w in r.timestamp["word"]]


def load_nemotron():
    from nemo.collections.asr.models import SortformerEncLabelModel
    return SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization").eval()


def nemotron_windows(m, wav, duration):
    turns = []
    with tempfile.NamedTemporaryFile(suffix=".wav") as win:
        for i in range(math.ceil(duration / WINDOW)):
            to_wav16k(wav, win.name, "-ss", str(WINDOW * i), "-t", str(WINDOW))
            segs = m.diarize(audio=[win.name], batch_size=1, verbose=False)[0]
            turns += [(float(a) + WINDOW * i, float(b) + WINDOW * i, f"w{i}_{c}") for a, b, c in (str(x).split() for x in segs)]
    return turns


def covered(words, pad):
    """Speech spans covered by words (each word dilated by pad), merged."""
    out = []
    for w in words:
        s, e = w["start"] - pad, w["end"] + pad
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def merge_spans(spans, tau):
    """WhisperX cut & merge: a chunk grows while it stays ≤ tau; spans longer than tau are split at tau."""
    out = []
    for s, e in sorted(spans):
        while e - s > tau:
            out.append([s, s + tau]); s += tau
        if out and e - out[-1][0] <= tau:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def gaps(spans, cov):
    """Speech spans minus word coverage, pieces ≥ MIN_GAP."""
    out = []
    for s, e in sorted(spans):
        cur = [(s, e)]
        for cs, ce in cov:
            cur = [p for a, b in cur for p in ((a, min(b, cs)), (max(a, ce), b)) if p[1] - p[0] > 0.01]
        out += [p for p in cur if p[1] - p[0] >= MIN_GAP]
    return out


def decode_spans(m, wav_path, spans, pad, drop_fillers=False):
    words = []
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
        for s, e in spans:
            a = max(0.0, s - pad)
            to_wav16k(wav_path, wav.name, "-ss", f"{a:.3f}", "-t", f"{e + pad - a:.3f}")
            ws = parakeet_words(m, wav.name)
            if drop_fillers and all(w["w"].lower().strip(".,!?-") in FILLERS for w in ws):
                continue
            words += [dict(w, start=round(w["start"] + a, 3), end=round(w["end"] + a, 3)) for w in ws
                      if s - pad <= (w["start"] + w["end"]) / 2 + a < e + pad]
    return words


def transcribe(m, d: Path, diar_file="nemotron_w60.json"):
    nemo = [(s, e) for s, e, _ in json.loads((d / diar_file).read_text())]
    if not (d / "words_full.json").exists():
        (d / "words_full.json").write_text(json.dumps(parakeet_words(m, d / "asr_input.wav")))
    full = json.loads((d / "words_full.json").read_text())
    spans = nemo + [tuple(c) for c in covered(full, 0.0)]
    words = decode_spans(m, d / "asr_input.wav", merge_spans(spans, CHUNK), PAD)
    words += [dict(w, src="redecode") for w in
              decode_spans(m, d / "asr_input.wav", gaps(nemo, covered(words, PAD)), GAP_CTX, drop_fillers=True)]
    return sorted(words, key=lambda w: w["start"])


if __name__ == "__main__":
    dirs = [Path(d) for d in sys.argv[1:]]
    asr, diar = load_parakeet(), load_nemotron()
    log = open(dirs[0].parent / "nemo_stage.log", "a")
    for d in tqdm(dirs, file=log, desc="nemo_stage", mininterval=1):
        if not (d / "nemotron_w60.json").exists():
            with tempfile.NamedTemporaryFile(suffix=".wav") as v:
                to_wav16k(d / "vocals.wav", v.name)
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v.name],
                                           check=True, capture_output=True, text=True).stdout)
                (d / "nemotron_w60.json").write_text(json.dumps(nemotron_windows(diar, v.name, dur)))
        if not (d / "words.json").exists():
            (d / "words.json").write_text(json.dumps(transcribe(asr, d)))
        log.flush()
