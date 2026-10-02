# /// script
# requires-python = ">=3.12"
# dependencies = ["pyannote.metrics>=4"]
# ///
"""ASR misses and hallucinations on AVA-AVD: where the model output words (words joined across pauses ≤ MAX_GAP) vs
reference RTTM speech. No text: AVA-AVD has no transcripts.

  uv run eval_asr.py <ava_root> <out_dir> <words_file> <id> ...     # words_file: words.json | words_parakeet_vocals.json ...
"""
import json, sys
from pathlib import Path
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.detection import DetectionErrorRate
import eval_der as D

MAX_GAP = 0.6  # same as pipeline.MAX_GAP


def speech(words):
    ann, cur = Annotation(), None
    for w in words:
        if cur and w["start"] - cur[1] <= MAX_GAP:
            cur[1] = max(cur[1], w["end"])
        else:
            if cur: ann[Segment(cur[0] + D.OFFSET, cur[1] + D.OFFSET)] = "speech"
            cur = [w["start"], w["end"]]
    if cur: ann[Segment(cur[0] + D.OFFSET, cur[1] + D.OFFSET)] = "speech"
    return ann


def main():
    root, out, name, ids = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    met = DetectionErrorRate(collar=0.25)
    for vid in ids:
        hyp = speech(json.load(open(out / vid / name)))
        for n in (1, 2, 3):
            rttm = root / "rttms" / f"{vid}_c_{n:02d}.rttm"
            if rttm.exists():
                uem = Timeline([Segment(600 + 300 * n, 900 + 300 * n)])
                met(D.load_rttm(rttm), hyp.crop(uem[0]), uem=uem)
    a = met.accumulated_
    print(f"{name}: miss {a['miss'] / a['total']:.3f}  FA {a['false alarm'] / a['total']:.3f}  "
          f"({a['total'] / 3600:.2f} h speech, {len(ids)} films, collar 0.25)")


if __name__ == "__main__":
    main()
