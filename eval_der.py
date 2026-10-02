# /// script
# requires-python = ">=3.12"
# dependencies = ["pyannote.metrics>=4"]
# ///
"""Pipeline DER on AVA-AVD: markup.json of a 15-minute film chunk vs the RTTMs of its three 5-minute clips.

  uv run eval_der.py <ava_root> <out_dir> [--hyp NAME] [<id> ...]   # no id — all films with markup.json

--hyp NAME: hypothesis from out_dir/<id>/NAME.json = [[start, end, speaker], ...] (another system, e.g. Nemotron)
instead of the pipeline's markup.json.

Audio is cut from 900 s into the film, so hypothesis times are shifted by +900. Clip c_0N = [600+300N, 900+300N].
Labels are mapped separately in each clip (AVA-AVD has its own speaker labels per clip).
Prints DER and its parts: miss (speech ASR did not find), FA, confusion. For diarization decisions look at
SER = confusion / (total − miss): the share of speaker confusion on speech found by both us and the reference.
"""
import json, sys
from pathlib import Path
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

OFFSET = 900.0


def load_rttm(path):
    ann = Annotation()
    for line in open(path):
        f = line.split()
        ann[Segment(float(f[3]), float(f[3]) + float(f[4]))] = f[7]
    return ann


def hypothesis(markup):
    """Whole line (first→last word) — the unit TTS will voice."""
    ann = Annotation()
    for seg in markup["segments"]:
        if seg["end"] > seg["start"]:
            ann[Segment(seg["start"] + OFFSET, seg["end"] + OFFSET)] = seg["speaker"]
    return ann


def turns(path):
    ann = Annotation()
    for s, e, spk in json.load(open(path)):
        ann[Segment(s + OFFSET, e + OFFSET)] = spk
    return ann


def main():
    root, out, args = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    name = None
    if args[:1] == ["--hyp"]:
        name, args = args[1], args[2:]
    ids = args or sorted(p.parent.name for p in out.glob("*/markup.json"))
    metrics = {c: DiarizationErrorRate(collar=c, skip_overlap=False) for c in (0.0, 0.25)}
    for vid in ids:
        hyp = turns(out / vid / f"{name}.json") if name else hypothesis(json.load(open(out / vid / "markup.json")))
        for n in (1, 2, 3):
            rttm = root / "rttms" / f"{vid}_c_{n:02d}.rttm"
            if not rttm.exists():
                continue
            uem = Timeline([Segment(600 + 300 * n, 900 + 300 * n)])
            ref = load_rttm(rttm)
            for m in metrics.values():
                m(ref, hyp.crop(uem[0]), uem=uem)
    for c, m in metrics.items():
        r = abs(m)
        tot = m.accumulated_["total"]
        parts = {k: m.accumulated_[k] / tot for k in ("missed detection", "false alarm", "confusion")}
        ser = m.accumulated_["confusion"] / (tot - m.accumulated_["missed detection"])
        print(f"collar {c}: DER {r:.3f}  miss {parts['missed detection']:.3f}  FA {parts['false alarm']:.3f}  "
              f"confusion {parts['confusion']:.3f}  SER {ser:.3f}  ({tot / 3600:.2f} h speech, {len(ids)} films)")


if __name__ == "__main__":
    main()
