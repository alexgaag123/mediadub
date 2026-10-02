# /// script
# dependencies = ["jiwer"]
# ///
"""Transcript WER (pipeline words.json) against .srt subtitles.

  uv run eval_wer.py out/<name>/words.json data/tos/TOS-en.srt
"""
import json, re, sys
import jiwer


def norm(s):
    s = s.lower().replace("’", "'").replace("thom", "tom")  # ponytail: the name is spelled differently in the SRT
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s).split())


hyp = norm(" ".join(w["w"] for w in json.load(open(sys.argv[1]))))
lines = open(sys.argv[2], encoding="utf-8-sig").read().splitlines()
ref = norm(" ".join(l for l in lines if "-->" not in l and not l.strip().isdigit()))
# ponytail: subtitles ≠ verbatim text (abbreviations, omissions) → WER here is an upper bound
o = jiwer.process_words(ref, hyp)
print(f"WER {o.wer:.3f}  sub {o.substitutions} del {o.deletions} ins {o.insertions}  ref {len(ref.split())} hyp {len(hyp.split())}")
if "-v" in sys.argv:
    print(jiwer.visualize_alignment(o))
