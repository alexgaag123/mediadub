"""Word-level diarization quality against a script with character names.

  uv run eval_diar.py out/<name>/markup.json data/tos/script.txt [-v]

Script lines are aligned to STT words (difflib) → each word gets a reference character.
WDER = share of matched words with the wrong speaker after the optimal 1:1 cluster→character mapping
(Shafey et al. 2019). Plus purity (a cluster doesn't mix characters — important for voice cloning)
and coverage (a character isn't split across clusters).
"""
import json, re, sys
from collections import Counter
from difflib import SequenceMatcher
import numpy as np
from scipy.optimize import linear_sum_assignment

ALIAS = {"ROBOT CELIA": "CELIA"}  # ponytail: same actress with voice processing; debatable, but one voice for dubbing


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower().replace("’", "'").replace("thom", "tom"))


def parse_script(path):
    """[(character, word)] from a pdftotext -layout script: ALL-CAPS name with ':' → indented line text."""
    out, spk = [], None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if m := re.fullmatch(r"([A-Z][A-Z .]+):(?: \(CONT\.\))?(?: \(cont’d\))?", s):
            spk = ALIAS.get(m[1], m[1]); continue
        if not s or not line.startswith("     ") or s.startswith("(") or spk is None:
            if not line.startswith("     "): spk = None  # an action line closes the dialogue line
            continue
        out += [(spk, t) for t in (norm(x) for x in re.sub(r"\([^)]*\)", " ", s).split()) if t]
    return out


def score(segments, script, verbose=False):
    words = [(seg["speaker"], norm(w["w"])) for seg in segments for w in seg["words"]]
    sm = SequenceMatcher(None, [w for _, w in words], [w for _, w in script], autojunk=False)
    pairs = [(words[b.a + k][0], script[b.b + k][0]) for b in sm.get_matching_blocks() for k in range(b.size)]

    hyps, refs = sorted({h for h, _ in pairs}), sorted({r for _, r in pairs})
    M = np.zeros((len(hyps), len(refs)), int)
    for h, r in pairs:
        M[hyps.index(h), refs.index(r)] += 1
    ri, ci = linear_sum_assignment(-M)
    n = len(pairs)
    wder = 1 - M[ri, ci].sum() / n
    purity = M.max(1).sum() / n   # cluster → dominant character
    coverage = M.max(0).sum() / n  # character → dominant cluster
    res = dict(wder=round(wder, 3), purity=round(purity, 3), coverage=round(coverage, 3), clusters=len(hyps))
    print(f"WDER {wder:.3f}  purity {purity:.3f}  coverage {coverage:.3f}  "
          f"clusters {len(hyps)} / chars {len(refs)}  matched {n}/{len(words)} words")
    if verbose:
        print("cluster      " + " ".join(f"{r[:7]:>7}" for r in refs))
        for i, h in enumerate(hyps):
            print(f"{h:12} " + " ".join(f"{v:7d}" if v else "      ." for v in M[i]))
    return res


def main():
    score(json.load(open(sys.argv[1]))["segments"], parse_script(sys.argv[2]), "-v" in sys.argv)


if __name__ == "__main__":
    main()
