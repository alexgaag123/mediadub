import numpy as np
from pipeline import assign_speakers, cannot_link, constrained_ahc, group_lines, local_speakers


def w(t, s, e, spk=None):
    return {"w": t, "start": s, "end": e, "p": 1.0, "speaker": spk}


def test_group_lines():
    words = [w("hi", 0.1, 0.5, "A"), w("there", 0.6, 1.0, "A"), w("far", 1.1, 1.5, "B"),
             w("gap", 1.6, 2.0, "B"), w("late", 6.0, 6.3, "B")]  # same B, but pause > MAX_GAP → new line
    lines = group_lines(words)
    assert [(l["speaker"], l["text"]) for l in lines] == [("A", "hi there"), ("B", "far gap"), ("B", "late")]
    assert lines[1]["start"] == 1.1 and lines[1]["end"] == 2.0


def test_assign_speakers():
    turns = [(0.0, 2.0, "A"), (2.0, 3.0, "B")]
    words = assign_speakers([w("x", 1.9, 2.5), w("y", 3.1, 3.3)], turns)  # more overlap with B; outside turns → nearest
    assert [x["speaker"] for x in words] == ["B", "B"]


def test_link_primitives():
    # 0,1 are similar but in the same window — must not merge; 2 is similar to 0 from another window
    X = np.array([[1, 0], [0.99, 0.14], [0.98, -0.2], [0, 1.0]]); X /= np.linalg.norm(X, axis=1, keepdims=True)
    G = np.array([[0.9, 0.1, 0], [0.9, 0.1, 0], [0.9, 0.1, 0], [0.1, 0.9, 0]])
    p, dist = constrained_ahc(X, cannot_link(["w0_a", "w0_b", "w1_a", "w1_b"], G), 2, 4)
    assert p[2][0] == p[2][2] and p[2][0] != p[2][1] and dist[3] < dist[2]
    # gender: 1 and 3 (confidently different gender) don't merge even from different windows
    C = cannot_link(["w0_a", "w1_a"], np.array([[0.95, 0.05, 0], [0.05, 0.95, 0]]), p=0.8)
    assert C[0, 1] and not cannot_link(["w0_a", "w1_a"], np.array([[0.6, 0.4, 0], [0.05, 0.95, 0]]), p=0.8)[0, 1]
    assert local_speakers([(0, 2, "w0_a"), (1, 3, "w0_b")]) == {"w0_a": [(0, 1)], "w0_b": [(2, 3)]}


def test_snap_edges(tmp_path=None):
    import json, tempfile
    from pathlib import Path
    from pipeline import snap_edges
    d = Path(tempfile.mkdtemp())
    (d / "nemotron_w60.json").write_text(json.dumps([[0.5, 2.0, "w0_a"], [1.8, 3.0, "w0_b"], [5.0, 6.0, "w0_a"]]))
    lines = [{"start": 1.0, "end": 1.5}, {"start": 1.6, "end": 2.9}, {"start": 5.5, "end": 5.6}]
    out = snap_edges(d, lines, max_ext=0.25)
    assert (out[0]["start"], out[0]["end"]) == (0.75, 1.6)  # ≤0.25 s, stops at the next line
    assert out[1]["end"] == 3.0 and out[2]["start"] == 5.25 and out[2]["end"] == 5.85


def test_chunking():
    from nemo_stage import gaps, merge_spans
    assert merge_spans([(0, 5), (8, 20), (25, 40), (41, 100)], 30) == [[0, 20], [25, 40], [41, 71], [71, 100]]
    assert gaps([(0, 10)], [[1, 4], [6, 9.8]]) == [(0, 1), (4, 6)]  # piece 9.8–10 < MIN_GAP is dropped


if __name__ == "__main__":
    test_group_lines(); test_assign_speakers(); test_link_primitives(); test_snap_edges(); test_chunking(); print("ok")
