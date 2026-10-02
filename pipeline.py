"""
Stereo (video/audio) → markup for translation and TTS.

  uv run pipeline.py <input> [-o out/] [--speakers N]

Stages (cached as files in out/<name>/; delete a file to recompute it):
  1. demix      Mel-RoFormer (becruily instrumental) → vocals.wav / background.wav (stereo, 44.1k)
  2. ASR input  asr_input.wav = voice + 10% background + dynaudnorm (background masks demix artifacts, AGC evens
                out whispers/shouts)
  3. NeMo stage (nemo_stage.py in .envs/nemo) words.json — Parakeet TDT v2; nemotron_w60.json — Nemotron on 60 s windows
  4. linking    local window speakers → characters: ReDimNet2 on all of a speaker's speech in the window, AHC that never
                merges within a window, k by silhouette (k ≥ max speakers per window) → turns.json. Lifts Nemotron's
                8-speaker ceiling.
  5. markup     word → speaker by overlap → lines (split on speaker change / pause)
  5a. refine    each line → nearest character by voice (ReDimNet2 line vs character centroid), words regrouped into
                lines. Fixes Nemotron errors within a window: SER AVA dev 0.141 → 0.119. The LLM variant
                (qwen3:8b on dialogue context, exp/llm_relabel.py) is no better than acoustics — not used.
  5b. video    (input with a video stream, or --video) faces + LR-ASD (av_stage.py in .envs/av); a line where one face confidently speaks
                moves to the character that face's OTHER lines mostly belong to (visual replaces audio where it is
                reliable, as in MERL AV-EEND late fusion; off-screen lines keep the audio decision)
  5c. edges     line edges extended to the enclosing Nemotron speech (≤0.25 s, never into a neighbour line)
  6. gender     audEERING wav2vec2 age-gender (CC-BY-NC-SA) on the speaker's ≤10 longest lines: female/male/child
  7. export     SPEAKER_xx/<id>.wav (line from vocals, mono 44.1k) + metadata.csv (LJSpeech "id|text"),
                lines.tsv (all lines), speakers.tsv (gender, age, amount of speech)

AVA-AVD, collar 0.25: ASR miss / FA on dev 0.182 / 0.139 (Whisper 0.208 / 0.206).
Full pipeline SER / DER: dev (37 films) 0.135 / 0.408, held-out test (4 films) 0.181 / 0.448
(v1 with Whisper + per-sentence embeddings: 0.227 / 0.609 dev, 0.359 / 0.712 test; whole-file Nemotron SER 0.180 dev).
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEMO_PY = ROOT / ".envs/nemo/bin/python"
SEP_MODEL = "mel_band_roformer_instrumental_becruily.ckpt"
EMB_MODEL = "Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM"
ASR_INPUT_FILTER = "[1]volume=0.1[b];[0][b]amix=inputs=2:normalize=0,dynaudnorm=f=150:g=15"
MAX_GAP = 0.6  # ponytail: pause (s) that breaks a line; tune to the editing pace
MAX_EMB_SEC = 30.0  # enough of a speaker's window speech for a voiceprint; longer only costs time
MAX_SPEAKERS = 30
GENDER_MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"
GENDER_LINES = 10  # longest lines per speaker are enough to average
CLIP_PAD = 0.05  # s of padding around a line clip (edges are already snapped to Nemotron speech)
SNAP = 0.25  # max s a line edge is extended to the enclosing Nemotron speech span (AVA dev DER 0.537 → 0.530)
AV_PY = ROOT / ".envs/av/bin/python"
FPS = 25
FACE_T = 0.7  # cosine distance for merging face tracks into one face id (ArcFace)
# per-line AV relabel (10 leakage-free AVA films: SER 0.171 → 0.156, leave-one-film-out picks the same setting)
AV_SHARE, AV_SUPPORT, AV_DOMINANCE = 0.3, 2, 0.6


def ffmpeg(src, dst, *args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", *args, str(dst)], check=True)


def separate(src: Path, d: Path):
    if (d / "vocals.wav").exists():
        return
    from audio_separator.separator import Separator
    mix = d / "mix.wav"
    ffmpeg(src, mix, "-ac", "2", "-ar", "44100")
    sep = Separator(output_dir=str(d), output_format="WAV")
    sep.load_model(SEP_MODEL)
    for f in sep.separate(str(mix)):
        stem = "vocals" if "(Vocals)" in f else "background"
        (d / f).rename(d / f"{stem}.wav")


def make_asr_input(d: Path):
    out = d / "asr_input.wav"
    if not out.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "vocals.wav"), "-i", str(d / "background.wav"),
                        "-filter_complex", ASR_INPUT_FILTER, "-ac", "1", "-ar", "16000", str(out)], check=True)


def load_16k(path: Path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def load_embedder():
    """ReDimNet2 via wespeaker. On import the package pulls in s3prl/whisper/w2vbert frontends; s3prl breaks on recent
    torchaudio (no sox_effects) — so the unused frontends are replaced with stubs. ReDimNet2 uses its own tfmel."""
    import types, torch
    from huggingface_hub import snapshot_download
    for mod, cls in [("s3prl", "S3prlFrontend"), ("whisper_encoder", "whisper_encoder"), ("w2vbert", "W2VBertFrontend")]:
        m = types.ModuleType(f"wespeaker.frontend.{mod}"); setattr(m, cls, None); sys.modules.setdefault(m.__name__, m)
    from wespeaker.cli.speaker import load_model_pt
    model = load_model_pt(snapshot_download(EMB_MODEL)).cuda().eval()

    @torch.inference_mode()
    def embed(x):
        w = torch.from_numpy(x.copy())[None].cuda()
        feat, _ = model.frontend(w, torch.LongTensor([w.shape[1]]).cuda())
        out = model(feat)
        return (out[-1] if isinstance(out, tuple) else out)[0].cpu().numpy()
    return embed


def local_speakers(turns):
    """{label: [(s, e), ...]} — a local speaker's speech without overlaps with other speakers of the same window."""
    by = {}
    for s, e, lab in turns:
        by.setdefault(lab, []).append((s, e))
    clean = {}
    for lab, segs in by.items():
        win = lab.split("_")[0]
        others = [p for l2, ss in by.items() if l2 != lab and l2.split("_")[0] == win for p in ss]
        out = []
        for s, e in segs:
            cur = [(s, e)]
            for os_, oe in others:
                cur = [p for a, b in cur for p in ((a, min(b, os_)), (max(a, oe), b)) if p[1] - p[0] > 0.05]
            out += cur
        clean[lab] = out or segs  # all speech overlaps — take it as is
    return clean


def constrained_ahc(X, cannot, k_min, k_max):
    """Average-linkage AHC on cosine distance; pairs with cannot[i, j] never end up in the same cluster.
    → (parts {k: labels} for k ≤ k_max, dist {k: distance of the merge that produced k clusters})."""
    import numpy as np
    D = 1 - X @ X.T
    clusters, parts, dist = [[i] for i in range(len(X))], {}, {}
    while True:
        if len(clusters) <= k_max:
            lab = np.empty(len(X), int)
            for c, mem in enumerate(clusters):
                lab[mem] = c
            parts[len(clusters)] = lab
        if len(clusters) <= k_min:
            break
        best, pair = np.inf, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if cannot[np.ix_(clusters[a], clusters[b])].any():
                    continue
                dd = D[np.ix_(clusters[a], clusters[b])].mean()
                if dd < best:
                    best, pair = dd, (a, b)
        if pair is None:  # no more merges allowed
            break
        clusters[pair[0]] += clusters.pop(pair[1])
        dist[len(clusters)] = best
    return parts, dist


def cannot_link(labs, G, p=None):
    """Must not merge: speakers of the same window (Nemotron already told them apart) and, if p is given, a confidently
    different gender (both ≥ p for their own female/male/child class)."""
    import numpy as np
    win = np.array([l.split("_")[0] for l in labs])
    C = win[:, None] == win[None]
    if p is not None:
        cls, conf = G.argmax(1), G.max(1) >= p
        C |= (cls[:, None] != cls[None]) & conf[:, None] & conf[None]
    np.fill_diagonal(C, False)
    return C


def local_embeddings(d: Path):
    """Cache local_emb.npz: local speaker labels, ReDimNet2 embeddings (normed) and gender probs from their speech."""
    import numpy as np
    out = d / "local_emb.npz"
    if out.exists():
        z = np.load(out)
        return list(z["labs"]), z["X"], z["G"]
    turns = json.loads((d / "nemotron_w60.json").read_text())
    audio, embed, gender = load_16k(d / "vocals.wav"), load_embedder(), load_gender()
    locs = local_speakers(turns)
    labs = sorted(locs)
    X, G = [], []
    for l in labs:
        x = np.concatenate([audio[int(s * 16000):int(e * 16000)] for s, e in locs[l]])[:int(MAX_EMB_SEC * 16000)]
        if len(x) < 16000:  # shorter than 1 s — pad by repetition so the frontend doesn't crash
            x = np.tile(x, int(np.ceil(16000 / max(len(x), 1))))[:16000]
        X.append(embed(x)); G.append(gender(x[:10 * 16000])[1])
    X = np.stack(X); X /= np.linalg.norm(X, axis=1, keepdims=True)
    np.savez(out, labs=np.array(labs), X=X, G=np.stack(G))
    return labs, X, np.stack(G)


def link(d: Path, speakers=None):
    """nemotron_w60.json → turns.json: local window speakers → global SPEAKER_xx."""
    out = d / "turns.json"
    if out.exists():
        return json.loads(out.read_text())
    from sklearn.metrics import silhouette_score
    labs, X, G = local_embeddings(d)
    C = cannot_link(labs, G)
    k_min = int(C.sum(1).max()) + 1
    parts, _ = constrained_ahc(X, C, k_min, min(MAX_SPEAKERS, len(labs)))
    ks = [k for k in parts if 2 <= k < len(labs)]
    if speakers and speakers in parts:
        k = speakers
    elif ks:
        k = max(ks, key=lambda k: silhouette_score(X, parts[k], metric="cosine"))
    else:
        k = min(parts)
    g = dict(zip(labs, parts[k]))
    res = [(s, e, f"SPEAKER_{g[l]:02d}") for s, e, l in json.loads((d / "nemotron_w60.json").read_text())]
    out.write_text(json.dumps(res))
    return res


def cluster_centroids(d: Path):
    """Character centroids: mean embedding of the local speakers assigned to the character in turns.json.
    → (characters, centroids)."""
    import numpy as np
    labs, X, _ = local_embeddings(d)
    idx, members = {l: i for i, l in enumerate(labs)}, {}
    for (_, _, loc), (_, _, g) in zip(json.loads((d / "nemotron_w60.json").read_text()), json.loads((d / "turns.json").read_text())):
        members.setdefault(g, set()).add(loc)
    chars = sorted(members)
    return chars, np.stack([(lambda v: v / np.linalg.norm(v))(X[[idx[l] for l in members[g]]].mean(0)) for g in chars])


def refine_lines(d: Path, lines):
    """Each line → nearest character by voice (ReDimNet2 line vs character centroid); words are regrouped into lines."""
    import numpy as np
    chars, M = cluster_centroids(d)
    if len(chars) < 2:
        return lines
    audio, embed = load_16k(d / "vocals.wav"), load_embedder()
    words = []
    for l in lines:
        s, e = l["start"], l["end"]
        if e - s < 1.0:  # shorter than 1 s — use a 1 s window around the center
            c = (s + e) / 2; s, e = c - 0.5, c + 0.5
        x = embed(audio[max(0, int(s * 16000)):int(e * 16000)])
        words += [dict(w, speaker=chars[int(np.argmax(M @ (x / np.linalg.norm(x))))]) for w in l["words"]]
    return group_lines(words)


def assign_speakers(words, turns):
    """Word speaker = turn with the largest overlap; with no overlap — the nearest one in time."""
    for w in words:
        best = max(turns, key=lambda t: (min(w["end"], t[1]) - max(w["start"], t[0]),
                                         -min(abs(w["start"] - t[1]), abs(w["end"] - t[0]))), default=None)
        w["speaker"] = best[2] if best else "SPEAKER_00"
    return words


def group_lines(words, max_gap=MAX_GAP):
    """Line = consecutive words of one speaker with no pause longer than max_gap."""
    lines = []
    for w in words:
        cur = lines[-1] if lines else None
        if cur and cur["speaker"] == w["speaker"] and w["start"] - cur["end"] <= max_gap:
            cur["words"].append(w); cur["end"] = w["end"]
        else:
            lines.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"], "words": [w]})
    for i, l in enumerate(lines):
        l["id"] = i
        l["text"] = " ".join(w["w"] for w in l["words"])
        for w in l["words"]:
            del w["speaker"]
    return [{k: l[k] for k in ("id", "speaker", "start", "end", "text", "words")} for l in lines]


def load_gender():
    """audEERING age-gender (model class from its card) → f(x 16 kHz) = (age 0..1, [p_female, p_male, p_child])."""
    import torch
    from torch import nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class Head(nn.Module):
        def __init__(self, config, n):
            super().__init__()
            self.dense, self.dropout = nn.Linear(config.hidden_size, config.hidden_size), nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, n)

        def forward(self, x):
            return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

    class AgeGender(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2, self.age, self.gender = Wav2Vec2Model(config), Head(config, 1), Head(config, 3)
            self.post_init()

        def forward(self, x):
            h = self.wav2vec2(x)[0].mean(1)
            return self.age(h), torch.softmax(self.gender(h), 1)

    proc = Wav2Vec2Processor.from_pretrained(GENDER_MODEL)
    model = AgeGender.from_pretrained(GENDER_MODEL).cuda().eval()

    @torch.inference_mode()
    def f(x):
        a, p = model(torch.from_numpy(proc(x, sampling_rate=16000).input_values[0])[None].cuda())
        return a.item(), p[0].cpu().numpy()
    return f


def face_ids(tracks, E, face_t=FACE_T):
    """Face tracks → face ids: average-linkage AHC on face embeddings (Lance–Williams updates), co-visible tracks
    never merged; stops when the closest allowed pair is farther than face_t (cosine distance)."""
    import numpy as np
    ok = [i for i, t in enumerate(tracks) if t["n_emb"] > 0]
    ids = {i: i for i in range(len(tracks))}
    if len(ok) < 2:
        return ids
    s = np.array([tracks[i]["f0"] for i in ok]); e = s + np.array([len(tracks[i]["scores"]) for i in ok])
    C = (s[:, None] < e[None]) & (s[None] < e[:, None])  # co-visible → cannot-link
    D = 1 - E[ok] @ E[ok].T
    n = len(ok)
    size, lab, alive = np.ones(n), np.arange(n), np.ones(n, bool)
    np.fill_diagonal(D, np.inf)
    while True:
        M = np.where(C | ~alive[None] | ~alive[:, None], np.inf, D)
        a, b = np.unravel_index(np.argmin(M), M.shape)
        if M[a, b] > face_t:
            break
        D[a] = D[:, a] = (size[a] * D[a] + size[b] * D[b]) / (size[a] + size[b])
        D[a, a] = np.inf
        C[a] = C[:, a] = C[a] | C[b]
        size[a] += size[b]; alive[b] = False; lab[lab == b] = a
    for p, i in enumerate(ok):
        ids[i] = 10_000 + int(lab[p])
    return ids


def line_faces(lines, tracks, fids, thr=0.0, share=AV_SHARE):
    """line index → face id speaking (LR-ASD logit > thr) during ≥ share of the line and ≥ 2× any other face."""
    speak = {}
    for i, t in enumerate(tracks):
        for j, s in enumerate(t["scores"]):
            if s > thr:
                speak.setdefault(t["f0"] + j, set()).add(fids[i])
    out = {}
    for k, l in enumerate(lines):
        frames = range(int(l["start"] * FPS), max(int(l["end"] * FPS), int(l["start"] * FPS) + 1))
        cnt = {}
        for f in frames:
            for fid in speak.get(f, ()):
                cnt[fid] = cnt.get(fid, 0) + 1
        if cnt:
            best = max(cnt, key=cnt.get)
            second = max((v for x, v in cnt.items() if x != best), default=0)
            if cnt[best] >= share * len(frames) and cnt[best] >= 2 * second:
                out[k] = best
    return out


def relabel(lines, lf, min_support=AV_SUPPORT, dominance=AV_DOMINANCE):
    """Face → character from the OTHER lines of that face (duration-weighted, leave-one-out); returns (lines, moved)."""
    by_face = {}
    for k, f in lf.items():
        by_face.setdefault(f, []).append(k)
    new, moved = [dict(l) for l in lines], 0
    for k, f in lf.items():
        votes = {}
        for o in by_face[f]:
            if o != k:
                votes[lines[o]["speaker"]] = votes.get(lines[o]["speaker"], 0) + lines[o]["end"] - lines[o]["start"]
        if len(by_face[f]) - 1 < min_support or not votes:
            continue
        x = max(votes, key=votes.get)
        if votes[x] >= dominance * sum(votes.values()) and x != lines[k]["speaker"]:
            new[k]["speaker"] = x; moved += 1
    return new, moved


def has_video(path: Path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_name",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip()) and r.stdout.split()[0] not in ("mjpeg", "png")  # cover art is not video


def video_relabel(d: Path, video: Path, lines):
    """av_stage.py (faces, tracks, LR-ASD) in .envs/av, then per-line relabel; words are regrouped into lines."""
    import numpy as np
    if not (d / "av_tracks.json").exists():
        subprocess.run([str(AV_PY), str(ROOT / "av_stage.py"), str(video), str(d)], check=True)
    tracks = json.loads((d / "av_tracks.json").read_text())
    new, _ = relabel(lines, line_faces(lines, tracks, face_ids(tracks, np.load(d / "av_emb.npy"))))
    return group_lines([dict(w, speaker=l["speaker"]) for l in new for w in l["words"]])


def snap_edges(d: Path, lines, max_ext=SNAP):
    """Extend line edges to the enclosing Nemotron speech span (≤ max_ext s), never past the neighbour line."""
    import bisect
    spans = []
    for s, e in sorted((s, e) for s, e, _ in json.loads((d / "nemotron_w60.json").read_text())):
        if spans and s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    starts = [s for s, _ in spans]
    out = []
    for i, l in enumerate(lines):
        s, e = l["start"], l["end"]
        prev_end = lines[i - 1]["end"] if i else 0.0
        next_start = lines[i + 1]["start"] if i + 1 < len(lines) else float("inf")
        j = bisect.bisect_right(starts, s) - 1
        if j >= 0 and spans[j][1] >= s:
            s = max(spans[j][0], s - max_ext, prev_end)
        k = bisect.bisect_right(starts, e) - 1
        if k >= 0 and spans[k][1] > e:
            e = min(spans[k][1], e + max_ext, next_start)
        out.append(dict(l, start=round(s, 3), end=round(e, 3)))
    return out


def speaker_gender(d: Path, lines):
    """{speaker: {"gender", "p", "age"}} — mean probabilities over the ≤GENDER_LINES longest lines."""
    import numpy as np
    gender, audio = load_gender(), load_16k(d / "vocals.wav")
    by = {}
    for l in lines:
        by.setdefault(l["speaker"], []).append(l)
    res = {}
    for spk, ls in by.items():
        ages, probs = [], []
        for l in sorted(ls, key=lambda l: l["end"] - l["start"], reverse=True)[:GENDER_LINES]:
            x = audio[int(l["start"] * 16000):int(l["end"] * 16000)]
            if len(x) < 8000:  # <0.5 s — not informative
                continue
            a, p = gender(x)
            ages.append(a); probs.append(p)
        if not probs:
            res[spk] = {"gender": "unknown", "p": 0.0, "age": None}
            continue
        p = np.mean(probs, 0)  # model class order: female, male, child
        i = int(p.argmax())
        res[spk] = {"gender": ["female", "male", "child"][i], "p": round(float(p[i]), 2), "age": round(float(np.mean(ages)) * 100)}
    return res


def export(d: Path, lines, genders):
    """Speaker folders with lines + metadata.csv (LJSpeech), lines.tsv, speakers.tsv."""
    import soundfile as sf
    audio, sr = sf.read(d / "vocals.wav", dtype="float32")
    audio = audio.mean(1) if audio.ndim == 2 else audio
    rows, stats = [], {}
    for l in lines:
        spk_dir = d / l["speaker"]
        spk_dir.mkdir(exist_ok=True)
        name = f"{l['id']:04d}.wav"
        a, b = max(0, int((l["start"] - CLIP_PAD) * sr)), int((l["end"] + CLIP_PAD) * sr)
        sf.write(spk_dir / name, audio[a:b], sr, subtype="PCM_16")
        text = " ".join(l["text"].split())
        with open(spk_dir / "metadata.csv", "a") as f:
            f.write(f"{name[:-4]}|{text}\n")
        g = genders[l["speaker"]]["gender"]
        rows.append(f"{l['id']}\t{l['speaker']}\t{g}\t{l['start']:.2f}\t{l['end']:.2f}\t{l['end'] - l['start']:.2f}\t{text}\t{l['speaker']}/{name}")
        n, t = stats.get(l["speaker"], (0, 0.0))
        stats[l["speaker"]] = (n + 1, t + l["end"] - l["start"])
    (d / "lines.tsv").write_text("id\tspeaker\tgender\tstart\tend\tduration\ttext\tfile\n" + "\n".join(rows) + "\n")
    (d / "speakers.tsv").write_text("speaker\tgender\tp_gender\tage\tlines\tspeech_sec\n" + "".join(
        f"{s}\t{g['gender']}\t{g['p']}\t{g['age']}\t{stats[s][0]}\t{stats[s][1]:.1f}\n" for s, g in sorted(genders.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--speakers", type=int, help="exact number of speakers, if known")
    ap.add_argument("--video", type=Path, help="video of the same work (same timeline) for per-line visual relabel; "
                    "default: the input itself if it has a video stream")
    a = ap.parse_args()
    d = a.out / a.input.stem
    d.mkdir(parents=True, exist_ok=True)

    separate(a.input, d)
    make_asr_input(d)
    if not ((d / "words.json").exists() and (d / "nemotron_w60.json").exists()):
        subprocess.run([str(NEMO_PY), str(ROOT / "nemo_stage.py"), str(d)], check=True)
    words = assign_speakers(json.loads((d / "words.json").read_text()), link(d, a.speakers))
    lines = refine_lines(d, group_lines(words))
    video = a.video or (a.input if has_video(a.input) else None)
    if video:
        lines = video_relabel(d, video, lines)
    lines = snap_edges(d, lines)
    genders = speaker_gender(d, lines)

    markup = {"source": str(a.input), "stems": {"vocals": "vocals.wav", "background": "background.wav"},
              "speakers": genders, "segments": lines}
    (d / "markup.json").write_text(json.dumps(markup, ensure_ascii=False, indent=1))
    import shutil
    for old in d.glob("SPEAKER_*"):  # a rerun must not append to old metadata.csv
        shutil.rmtree(old)
    export(d, lines, genders)
    for l in lines:
        print(f"[{l['start']:7.2f}-{l['end']:7.2f}] {l['speaker']} ({genders[l['speaker']]['gender']}): {l['text']}")


if __name__ == "__main__":
    main()
