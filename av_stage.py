"""Audio-visual stage (run from .envs/av): faces → tracks → LR-ASD speaking scores → face embeddings.

  .envs/av/bin/python av_stage.py <video> <film_dir> [<video> <film_dir> ...]   # progress log: av_stage.log in cwd

Per film writes <film_dir>/av_tracks.json ([{"f0": first frame, "scores": per-frame LR-ASD logit (>0 = speaking),
"n_emb": embeddings averaged}], 25 fps, time 0 = start of the film chunk) and av_emb.npy (L2-normalized ArcFace
mean embedding per track). Audio for ASD is the original mix (mix.wav): the model was trained on film audio.
Preprocessing follows LR-ASD's demo (third_party/LR-ASD/Columbia_test.py) but keeps crops in memory.
"""
import json, math, subprocess, sys
from pathlib import Path
import cv2, numpy as np, torch
import python_speech_features
from scipy import signal
from scipy.interpolate import interp1d
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/LR-ASD"))
FPS, HEIGHT = 25, 720
IOU_T, MAX_GAP, MIN_TRACK = 0.5, 10, 10  # as in LR-ASD's demo
CROP_SCALE = 0.40
EMB_EVERY = 5  # face embeddings every 5th frame of a detection
ASD_WEIGHTS = ROOT / "third_party/LR-ASD/weight/pretrain_AVA.model"


def frames(video, w, h):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", f"fps={FPS},scale={w}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def size(video):
    w0, h0 = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                      "stream=width,height", "-of", "csv=p=0", str(video)],
                                     check=True, capture_output=True, text=True).stdout.split(","))
    h = min(HEIGHT, h0 // 2 * 2)
    return int(round(w0 * h / h0 / 2)) * 2, h


def shots(video):
    from scenedetect import ContentDetector, detect
    return [int(round(s.get_seconds() * FPS)) for s, _ in detect(str(video), ContentDetector())]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def track(dets, cuts):
    """dets: per frame list of (bbox, emb|None). Greedy IoU tracking that never crosses a shot cut."""
    cutset, active, done = set(cuts), [], []
    for f, faces in enumerate(dets):
        if f in cutset:
            done += active; active = []
        active_keep = []
        for t in active:
            if f - t["frames"][-1] > MAX_GAP:
                done.append(t)
            else:
                active_keep.append(t)
        active, used = active_keep, set()
        for t in sorted(active, key=lambda t: -len(t["frames"])):
            best = max(((iou(t["boxes"][-1], b), i) for i, (b, _) in enumerate(faces) if i not in used), default=(0, None))
            if best[0] > IOU_T:
                used.add(best[1]); t["frames"].append(f); t["boxes"].append(faces[best[1]][0])
                if faces[best[1]][1] is not None:
                    t["embs"].append(faces[best[1]][1])
        for i, (b, e) in enumerate(faces):
            if i not in used:
                active.append({"frames": [f], "boxes": [b], "embs": [e] if e is not None else []})
    done += active
    out = []
    for t in done:
        if len(t["frames"]) < MIN_TRACK:
            continue
        fr, bx = np.array(t["frames"]), np.array(t["boxes"])
        full = np.arange(fr[0], fr[-1] + 1)
        boxes = np.stack([interp1d(fr, bx[:, j])(full) for j in range(4)], 1)
        s = signal.medfilt(np.maximum(boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]) / 2, 13)
        y = signal.medfilt((boxes[:, 1] + boxes[:, 3]) / 2, 13)
        x = signal.medfilt((boxes[:, 0] + boxes[:, 2]) / 2, 13)
        emb = None
        if t["embs"]:
            m = np.mean(t["embs"], 0); emb = m / np.linalg.norm(m)
        out.append({"f0": int(full[0]), "s": s, "x": x, "y": y, "emb": emb, "n_emb": len(t["embs"])})
    return out


def crop(img, s, x, y):
    bsi = int(s * (1 + 2 * CROP_SCALE))
    pad = np.pad(img, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
    my, mx = y + bsi, x + bsi
    face = pad[int(my - s):int(my + s * (1 + 2 * CROP_SCALE)), int(mx - s * (1 + CROP_SCALE)):int(mx + s * (1 + CROP_SCALE))]
    face = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2GRAY)
    return face[56:168, 56:168]


def asd_scores(model, mfcc, crops):
    """LR-ASD scoring averaged over window lengths, as in the demo."""
    length = min((mfcc.shape[0] - mfcc.shape[0] % 4) / 100, len(crops) / FPS)
    a, v = mfcc[:int(round(length * 100))], np.array(crops[:int(round(length * FPS))])
    runs = []
    for dur in (1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6):
        sc = []
        with torch.no_grad():
            for i in range(int(math.ceil(length / dur))):
                ia = torch.FloatTensor(a[i * dur * 100:(i + 1) * dur * 100]).unsqueeze(0).cuda()
                iv = torch.FloatTensor(v[i * dur * FPS:(i + 1) * dur * FPS]).unsqueeze(0).cuda()
                if ia.shape[1] == 0 or iv.shape[1] == 0:
                    continue
                out = model.model.forward_audio_visual_backend(model.model.forward_audio_frontend(ia),
                                                               model.model.forward_visual_frontend(iv))
                sc.extend(model.lossAV.forward(out, labels=None))
        runs.append(sc[:len(v)])
    n = min(len(r) for r in runs)
    res = np.mean([r[:n] for r in runs], 0) if n else np.zeros(0)
    return np.pad(res, (0, len(crops) - len(res)), constant_values=res[-1] if len(res) else -5.0)


def film(video, d, det, rec, asd):
    w, h = size(video)
    cuts = shots(video)
    dets = []
    for f, img in enumerate(frames(video, w, h)):
        boxes, kpss = det.detect(img, max_num=0)
        faces = []
        for b, k in zip(boxes, kpss if kpss is not None else [None] * len(boxes)):
            e = None
            if f % EMB_EVERY == 0 and k is not None and b[4] >= 0.6:
                from insightface.utils import face_align
                al = face_align.norm_crop(img, k)
                e = rec.get_feat(al).ravel(); e = e / np.linalg.norm(e)
            faces.append((b[:4].tolist(), e))
        dets.append(faces)
    tracks = track(dets, cuts)
    crops = [[] for _ in tracks]
    by_frame = {}
    for i, t in enumerate(tracks):
        for j in range(len(t["s"])):
            by_frame.setdefault(t["f0"] + j, []).append((i, j))
    for f, img in enumerate(frames(video, w, h)):
        for i, j in by_frame.get(f, []):
            t = tracks[i]
            crops[i].append(crop(img, t["s"][j], t["x"][j], t["y"][j]))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(d / "mix.wav"), "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    audio = np.frombuffer(raw, np.int16)
    res, embs = [], []
    for t, c in zip(tracks, crops):
        if not c:
            continue
        a0, a1 = int(t["f0"] / FPS * 16000), int((t["f0"] + len(c)) / FPS * 16000)
        mfcc = python_speech_features.mfcc(audio[a0:a1], 16000, numcep=13, winlen=0.025, winstep=0.010)
        sc = asd_scores(asd, mfcc, c)
        res.append({"f0": t["f0"], "scores": [round(float(x), 2) for x in sc], "n_emb": t["n_emb"]})
        embs.append(t["emb"] if t["emb"] is not None else np.zeros(512, np.float32))
    (d / "av_tracks.json").write_text(json.dumps(res))
    np.save(d / "av_emb.npy", np.stack(embs) if embs else np.zeros((0, 512), np.float32))
    return len(res)


if __name__ == "__main__":
    from insightface.app import FaceAnalysis
    from ASD import ASD
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    det, rec = app.det_model, app.models["recognition"]
    asd = ASD(); asd.loadParameters(str(ASD_WEIGHTS)); asd.eval()
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    for video, d in tqdm(pairs, file=open("av_stage.log", "a"), desc="av_stage", mininterval=1):
        d = Path(d)
        if not (d / "av_emb.npy").exists():
            film(Path(video), d, det, rec, asd)
