"""Run the checked Friends S01 release through MediaDub with Parakeet v3 only."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
EPISODES = [f"S01E{i:02d}" for i in range(1, 25)]
V3 = "nvidia/parakeet-tdt-0.6b-v3"
MODEL = ROOT / "parakeet_test/parakeet-tdt-0.6b-v3.nemo"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def identity(path):
    info = path.stat()
    return {"path": str(path.resolve()), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def original_track(path, episode):
    streams = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index,codec_name,channels:stream_tags=language,title", "-of", "json", str(path)],
        text=True,
    ))["streams"]
    track = next((stream for stream in streams if stream["index"] == 2), {})
    tags = track.get("tags", {})
    language = tags.get("language")
    known_tag_error = episode == "S01E17" and language == "rus" and "Оригинал" in tags.get("title", "")
    if track.get("codec_name") != "ac3" or track.get("channels") != 6 or not (language == "eng" or known_tag_error):
        raise ValueError(f"{episode}: поток 0:2 отличается от проверенного оригинала: {path}")
    return track


def find_episodes(folder, selected):
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    files = {}
    for path in sorted(folder.glob("*.mkv")):
        match = re.search(r"\bS01E\d{2}\b", path.stem, re.IGNORECASE)
        if not match or match[0].upper() not in selected:
            continue
        episode = match[0].upper()
        if episode in files:
            raise ValueError(f"{episode}: найдено два видео")
        files[episode] = path
    missing = sorted(set(selected) - files.keys())
    if missing:
        raise ValueError(f"Отсутствуют серии: {', '.join(missing)}")
    return [(episode, files[episode], original_track(files[episode], episode)) for episode in selected]


def run_episode(episode, video, track, output, model):
    name = f"Friends.{episode}"
    folder = output / name
    source_file = folder / "friends_source.json"
    done = folder / "friends_complete.json"
    state = {"video": identity(video), "stream": track["index"], "asr": V3, "model": identity(model)}
    previous_input = source_file.is_file()
    if previous_input:
        if read_json(source_file) != state:
            raise ValueError(f"{episode}: изменился вход или модель. Укажи новую папку через --out.")
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"{folder}: есть старые результаты. Укажи новую папку через --out.")
        folder.mkdir(parents=True, exist_ok=True)
    if done.is_file() and all((folder / name).is_file() for name in ("lines.tsv", "markup.json", "speakers.tsv")):
        print(f"{episode}: готово, пропускаю", flush=True)
        return

    audio = output / "_audio" / f"{name}.flac"
    audio.parent.mkdir(parents=True, exist_ok=True)
    if not previous_input or not audio.is_file():
        partial = audio.with_suffix(".partial.flac")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video),
             "-map", f"0:{track['index']}", "-vn", "-ac", "2", "-ar", "44100", str(partial)],
            check=True,
        )
        partial.replace(audio)
    if not previous_input:
        save_json(source_file, state)

    print(f"{episode}: MediaDub + Parakeet v3", flush=True)
    env = dict(os.environ, MEDIADUB_PARAKEET_V3=str(model))
    subprocess.run(["bash", str(ROOT / "run-local.sh"), str(audio), "-o", str(output)],
                   env=env, check=True)
    profile = read_json(folder / "local_profile.json")
    if profile.get("asr_model") != V3:
        raise ValueError(f"{episode}: MediaDub не подтвердил запуск Parakeet v3")
    save_json(done, {"asr": V3})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder with all Friends S01 MKVs")
    parser.add_argument("--episode", choices=EPISODES, help="Run one episode instead of all 24")
    parser.add_argument("--out", type=Path, default=ROOT / "out/friends_s01_v3")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--dry-run", action="store_true", help="Verify the tracks without decoding")
    args = parser.parse_args()
    try:
        jobs = find_episodes(args.input.resolve(), [args.episode] if args.episode else EPISODES)
        for episode, _, track in jobs:
            language = track["tags"]["language"]
            note = " (исходная метка ошибочна)" if language == "rus" else ""
            print(f"{episode}: 0:{track['index']} · AC3 5.1 · {language}{note}", flush=True)
        if args.dry_run:
            return
        if not args.model.is_file() or args.model.name != MODEL.name:
            raise ValueError(f"Нужен файл {MODEL.name}. Проверь путь: {args.model}")
        model = args.model.resolve()
        local_stage = ROOT / "nemo-stage-local.py"
        if "MEDIADUB_PARAKEET_V3" not in local_stage.read_text(encoding="utf-8"):
            raise ValueError("Установи обновлённый nemo-stage-local.py с поддержкой Parakeet v3.")
        for episode, video, track in jobs:
            run_episode(episode, video, track, args.out.resolve(), model)
        print(args.out.resolve())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
