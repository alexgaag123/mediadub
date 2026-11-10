# /// script
# requires-python = ">=3.12"
# dependencies = ["jiwer>=4,<5", "srt>=3.5,<4", "pyannote.metrics>=4,<5"]
# ///
"""Score completed Friends episodes against subtitles or a manually annotated sample."""

import argparse
import csv
from collections import Counter, defaultdict
from html import unescape
import json
from math import isfinite
from pathlib import Path
import re
import subprocess
import unicodedata

import jiwer


EPISODES = [f"S01E{i:02d}" for i in range(1, 25)]
CHAR_COUNTS = ("ref_chars", "substitutions", "deletions", "insertions")
SPEECH_COUNTS = ("total", "missed detection", "false alarm", "confusion")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]|_", " ", re.sub(r"['’]", "", text)).split())


def subtitle_text(video):
    import srt

    tracks = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
         "stream=index:stream_tags=language", "-of", "json", str(video)], text=True,
    ))["streams"]
    if not any(t["index"] == 4 and t.get("tags", {}).get("language") == "eng" for t in tracks):
        raise ValueError(f"{video.name}: не найден проверенный английский поток субтитров 0:4")
    data = subprocess.check_output(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-map", "0:4", "-f", "srt", "pipe:1"],
    ).decode("utf-8-sig")
    lines = []
    for cue in srt.parse(data):
        text = re.sub(r"<[^>]*>|\{\\[^}]*\}", "", unescape(cue.content))
        text = re.sub(r"\[[^\]]*\]", "", text)
        text = re.sub(r"(?m)^\s*-?\s*[A-Z][A-Z .'-]{0,35}:\s*", "", text)
        lines.append(text)
    return " ".join(lines)


def cer_counts(reference, hypothesis):
    score = jiwer.process_characters(normalize(reference), normalize(hypothesis))
    return Counter(ref_chars=score.hits + score.substitutions + score.deletions,
                   substitutions=score.substitutions, deletions=score.deletions, insertions=score.insertions)


def cer_row(episode, metric, counts):
    ref = counts["ref_chars"]
    errors = sum(counts[key] for key in CHAR_COUNTS[1:])
    return {"episode": episode, "metric": metric, "value_pct": 100 * errors / ref if ref else None,
            **{key: counts[key] for key in CHAR_COUNTS}}


def result_dir(root, episode):
    folder = root / f"Friends.{episode}"
    if not (folder / "friends_complete.json").is_file():
        raise ValueError(f"{episode}: прогон MediaDub ещё не завершён")
    return folder


def score_subtitles(args):
    rows, total, skipped = [], Counter(), []
    for episode in ([args.episode] if args.episode else EPISODES):
        if not (args.results / f"Friends.{episode}" / "friends_complete.json").is_file():
            skipped.append(episode)
            continue
        folder = result_dir(args.results, episode)
        videos = list(args.videos.glob(f"*.{episode}.*.mkv"))
        if len(videos) != 1:
            raise ValueError(f"{episode}: ожидается один MKV в {args.videos}, найдено {len(videos)}")
        reference = subtitle_text(videos[0])
        if not normalize(reference):
            raise ValueError(f"{episode}: субтитры пустые после нормализации")
        hypothesis = " ".join(w["w"] for w in load_json(folder / "words.json"))
        counts = cer_counts(reference, hypothesis)
        total.update(counts)
        rows.append(cer_row(episode, "CER_subtitles", counts))
    if skipped:
        print(f"Серий без завершённого прогона: {len(skipped)} ({', '.join(skipped)})")
    if not rows:
        raise ValueError("Пока нет завершённых серий для оценки.")
    return [*rows, cer_row("TOTAL", "CER_subtitles", total)]


def read_intervals(path, field, *, optional=False):
    intervals = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"episode", "start", "end"}
        if not optional:
            required.add(field)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: нужны колонки {', '.join(sorted(required))} с разделителем табуляцией")
        for row in reader:
            episode = row["episode"].strip()
            start, end = (float(row[key].replace(",", ".")) for key in ("start", "end"))
            if episode not in EPISODES or not all(isfinite(t) for t in (start, end)) or not 0 <= start < end:
                raise ValueError(f"{path}, строка {reader.line_num}: проверь серию и таймкоды")
            value = (row.get(field) or "").strip()
            if not optional and not value:
                raise ValueError(f"{path}, строка {reader.line_num}: заполни {field}")
            intervals[episode].append((start, end, value))
    return intervals


def annotation(turns, episode):
    from pyannote.core import Annotation, Segment

    result = Annotation(uri=episode)
    for track, (start, end, speaker) in enumerate(turns):
        if end > start:
            result[Segment(start, end), track] = speaker
    return result.support()


def der_row(episode, name, collar, counts):
    total = counts["total"]
    errors = sum(counts[key] for key in SPEECH_COUNTS[1:])
    return {"episode": episode, "metric": f"DER_{name}", "collar": collar,
            "value_pct": 100 * errors / total if total else None,
            "speech_s": total, "miss_s": counts["missed detection"],
            "fa_s": counts["false alarm"], "confusion_s": counts["confusion"]}


def ser_row(episode, name, collar, counts):
    detected = counts["total"] - counts["missed detection"]
    return {"episode": episode, "metric": f"SER_{name}", "collar": collar,
            "value_pct": 100 * counts["confusion"] / detected if detected > 0 else None,
            "detected_s": detected, "confusion_s": counts["confusion"]}


def score_manual(args):
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    regions = read_intervals(args.regions, "text", optional=True)
    speakers = read_intervals(args.speakers, "speaker")
    if not regions or not speakers or speakers.keys() != regions.keys():
        raise ValueError("Укажи сцены и хотя бы один эталонный речевой интервал для каждой размеченной серии.")
    rows, char_total = [], Counter()
    speech_totals = {0.0: Counter(), 0.25: Counter()}
    for episode, scenes in sorted(regions.items()):
        scenes.sort()
        if any(left[1] > right[0] for left, right in zip(scenes, scenes[1:])):
            raise ValueError(f"{episode}: сцены в regions.tsv пересекаются")
        if any(not any(a <= start < end <= b for a, b, _ in scenes) for start, end, _ in speakers[episode]):
            raise ValueError(f"{episode}: интервал говорящего выходит за пределы размеченной сцены")
        folder = result_dir(args.results, episode)
        words = load_json(folder / "words.json")
        counts = Counter()
        for start, end, reference in scenes:
            if not normalize(reference):
                continue
            text = " ".join(w["w"] for w in words if start <= (w["start"] + w["end"]) / 2 < end)
            counts.update(cer_counts(reference, text))
        if counts["ref_chars"]:
            char_total.update(counts)
            rows.append(cer_row(episode, "CER_manual", counts))

        turns = (load_json(folder / "turns.json") if args.hyp == "turns" else
                 [(s["start"], s["end"], s["speaker"]) for s in load_json(folder / "markup.json")["segments"]])
        reference = annotation(speakers[episode], episode)
        hypothesis = annotation(turns, episode)
        uem = Timeline([Segment(start, end) for start, end, _ in scenes], uri=episode)
        for collar, total in speech_totals.items():
            metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
            details = metric(reference, hypothesis, uem=uem, detailed=True)
            counts = {key: float(details[key]) for key in SPEECH_COUNTS}
            rows.append(der_row(episode, args.hyp, collar, counts))
            rows.append(ser_row(episode, args.hyp, collar, counts))
            total.update(counts)
    if char_total["ref_chars"]:
        rows.append(cer_row("TOTAL", "CER_manual", char_total))
    for collar, counts in speech_totals.items():
        rows.append(der_row("TOTAL", args.hyp, collar, counts))
        rows.append(ser_row("TOTAL", args.hyp, collar, counts))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("out/friends_s01_v3"))
    parser.add_argument("--report", type=Path)
    modes = parser.add_subparsers(dest="mode", required=True)
    subtitles = modes.add_parser("subtitles", help="CER against English subtitles")
    subtitles.add_argument("videos", type=Path)
    subtitles.add_argument("--episode", choices=EPISODES)
    manual = modes.add_parser("manual", help="CER and DER against manual references")
    manual.add_argument("regions", type=Path, help="TSV with episode, start, end and optional text")
    manual.add_argument("speakers", type=Path, help="TSV with episode, start, end, speaker")
    manual.add_argument("--hyp", choices=("turns", "markup"), default="turns")
    args = parser.parse_args()
    try:
        rows = score_subtitles(args) if args.mode == "subtitles" else score_manual(args)
        tag = (args.episode or "season") if args.mode == "subtitles" else args.hyp
        report = args.report or Path(f"out/friends_eval/{args.mode}_{tag}.csv")
        report.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with report.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            if row["episode"] == "TOTAL":
                value = "—" if row["value_pct"] is None else f'{row["value_pct"]:.2f}%'
                collar = f' (collar {row["collar"]:.2f} s)' if "collar" in row else ""
                print(f'{row["metric"]}{collar}: {value}')
        print(report.resolve())
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
