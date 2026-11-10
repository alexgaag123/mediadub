# /// script
# requires-python = ">=3.12"
# dependencies = ["scipy>=1.13,<2", "srt>=3.5,<4"]
# ///
"""Prepare MELD reference data for Friends and score utterance speaker attribution."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
from html import unescape
import json
from pathlib import Path
import re
from statistics import median
import subprocess

import srt


MELD_COMMIT = "e8cedf27b5d2877e198332c957127e16eb214afe"
SPLITS = ("train", "dev", "test")


def tokens(text):
    text = unescape(text).casefold().replace("’", "'").replace("'", "")
    return tuple(re.findall(r"[^\W_]+", text))


def seconds(value):
    hours, minutes, tail = value.strip().replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(tail)


def read_meld(directory, last_episode):
    episodes = defaultdict(list)
    for split in SPLITS:
        with (directory / f"{split}_sent_emo.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if int(row["Season"]) != 1 or not 1 <= int(row["Episode"]) <= last_episode:
                    continue
                start, end = seconds(row["StartTime"]), seconds(row["EndTime"])
                if 0 <= start < end and row["Speaker"].strip():
                    episode = f"S01E{int(row['Episode']):02d}"
                    episodes[episode].append({
                        "start": start, "end": end, "speaker": row["Speaker"].strip(),
                        "text": row["Utterance"], "dialogue": f"{split}:{row['Dialogue_ID']}",
                        "utterance": int(row["Utterance_ID"]),
                    })
    return episodes


def read_captions(video):
    raw = subprocess.check_output(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-map", "0:4",
         "-c:s", "copy", "-f", "srt", "pipe:1"],
    ).decode("utf-8-sig")
    cues = []
    for cue in srt.parse(raw):
        text = re.sub(r"<[^>]*>|\[[^\]]*\]", "", cue.content)
        text = re.sub(r"(?m)^\s*-?\s*[A-Z][A-Z .'-]{0,35}:\s*", "", text)
        cues.append((cue.start.total_seconds(), cue.end.total_seconds(), tokens(text)))
    return sorted(cues)


def align(rows, cues):
    by_text = defaultdict(list)
    for start, _, words in cues:
        by_text[words].append(start)
    frequency = Counter(tokens(row["text"]) for row in rows)
    candidates = [by_text[key][0] - row["start"] for row in rows if
                  (key := tokens(row["text"])) and len(" ".join(key)) >= 18
                  and frequency[key] == 1 and len(by_text[key]) == 1]
    offset = median(candidates) if candidates else 0.0
    anchors = [delta for delta in candidates if abs(delta - offset) <= .35]
    if len(anchors) < 3 or len(anchors) * 2 <= len(candidates):
        raise ValueError("Недостаточно совпавших субтитров для надёжной проверки таймкодов.")
    offset = median(anchors)
    starts = [start for start, _, _ in cues]
    for row in rows:
        start, end = (round(row[key] + offset, 3) for key in ("start", "end"))
        row.update(start=max(0, start), end=end)
        wanted = tokens(row["text"])
        window = [words for a, b, words in cues[:bisect_right(starts, end + .25)] if b >= start - .25]
        nearby = tuple(word for words in window for word in words)
        exact = bool(wanted) and any(nearby[i:i+len(wanted)] == wanted for i in range(len(nearby)))
        matching = sum(block.size for block in SequenceMatcher(None, wanted, nearby, autojunk=False).get_matching_blocks())
        row["text_verified"] = exact or (len(wanted) >= 5 and matching >= .85 * len(wanted))
        row["overlap"] = False
    ordered = sorted(rows, key=lambda row: row["start"])
    for i, row in enumerate(ordered):
        for other in ordered[i+1:]:
            if other["start"] >= row["end"] - .25:
                break
            overlap = min(row["end"], other["end"]) - other["start"]
            if overlap > .25 and row["speaker"] != other["speaker"]:
                row["overlap"] = other["overlap"] = True
    return {"offset_s": round(offset, 3), "anchors": len(anchors), "rows": rows}


def write_table(path, fields, rows, delimiter=","):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    references, videos, candidates = {}, {}, []
    for episode, rows in sorted(read_meld(args.meld_csv, args.last_episode).items()):
        matches = list(args.videos.glob(f"*.{episode}.*.mkv"))
        if len(matches) != 1:
            raise ValueError(f"{episode}: ожидается один MKV")
        video = videos[episode] = matches[0]
        references[episode] = align(rows, read_captions(video))
        dialogues = defaultdict(list)
        for row in rows:
            dialogues[row["dialogue"]].append(row)
        for dialogue in dialogues.values():
            start = max(0, min(r["start"] for r in dialogue) - .5)
            end = max(r["end"] for r in dialogue) + .5
            if (35 <= end - start <= 120 and sum(r["text_verified"] for r in dialogue) >= .8 * len(dialogue)
                    and len({r["speaker"] for r in dialogue}) >= 2):
                candidates.append((episode, round(start, 3), round(end, 3), dialogue))
        print(f"{episode}: {len(rows)} строк MELD, {sum(r['text_verified'] for r in rows)} подтверждены по тексту", flush=True)
    if not references:
        raise ValueError("В CSV не найден первый сезон.")

    selected, seen, duration = [], set(), 0.0
    for episode, start, end, dialogue in sorted(candidates, key=lambda item: item[1] - item[2]):
        if episode in seen:
            continue
        selected.append((episode, start, end, dialogue))
        seen.add(episode)
        duration += end - start
        if duration >= 300 and len(selected) >= 4:
            break
    if not selected:
        raise ValueError("Не нашлось целых диалогов для ручной проверки.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meld_reference.json").write_text(json.dumps(
        {"dataset": "declare-lab/MELD", "commit": MELD_COMMIT, "episodes": references},
        ensure_ascii=False, indent=2,
    ) + "\n", encoding="utf-8")
    clips = args.out / "audio"
    clips.mkdir(exist_ok=True)
    regions, speakers = [], []
    for episode, start, end, dialogue in sorted(selected):
        filename = f"{episode}_{start:.3f}-{end:.3f}.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", str(start), "-i", str(videos[episode]),
             "-t", f"{end-start:.3f}", "-map", "0:2", "-ac", "1", "-ar", "16000", str(clips / filename)],
            check=True,
        )
        regions.append(dict(episode=episode, start=start, end=end, audio=f"audio/{filename}"))
        for row in sorted(dialogue, key=lambda r: (r["start"], r["end"])):
            speakers.append(dict(row, episode=episode, clip_start=round(row["start"] - start, 3),
                                 clip_end=round(row["end"] - start, 3)))
    write_table(args.out / "regions_draft.tsv", ("episode", "start", "end", "audio"), regions, "\t")
    write_table(args.out / "speakers_draft.tsv",
                ("episode", "start", "end", "speaker", "clip_start", "clip_end", "text", "overlap", "text_verified"),
                speakers, "\t")
    print(f"{len(selected)} сцен, {duration:.1f} с · {args.out.resolve()}")


def combine_turns(turns):
    combined = defaultdict(list)
    for start, end, speaker in sorted(turns):
        if end <= start:
            continue
        spans = combined[speaker]
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return combined


def guess_speaker(row, turns):
    start, end = row["start"], row["end"]
    overlaps = Counter()
    for speaker, spans in turns.items():
        overlaps[speaker] = sum(max(0.0, min(b, end) - max(a, start)) for a, b in spans)
    ranked = overlaps.most_common(2)
    if not ranked or ranked[0][1] < min(.15, (end - start) / 2):
        return None, "no_speaker"
    if len(ranked) == 2 and ranked[0][1] - ranked[1][1] <= .08:
        return None, "ambiguous"
    return ranked[0][0], "assigned"


def map_speakers(rows):
    from scipy.optimize import linear_sum_assignment

    pairs = Counter((row["predicted_id"], row["speaker"]) for row in rows if row["predicted_id"])
    if not pairs:
        return {}
    ids = sorted({speaker for speaker, _ in pairs})
    names = sorted({name for _, name in pairs})
    weights = [[pairs[speaker, name] for name in names] for speaker in ids]
    row_ids, column_ids = linear_sum_assignment(weights, maximize=True)
    return {ids[i]: names[j] for i, j in zip(row_ids, column_ids) if weights[i][j]}


def summarize(episode, rows):
    counts = Counter(row["result"] for row in rows)
    total = sum(counts[key] for key in ("correct", "wrong", "no_speaker", "ambiguous"))
    return {"episode": episode, "evaluated": total,
            "accuracy_pct": round(100 * counts["correct"] / total, 2) if total else "",
            **{key: counts[key] for key in ("correct", "wrong", "no_speaker", "ambiguous", "text_unverified", "overlap")}}


def score(args):
    episodes = json.loads(args.reference.read_text(encoding="utf-8"))["episodes"]
    summary, details, skipped = [], [], []
    for episode, record in sorted(episodes.items()):
        folder = args.results / f"Friends.{episode}"
        if not (folder / "friends_complete.json").is_file():
            skipped.append(episode)
            continue
        turns = combine_turns(json.loads((folder / "turns.json").read_text(encoding="utf-8")))
        rows = []
        for row in record["rows"]:
            if not row["text_verified"]:
                predicted, status = None, "text_unverified"
            elif row["overlap"]:
                predicted, status = None, "overlap"
            else:
                predicted, status = guess_speaker(row, turns)
            rows.append(dict(row, episode=episode, predicted_id=predicted, result=status))
        mapping = map_speakers(rows)
        for row in rows:
            row["predicted_name"] = mapping.get(row["predicted_id"])
            if row["result"] == "assigned":
                row["result"] = "correct" if row["predicted_name"] == row["speaker"] else "wrong"
        summary.append(summarize(episode, rows))
        details.extend(rows)
    if skipped:
        print("Серии без завершённого прогона: " + ", ".join(skipped))
    total = summarize("TOTAL", details)
    if not total["evaluated"]:
        raise ValueError("Нет реплик для оценки в завершённых сериях.")
    args.out.mkdir(parents=True, exist_ok=True)
    write_table(args.out / "meld_speaker_summary.csv", tuple(total), [*summary, total])
    write_table(args.out / "meld_speaker_utterances.csv",
                ("episode", "dialogue", "utterance", "start", "end", "speaker", "predicted_id",
                 "predicted_name", "result", "text"), details)
    print(f"Точность назначения говорящего на репликах MELD: {total['accuracy_pct']:.2f}% "
          f"({total['correct']} / {total['evaluated']})")
    print(args.out.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("videos", type=Path)
    prep.add_argument("meld_csv", type=Path)
    prep.add_argument("--last-episode", type=int, choices=range(1, 25), default=12)
    prep.add_argument("--out", type=Path, default=Path("refs/friends_meld"))
    evaluate = commands.add_parser("score")
    evaluate.add_argument("reference", type=Path)
    evaluate.add_argument("--results", type=Path, default=Path("out/friends_s01_v3"))
    evaluate.add_argument("--out", type=Path, default=Path("out/friends_eval"))
    args = parser.parse_args()
    try:
        (prepare if args.command == "prepare" else score)(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
