"""Compare Community-1 with MediaDub using existing Parakeet v3 words."""

import argparse
from collections import Counter
from getpass import getpass
from itertools import groupby
import json
from pathlib import Path
import subprocess

from experiments.parakeet.compare_parakeet import write_report


MODEL = 'pyannote/speaker-diarization-community-1'
UNKNOWN = '?'
EPISODES = [f'S01E{i:02d}' for i in range(1, 19)]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_pipeline(device):
    import torch
    from huggingface_hub import get_token
    from pyannote.audio import Pipeline

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('PyTorch не видит CUDA. Проверь nvidia-smi и версию PyTorch.')
    token = get_token() or getpass('Hugging Face Read token: ')
    pipeline = Pipeline.from_pretrained(MODEL, token=token)
    if pipeline is None:
        raise RuntimeError('Не удалось загрузить Community-1. Проверь доступ к модели и токен Hugging Face.')
    pipeline.to(torch.device(device))
    return pipeline


def run_diarization(pipeline, path):
    import torch
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    pcm = subprocess.run(
        ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-vn',
         '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    if not pcm:
        raise ValueError(f'Пустая аудиодорожка: {path}')
    waveform = torch.frombuffer(bytearray(pcm), dtype=torch.float32).unsqueeze(0)
    with ProgressHook() as hook:
        return pipeline({'waveform': waveform, 'sample_rate': 16000}, hook=hook)


def annotation_turns(annotation):
    return [[float(segment.start), float(segment.end), speaker]
            for segment, _, speaker in annotation.itertracks(yield_label=True)]


def merge_same_speaker(turns):
    by_speaker = {}
    for start, end, speaker in sorted(turns):
        spans = by_speaker.setdefault(speaker, [])
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end, speaker])
    return sorted(span for spans in by_speaker.values() for span in spans)


def assign_words(words, turns):
    spans = merge_same_speaker(turns)
    assigned = []
    for word in words:
        start, end = word['start'], word['end']
        duration = end - start
        coverage = Counter()
        for left, right, speaker in spans:
            if left >= end:
                break
            if right > start:
                coverage[speaker] += min(end, right) - max(start, left)
        ranked = coverage.most_common(2)
        speaker = UNKNOWN
        if ranked and duration > 0:
            winner, overlap = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0
            if overlap >= .5 * duration and overlap - runner_up >= .2 * duration:
                speaker = winner
        assigned.append({**word, 'speaker': speaker})
    return assigned


def match_ids(baseline, candidate):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    left = sorted({word['speaker'] for word in baseline} - {UNKNOWN})
    right = sorted({word['speaker'] for word in candidate} - {UNKNOWN})
    if not left or not right:
        return {}
    weights = Counter()
    left_time, right_time = Counter(), Counter()
    for old, new in zip(baseline, candidate, strict=True):
        duration = old['end'] - old['start']
        left_time[old['speaker']] += duration
        right_time[new['speaker']] += duration
        weights[old['speaker'], new['speaker']] += duration

    def matching_weight(old, new):
        seconds = weights[old, new]
        minimum = max(1.0, .2 * left_time[old], .2 * right_time[new])
        return seconds if seconds >= minimum else 0

    matrix = np.array([[matching_weight(old, new) for new in right] for old in left])
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return {right[j]: {'mediadub_id': left[i], 'shared_word_seconds': round(float(matrix[i, j]), 3)}
            for i, j in zip(rows, columns, strict=True) if matrix[i, j] > 0}


def smooth_short_switches(words):
    """Bridge isolated A-B-A switches inside a sentence, keeping the original word labels."""
    smoothed = [word.copy() for word in sorted(words, key=lambda word: word['start'])]
    runs = [(speaker, list(group)) for speaker, group in groupby(smoothed, key=lambda word: word['speaker'])]
    bridges = {}
    for i in range(1, len(runs) - 1):
        previous, before = runs[i - 1]
        _, middle = runs[i]
        following, after = runs[i + 1]
        if previous == UNKNOWN or previous != following or middle[-1]['end'] - middle[0]['start'] > 2.0:
            continue
        neighbors = [before[-1], *middle, after[0]]
        if any(left['w'].rstrip().rstrip('"”’)]').endswith(('.', '!', '?', '…'))
               or right['start'] - left['end'] > 1.0
               for left, right in zip(neighbors, neighbors[1:])):
            continue
        bridges[i] = previous

    for i, speaker in bridges.items():
        if i - 1 in bridges or i + 1 in bridges:
            continue
        for word in runs[i][1]:
            word['speaker_before_smoothing'] = word['speaker']
            word['speaker'] = speaker
    return smoothed


def save_comparison(output, episode, vocals, baseline, candidate, mapping, mode):
    suffix = '' if mode == 'exclusive' else f'_{mode}'
    write_json(output / f'pyannote_words{suffix}.json', candidate)
    write_json(output / f'speaker_map{suffix}.json', mapping)
    differences = write_report(output, vocals, 15, episode, diarization=mode)
    print(f'{episode} ({mode}): {differences} из {len(baseline)} слов с разными метками после сопоставления. '
          f'Без ID: MediaDub {sum(w["speaker"] == UNKNOWN for w in baseline)}, '
          f'Community-1 {sum(w["speaker"] == UNKNOWN for w in candidate)}.', flush=True)


def compare(output, episode, vocals, baseline, turns, mode, *, smooth=False):
    candidate = assign_words(baseline, turns)
    mapping = match_ids(baseline, candidate)
    for word in candidate:
        raw = word['speaker']
        word['raw_speaker'] = None if raw == UNKNOWN else raw
        if raw in mapping:
            word['speaker'] = mapping[raw]['mediadub_id']
        elif raw != UNKNOWN:
            word['speaker'] = f'PYANNOTE:{raw}'
    if smooth:
        smoothed = smooth_short_switches(candidate)
        count = sum('speaker_before_smoothing' in word for word in smoothed)
        print(f'{episode}: сглаживание изменило ID у {count} слов.', flush=True)
        save_comparison(output, episode, vocals, baseline, smoothed, mapping, 'smoothed')
    save_comparison(output, episode, vocals, baseline, candidate, mapping, mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--episode', choices=EPISODES, default=EPISODES[0])
    choice.add_argument('--all', action='store_true', help='Run all 18 episodes')
    parser.add_argument('--asr-root', type=Path, default=Path('parakeet_test/ldr_season_v3'))
    parser.add_argument('--output', type=Path, default=Path('out/ldr_pyannote'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--compare-only', action='store_true', help='Build reports from saved turns without running Community-1')
    parser.add_argument('--smooth', action='store_true', help='Also report Community-1 with short A-B-A switches bridged')
    args = parser.parse_args()

    pipeline = None
    for episode in EPISODES if args.all else [args.episode]:
        try:
            source = args.asr_root / episode
            audio_dir = Path(read_json(source / 'source.json')['source']).parent
            vocals = audio_dir / 'vocals.wav'
            if not vocals.is_file():
                raise FileNotFoundError(vocals)
            baseline = read_json(source / 'v3.json')
            output = args.output / episode
            regular_path = output / 'turns.json'
            exclusive_path = output / 'turns_exclusive.json'
            if not args.compare_only:
                if pipeline is None:
                    pipeline = load_pipeline(args.device)
                print(f'{episode}: Community-1 ({args.device}) · {vocals}', flush=True)
                result = run_diarization(pipeline, vocals)
                output.mkdir(parents=True, exist_ok=True)
                write_json(regular_path, annotation_turns(result.speaker_diarization))
                write_json(exclusive_path, annotation_turns(result.exclusive_speaker_diarization))
            regular_turns, exclusive_turns = read_json(regular_path), read_json(exclusive_path)
            write_json(output / 'mediadub_words.json', baseline)
            compare(output, episode, vocals, baseline, regular_turns, 'regular', smooth=args.smooth)
            compare(output, episode, vocals, baseline, exclusive_turns, 'exclusive')
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
            parser.exit(1, f'{episode}: {error}\n')


if __name__ == '__main__':
    main()
