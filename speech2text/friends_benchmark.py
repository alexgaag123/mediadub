'''Prepare complete Friends episodes and evaluate one standalone Parakeet model.'''

import argparse
import hashlib
import html
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


RATE = 16000
GIB = 1024 ** 3
TIME = r'(\d{2}):(\d{2}):(\d{2}),(\d{3})'
TIMING = re.compile('^' + TIME + r'\s+-->\s+' + TIME + r'(?:\s+.*)?$')


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(manifest, expected):
    actual = {episode['episode']: {
        'samples': episode['audio_samples'],
        'audio_sha256': episode['sha256']['audio.wav'],
        'captions_sha256': episode['sha256']['captions.json'],
    } for episode in manifest['episodes']}
    if actual != expected:
        raise ValueError('Configuration does not match the exact audio and caption files')


def limits():
    line = next(line for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    root = Path('/sys/fs/cgroup') / line.split('::', 1)[1].lstrip('/')
    values = {name: (root / name).read_text().strip() for name in ['memory.max', 'memory.swap.max', 'cpu.max']}
    quota, period = values['cpu.max'].split()
    if values['memory.max'] == 'max' or int(values['memory.max']) > 8 * GIB:
        raise RuntimeError('Run inside a systemd scope with MemoryMax=8G')
    if values['memory.swap.max'] != '0' or quota == 'max' or int(quota) > 2 * int(period):
        raise RuntimeError('Run with MemorySwapMax=0 and CPUQuota=200%')
    return values


def headroom(destination):
    if shutil.disk_usage('/mnt/c').free < 20 * GIB or shutil.disk_usage(destination).free < 8 * GIB:
        raise RuntimeError('Stopping: insufficient free space on Windows C or output filesystem')
    available = next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if int(available) * 1024 < 4 * GIB:
        raise RuntimeError('Stopping: less than 4 GiB available host memory')


def clean_caption(text):
    text = html.unescape(re.sub(r'<[^>]+>', '', text))
    text = re.sub(r'\[[^\]]*\]', '', text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r'^\s*[-–—]\s*', '', line)
        line = re.sub(r'^[A-Z](?:[A-Z0-9 .&\x27/–-]| and )*:\s*', '', line)
        lines.append(line.replace('♪', '').replace('♫', ''))
    return ' '.join(' '.join(lines).split())


def parse_srt(text):
    cues = []
    text = text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    for block in re.split(r'\n\s*\n', text.strip()):
        lines = block.splitlines()
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        match = TIMING.fullmatch(lines[0].strip()) if lines else None
        if not match:
            raise ValueError('Invalid subtitle block: ' + block[:100])
        numbers = list(map(int, match.groups()))
        def seconds(parts):
            hours, minutes, seconds, milliseconds = parts
            if minutes >= 60 or seconds >= 60:
                raise ValueError('Invalid subtitle timestamp')
            return 3600 * hours + 60 * minutes + seconds + milliseconds / 1000
        start, end = seconds(numbers[:4]), seconds(numbers[4:])
        if end <= start or (cues and start < cues[-1]['start']):
            raise ValueError('Unordered or nonpositive subtitle interval')
        raw = '\n'.join(lines[1:])
        cues.append({'start': start, 'end': end, 'raw_text': raw, 'text': clean_caption(raw)})
    if not cues:
        raise ValueError('Empty subtitles')
    return cues


def normalize(text):
    text = text.lower().replace('’', chr(39)).replace('‘', chr(39))
    return ' '.join(re.sub(r'[^a-z0-9\x27 ]+', ' ', text).split())


def audio_chunks(audio):
    start, maximum = 0, 60 * RATE
    window = RATE // 5
    while start < len(audio):
        end = min(start + maximum, len(audio))
        if end < len(audio):
            search = audio[end - 10 * RATE:end]
            energy = np.mean(search[:len(search) // window * window].reshape(-1, window) ** 2, axis=1)
            end = end - 10 * RATE + int(np.argmin(energy)) * window + window // 2
        yield start, end
        start = end


def retained_spans(samples, exclusions):
    spans, previous = [], 0
    for interval in exclusions:
        start, end = interval['start'], interval['end']
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in [start, end]):
            raise ValueError('Exclusion timestamps must be finite numbers')
        left, right = round(start * RATE), round(end * RATE)
        if not 0 <= previous <= left < right <= samples:
            raise ValueError('Exclusions must be ordered, disjoint and inside audio')
        if previous < left:
            spans.append((previous, left))
        previous = right
    if previous < samples:
        spans.append((previous, samples))
    if not spans:
        raise ValueError('Exclusions remove all audio')
    return spans


def retained_reference(cues, exclusions):
    texts = []
    for cue in cues:
        if not cue['text']:
            continue
        overlaps = [item for item in exclusions if cue['start'] < item['end'] and cue['end'] > item['start']]
        if overlaps and not any(item['start'] <= cue['start'] and cue['end'] <= item['end'] for item in overlaps):
            raise ValueError('Exclusion boundary crosses a spoken caption')
        if not overlaps:
            texts.append(cue['text'])
    return ' '.join(texts)


def english_stream(streams, kind):
    choices = [item for item in streams if item['codec_type'] == kind and item.get('tags', {}).get('language') == 'eng']
    if not choices and kind == 'audio':
        # S01E17 labels its explicitly named original soundtrack as Russian.
        choices = [item for item in streams if item['codec_type'] == kind and item.get('tags', {}).get('title', '').startswith('Оригинал ')]
    if len(choices) != 1:
        raise ValueError(f'Expected one identifiable English {kind} track')
    return choices[0]


def channel_filter(stream, center_channel):
    if not center_channel:
        return []
    if stream.get('channels') != 6 or stream.get('channel_layout') not in ['5.1', '5.1(side)']:
        raise ValueError('Center-channel experiment requires an explicit 5.1 layout with FC')
    return ['-af', 'pan=mono|c0=FC']


def fixed_chunks(parts, spans):
    boundaries, index = [], 0
    for left, right in spans:
        cursor = left
        while cursor < right:
            if index >= len(parts):
                raise ValueError('Fixed chunks do not cover retained audio')
            part = parts[index]
            if part['error'] is not None or not all(math.isfinite(part[key]) for key in ['start', 'end']):
                raise ValueError('Fixed chunks contain failures or invalid timestamps')
            start, end = round(part['start'] * RATE), round(part['end'] * RATE)
            if start != cursor or not start < end <= min(right, start + 60 * RATE):
                raise ValueError('Fixed chunks must cover each retained span exactly once')
            boundaries.append((start, end))
            cursor, index = end, index + 1
    if index != len(parts):
        raise ValueError('Fixed chunks contain excluded or extra audio')
    return boundaries


def prepare(source, destination, center_channel=False):
    import soundfile as sf

    source, destination = source.resolve(), destination.resolve()
    files = sorted(source.glob('*.mkv'))
    episodes = [re.search(r'S01E\d{2}', path.name).group() for path in files]
    if episodes != [f'S01E{index:02}' for index in range(1, 25)]:
        raise ValueError('Expected all 24 season 1 episodes, exactly once')
    destination.mkdir(parents=True, exist_ok=True)
    headroom(destination)
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'limits': limits(),
        'source': str(source), 'planned_episodes': episodes,
        'reference_status': 'Embedded English SDH, not independently verified verbatim.',
        'cleanup': 'Remove HTML formatting, square-bracket sound descriptions, uppercase speaker labels and music-note symbols; preserve spoken repetitions and all other words.',
        'audio': 'Explicit English track, standard mono downmix, 16 kHz float32 WAV; one episode at a time; FFmpeg threads=1.',
    }
    if center_channel:
        protocol['audio'] = 'Explicit English 5.1 FC channel only (pan=mono|c0=FC), no gain or denoising, 16 kHz float32 WAV; one episode at a time; FFmpeg threads=1.'
    protocol_file = destination / 'preparation.json'
    if not protocol_file.exists():
        dump(protocol_file, protocol)
    elif json.loads(protocol_file.read_text())['audio'] != protocol['audio']:
        raise ValueError('Existing preparation uses a different audio channel')
    records = []
    for path, episode in zip(files, episodes):
        headroom(destination)
        target = destination / episode
        if target.exists():
            record = json.loads((target / 'source.json').read_text())
            if record['source'] != str(path) or record['source_bytes'] != path.stat().st_size or record.get('audio_channel', 'mix') != ('FC' if center_channel else 'mix'):
                raise ValueError('Existing data belongs to another input')
            for name, checksum in record['sha256'].items():
                if sha256(target / name) != checksum:
                    raise ValueError('Existing output checksum mismatch: ' + str(target / name))
        else:
            info = subprocess.run([
                'ffprobe', '-v', 'error', '-threads', '1', '-show_entries',
                'format=duration,start_time:stream=index,codec_type,codec_name,channels,channel_layout,sample_rate,start_time:stream_tags=language,title',
                '-of', 'json', str(path),
            ], check=True, capture_output=True, text=True, timeout=30)
            info = json.loads(info.stdout)
            audio = english_stream(info['streams'], 'audio')
            filters = channel_filter(audio, center_channel)
            subtitles = english_stream(info['streams'], 'subtitle')
            if subtitles['codec_name'] != 'subrip':
                raise ValueError('Expected embedded text subtitles')
            before = path.stat()
            with TemporaryDirectory(prefix=episode + '_', dir=destination) as temporary:
                work = Path(temporary)
                subprocess.run([
                    'ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '1',
                    '-filter_threads', '1', '-i', str(path),
                    '-map', '0:' + str(audio['index']), '-vn', '-sn', '-threads', '1',
                    *filters,
                    '-ac', '1', '-ar', str(RATE), '-c:a', 'pcm_f32le', str(work / 'audio.wav'),
                    '-map', '0:' + str(subtitles['index']), '-vn', '-an', '-c:s', 'copy', str(work / 'captions.srt'),
                ], check=True, capture_output=True, text=True, timeout=300)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError('Source changed during extraction')
                wav = sf.info(work / 'audio.wav')
                duration = wav.frames / wav.samplerate
                if wav.samplerate != RATE or wav.channels != 1 or abs(duration - float(info['format']['duration'])) > 0.5:
                    raise ValueError('Audio duration or format does not match the episode')
                cues = parse_srt((work / 'captions.srt').read_text(encoding='utf-8-sig'))
                if max(cue['end'] for cue in cues) > duration + 0.5:
                    raise ValueError('Subtitles extend beyond audio')
                reference = ' '.join(cue['text'] for cue in cues if cue['text'])
                (work / 'reference.txt').write_text(reference + '\n', encoding='utf-8')
                dump(work / 'captions.json', cues)
                dump(work / 'streams.json', info)
                record = {
                    'episode': episode, 'source': str(path), 'source_bytes': before.st_size,
                    'audio': str(target / 'audio.wav'), 'reference': str(target / 'reference.txt'),
                    'captions': str(target / 'captions.json'), 'audio_seconds': duration,
                    'audio_samples': wav.frames, 'audio_stream': audio, 'subtitle_stream': subtitles,
                    'audio_channel': 'FC' if center_channel else 'mix',
                    'subtitle_cues': len(cues), 'speech_cues': sum(bool(cue['text']) for cue in cues),
                    'reference_words': len(normalize(reference).split()),
                    'sha256': {name: sha256(work / name) for name in ['audio.wav', 'captions.srt', 'reference.txt', 'captions.json']},
                }
                dump(work / 'source.json', record)
                work.rename(target)
        records.append(record)
        dump(destination / 'manifest.json', {'complete': len(records) == 24, 'reference_status': protocol['reference_status'], 'episodes': records})
        print(f'{episode}: {record['audio_seconds']:.1f}s, {record['reference_words']} reference words', flush=True)


def evaluate(dataset, output, exclusions_path=None, boundaries_from=None, boundaries_path=None):
    import jiwer
    import soundfile as sf
    import torch
    from .parakeet import MODEL, MODEL_DIR, MODEL_REVISION, load_model, transcribe_audio

    dataset, output = dataset.resolve(), output.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if not manifest['complete'] or len(manifest['episodes']) != 24:
        raise ValueError('The full 24-episode dataset must be prepared first')
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path else None
    if exclusions is not None:
        if 'inputs' in exclusions:
            validate_inputs(manifest, exclusions['inputs'])
        elif exclusions['dataset_sha256'] != sha256(dataset / 'manifest.json'):
            raise ValueError('Exclusions belong to a different dataset')
        if set(exclusions['episodes']) != {item['episode'] for item in manifest['episodes']}:
            raise ValueError('Exclusions must match this exact dataset and all episodes')
        for episode in manifest['episodes']:
            cuts = exclusions['episodes'][episode['episode']]
            retained_spans(episode['audio_samples'], cuts)
            if sha256(Path(episode['captions'])) != episode['sha256']['captions.json']:
                raise ValueError('Caption checksum changed after dataset freeze')
            retained_reference(json.loads(Path(episode['captions']).read_text()), cuts)
    fixed = {}
    if boundaries_path is not None:
        if boundaries_from is not None:
            raise ValueError('Choose one source of fixed boundaries')
        plan = json.loads(boundaries_path.read_text())
        validate_inputs(manifest, plan['inputs'])
        if set(plan['episodes']) != {item['episode'] for item in manifest['episodes']}:
            raise ValueError('Boundaries must cover all episodes')
        for episode in manifest['episodes']:
            cuts = exclusions['episodes'][episode['episode']] if exclusions is not None else []
            fixed[episode['episode']] = fixed_chunks(plan['episodes'][episode['episode']], retained_spans(episode['audio_samples'], cuts))
    if boundaries_from is not None:
        for episode in manifest['episodes']:
            target = boundaries_from / episode['episode']
            cuts = exclusions['episodes'][episode['episode']] if exclusions is not None else []
            expected_reference = retained_reference(json.loads(Path(episode['captions']).read_text()), cuts)
            if (target / 'reference.txt').read_text().strip() != expected_reference:
                raise ValueError('Fixed boundaries use a different reference or evaluation scope')
            parts = json.loads((target / 'transcript.json').read_text())['chunks']
            fixed[episode['episode']] = fixed_chunks(parts, retained_spans(episode['audio_samples'], cuts))
    output.mkdir(parents=True, exist_ok=False)
    headroom(output)
    weights_sha256 = sha256(MODEL_DIR / 'model.safetensors')
    if weights_sha256 != '3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11':
        raise ValueError('Model weights differ from the previously verified official checkpoint')
    protocol = {
        'frozen_at': datetime.now(timezone.utc).isoformat(), 'model': MODEL, 'revision': MODEL_REVISION,
        'dataset_sha256': sha256(dataset / 'manifest.json'), 'runner_sha256': sha256(Path(__file__)),
        'model_code_sha256': sha256(Path(__file__).with_name('parakeet.py')),
        'weights_sha256': weights_sha256,
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'jiwer', 'soundfile']},
        'limits': limits(), 'gpu_allocator_fraction': 0.5,
        'decoding': 'Native greedy, FP32, batch=1. At most 60 seconds per input, cuts at minimum 200ms RMS in seconds 50-60; no dropped or duplicated audio; no reference-based segmentation.',
        'normalization': 'Lowercase; curly apostrophes to ASCII; non-English-letter/digit/apostrophe characters to spaces; collapse spaces. No filler, contraction or number equivalences.',
        'selection': 'All 24 full episodes, including intro, music, silence and uncaptioned speech. No exclusions or retries.',
        'reference_status': manifest['reference_status'],
        'meaning': 'WER/CER relative to cleaned supplied SDH captions, not independently verified verbatim ASR accuracy.',
    }
    if exclusions is not None:
        protocol['exclusions'] = exclusions
        protocol['exclusions_sha256'] = sha256(exclusions_path)
        protocol['selection'] = 'All 24 episodes minus fixed approximate title intervals. Final dialogue scenes retained. Each retained span is decoded independently; original timestamps preserved. No retries.'
    if boundaries_from is not None:
        previous = json.loads((boundaries_from / 'protocol.json').read_text())
        for key in ['model', 'revision', 'weights_sha256', 'model_code_sha256', 'versions', 'normalization']:
            if previous[key] != protocol[key]:
                raise ValueError('Fixed-boundary comparison requires identical model, runtime and normalization')
        protocol['boundaries_from'] = str(boundaries_from.resolve())
        protocol['boundary_files_sha256'] = {episode['episode']: sha256(boundaries_from / episode['episode'] / 'transcript.json') for episode in manifest['episodes']}
        protocol['decoding'] = 'Native greedy, FP32, batch=1. Fixed sample boundaries reused from the supplied plan, at most 60 seconds; no dropped or duplicated retained audio.'
    if boundaries_path is not None:
        protocol['boundaries'] = plan
        protocol['boundaries_sha256'] = sha256(boundaries_path)
        protocol['decoding'] = 'Native greedy, FP32, batch=1. Exact frozen 585 FC benchmark boundaries; at most 60 seconds, no dropped or duplicated retained audio.'
    dump(output / 'protocol.json', protocol)
    frozen = output / 'frozen_source'
    frozen.mkdir()
    for name in ['friends_benchmark.py', 'parakeet.py']:
        shutil.copy2(Path(__file__).with_name(name), frozen / name)
    headroom(output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.5)
    if torch.cuda.mem_get_info()[0] < 4 * GIB:
        raise RuntimeError('Stopping: less than 4 GiB free GPU memory')
    engine = load_model()
    rows, started = [], time.monotonic()
    for episode in manifest['episodes']:
        headroom(output)
        source = Path(episode['audio'])
        if sha256(source) != episode['sha256']['audio.wav']:
            raise ValueError('Audio checksum changed after dataset freeze')
        reference_path = Path(episode['reference'])
        if sha256(reference_path) != episode['sha256']['reference.txt']:
            raise ValueError('Reference checksum changed after dataset freeze')
        audio, rate = sf.read(source, dtype='float32')
        assert rate == RATE and audio.ndim == 1 and len(audio) == episode['audio_samples']
        parts, errors = [], []
        cuts = exclusions['episodes'][episode['episode']] if exclusions is not None else []
        spans = retained_spans(len(audio), cuts)
        boundaries = fixed[episode['episode']] if fixed else [(left + start, left + end) for left, right in spans for start, end in audio_chunks(audio[left:right])]
        for index, (start, end) in enumerate(boundaries, 1):
            headroom(output)
            try:
                text = transcribe_audio(audio[start:end], engine)
                error = None
            except Exception as exc:
                text, error = '', repr(exc)
                errors.append({'chunk': index, 'error': error})
                torch.cuda.empty_cache()
            parts.append({'start': start / RATE, 'end': end / RATE, 'text': text, 'error': error})
            if index % 10 == 0:
                print(f'{episode['episode']}: {index}/{len(boundaries)} chunks', flush=True)
        hypothesis = ' '.join(part['text'] for part in parts if part['text'])
        reference = reference_path.read_text()
        if exclusions is not None:
            reference = retained_reference(json.loads(Path(episode['captions']).read_text()), cuts)
        words = jiwer.process_words(normalize(reference), normalize(hypothesis))
        chars = jiwer.process_characters(normalize(reference), normalize(hypothesis))
        row = {
            'episode': episode['episode'], 'audio_seconds': sum(end - start for start, end in spans) / RATE,
            'source_audio_seconds': episode['audio_seconds'], 'retained_spans': [[start / RATE, end / RATE] for start, end in spans],
            'reference_words': words.hits + words.substitutions + words.deletions,
            'substitutions': words.substitutions, 'deletions': words.deletions, 'insertions': words.insertions,
            'wer': words.wer, 'reference_characters': chars.hits + chars.substitutions + chars.deletions,
            'character_errors': chars.substitutions + chars.deletions + chars.insertions,
            'cer': chars.cer, 'chunks': len(parts), 'errors': errors,
        }
        target = output / episode['episode']
        target.mkdir()
        (target / 'transcript.txt').write_text(hypothesis + '\n', encoding='utf-8')
        (target / 'reference.txt').write_text(reference.strip() + '\n', encoding='utf-8')
        dump(target / 'transcript.json', {'model': MODEL, 'revision': MODEL_REVISION, 'chunks': parts})
        (target / 'alignment.txt').write_text(jiwer.visualize_alignment(words), encoding='utf-8')
        rows.append(row)
        dump(output / 'per_episode.json', rows)
        print(f'{episode['episode']} WER={words.wer:.4f}, errors={len(errors)}, elapsed={time.monotonic() - started:.1f}s', flush=True)
        del audio
    counts = {name: sum(row[name] for row in rows) for name in ['reference_words', 'substitutions', 'deletions', 'insertions', 'reference_characters', 'character_errors']}
    report = {
        **counts, 'episodes': len(rows), 'audio_seconds': sum(row['audio_seconds'] for row in rows),
        'wer': sum(counts[name] for name in ['substitutions', 'deletions', 'insertions']) / counts['reference_words'],
        'cer': counts['character_errors'] / counts['reference_characters'],
        'failed_chunks': sum(len(row['errors']) for row in rows),
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'elapsed_seconds': time.monotonic() - started,
        'reference_status': protocol['reference_status'], 'meaning': protocol['meaning'],
    }
    dump(output / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'evaluate'])
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--exclusions', type=Path, help='Fixed per-episode title intervals for this dataset')
    parser.add_argument('--center-channel', action='store_true', help='Prepare only the FC channel of the English 5.1 track')
    parser.add_argument('--boundaries-from', type=Path, help='Reuse exact audio chunk boundaries from a baseline evaluation')
    parser.add_argument('--boundaries', type=Path, help='Use the portable fixed-boundary JSON for the best FC experiment')
    args = parser.parse_args()
    for key, value in {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}.items():
        os.environ[key] = value
    limits()
    if args.action == 'prepare':
        if args.exclusions or args.boundaries_from or args.boundaries:
            parser.error('Exclusions and fixed boundaries are only supported for evaluate')
        prepare(args.source, args.destination, args.center_channel)
    else:
        if args.center_channel:
            parser.error('--center-channel is only supported for prepare')
        evaluate(args.source, args.destination, args.exclusions, args.boundaries_from, args.boundaries)


if __name__ == '__main__':
    main()
