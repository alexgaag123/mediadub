"""Run the RTX 3060 Nemotron test profile on all five MediaDub episodes."""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path.cwd()
TEST = ROOT / 'nemotron_test'
NEMO = TEST / 'nemo-cuda/bin/nemo-speech'
MODEL = TEST / 'models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-IQ4_NL.gguf'
ENCODER = TEST / 'models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf'
OUTPUT = ROOT / 'out/chernobyl_nemotron_gpu'
CHUNK_SECONDS = 600
GPU_PROFILE = [
    '--device', 'cuda', '--format', 'json', '--diarize',
    '--asr.salm.n_cpu_moe', '40',
    '--asr.salm.max_sequences', '1',
    '--asr.salm.n_ctx_per_sequence', '32768',
    '--asr.salm.max_context', '32768',
    '--asr.salm.n_batch', '512',
    '--asr.salm.encoder_max_batch', '1',
    '--asr.salm.encoder_ahead_seconds', '600',
]


def read_result(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data.get('text'), str) or not isinstance(data.get('words'), list):
        raise ValueError(f'Unexpected Nemotron JSON: {path}')
    return data


def write_atomic(path, content):
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(content, encoding='utf-8')
    partial.replace(path)


def process_episode(source, dest):
    duration = float(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(source),
    ], text=True))
    dest.mkdir(parents=True, exist_ok=True)
    source_stat = source.stat()
    job = {'source': str(source), 'size': source_stat.st_size,
           'mtime_ns': source_stat.st_mtime_ns, 'chunk_seconds': CHUNK_SECONDS,
           'profile': GPU_PROFILE, 'model': str(MODEL)}
    job_file = dest / 'job.json'
    if job_file.exists() and json.loads(job_file.read_text(encoding='utf-8')) != job:
        raise ValueError(f'Audio or settings changed; choose a new output folder: {dest}')
    write_atomic(job_file, json.dumps(job, ensure_ascii=False, indent=2) + '\n')

    words, chunks = [], []
    with tempfile.TemporaryDirectory(prefix='nemotron-chunk-', dir=dest) as tmp:
        wav = Path(tmp) / 'chunk.wav'
        for index, offset in enumerate(range(0, math.ceil(duration), CHUNK_SECONDS), 1):
            name = f'part_{index:03d}'
            result = dest / f'{name}.json'
            print(f'{source.parent.name} | {name} | {offset}s', flush=True)
            if not result.exists():
                subprocess.run([
                    'ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', str(offset),
                    '-i', str(source), '-t', str(CHUNK_SECONDS), '-ac', '1',
                    '-ar', '16000', '-c:a', 'pcm_s16le', str(wav),
                ], check=True)
                partial = result.with_suffix('.json.partial')
                with partial.open('wb') as out:
                    subprocess.run([str(NEMO), 'transcribe', str(wav), '--model',
                                    str(MODEL), *GPU_PROFILE], stdout=out, check=True)
                read_result(partial)
                partial.replace(result)
            else:
                print('  Reusing completed part', flush=True)
            data = read_result(result)
            chunks.append({'file': result.name, 'offset': offset, 'text': data['text'],
                           'word_count': len(data['words'])})
            if data['text'].strip() and not data['words']:
                print(f'  Warning: {name} has text but no aligned word timestamps', flush=True)
            for word in data['words']:
                words.append({**word, 'start': round(offset + word['start'], 3),
                              'end': round(offset + word['end'], 3), 'chunk': name,
                              'speaker': f'{name}:{word["speaker"]}' if word.get('speaker') else None})

    transcript = '\n\n'.join(chunk['text'].strip() for chunk in chunks if chunk['text'].strip())
    write_atomic(dest / 'transcript.txt', transcript + '\n')
    write_atomic(dest / 'episode.json', json.dumps({
        'source': str(source), 'duration': duration, 'text': transcript,
        'chunks': chunks, 'words': words,
    }, ensure_ascii=False, indent=2) + '\n')
    print(f'Done: {dest}', flush=True)


def main():
    for path in (NEMO, MODEL, ENCODER):
        if not path.is_file():
            raise SystemExit(f'File not found: {path}')
    if not os.access(NEMO, os.X_OK):
        raise SystemExit(f'Cannot execute: {NEMO}')
    sources = {}
    for source in sorted((ROOT / 'out/ldr_season3').glob('*/asr_input.wav')):
        match = re.match(r'Chernobyl\.s01e(0[1-5])\b', source.parent.name, re.IGNORECASE)
        if match:
            episode = match[1]
            if episode in sources:
                raise SystemExit(f'Multiple audio files for episode {episode}')
            sources[episode] = source
    missing = [f'{index:02d}' for index in range(1, 6) if f'{index:02d}' not in sources]
    if missing:
        raise SystemExit('Missing MediaDub asr_input.wav for episodes: ' + ', '.join(missing))
    for episode, source in sorted(sources.items()):
        process_episode(source, OUTPUT / f'S01E{episode}')


if __name__ == '__main__':
    main()
