'''Audio decoding and transcript output shared by Parakeet.'''

import json
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def read_audio(source):
    process = subprocess.run(
        [
            'ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
            '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE),
            '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1',
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    audio = np.frombuffer(process.stdout, dtype='<f4').copy()
    if not audio.size:
        raise ValueError(f'Audio track is empty: {source}')
    return audio


def save_transcript(result, output):
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'transcript.json').write_text(encoded + '\n', encoding='utf-8')
    (output / 'transcript.txt').write_text(result['text'] + '\n', encoding='utf-8')
