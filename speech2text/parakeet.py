'''Standalone transcription with NVIDIA Parakeet TDT 0.6B v3.'''

import argparse
import sys
from pathlib import Path

from .transcribe import SAMPLE_RATE, read_audio, save_transcript


MODEL = 'nvidia/parakeet-tdt-0.6b-v3'
MODEL_REVISION = '541d1f99c6b0c3cd0b11a95167540bb8edefd82b'
MODEL_DIR = Path(__file__).resolve().parents[1] / 'data/models/parakeet-tdt-0.6b-v3'


def load_model(device='cuda'):
    import torch
    from transformers import AutoModelForTDT, AutoProcessor

    if not (MODEL_DIR / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Parakeet v3 weights are missing: {MODEL_DIR}')
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForTDT.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
        attn_implementation='sdpa',
        local_files_only=True,
    ).to(device).eval()
    return model, processor


def transcribe_audio(audio, engine):
    import torch

    if audio.ndim != 1 or not audio.size:
        raise ValueError('Expected nonempty mono audio at 16 kHz')
    model, processor = engine
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors='pt')
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, return_dict_in_generate=True)
    texts = processor.batch_decode(output.sequences, skip_special_tokens=True)
    if len(texts) != 1:
        raise RuntimeError(f'Expected one transcription, received {len(texts)}')
    return texts[0].strip()


def transcribe(source, engine):
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    audio = read_audio(source)
    return {
        'input': str(source),
        'model': MODEL,
        'model_revision': MODEL_REVISION,
        'duration': audio.size / SAMPLE_RATE,
        'text': transcribe_audio(audio, engine),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Audio or video file; first audio track is used')
    parser.add_argument('-o', '--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Input file does not exist: {args.input}')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}; choose a new directory')
    print(f'Loading {MODEL} on {args.device}...', file=sys.stderr, flush=True)
    engine = load_model(args.device)
    result = transcribe(args.input, engine)
    save_transcript(result, args.output)
    print(args.output / 'transcript.txt')
    print(args.output / 'transcript.json')


if __name__ == '__main__':
    main()
