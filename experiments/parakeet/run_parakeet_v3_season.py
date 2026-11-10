"""Run MediaDub's ASR recipe with Parakeet v3 and reuse season-one speaker turns."""

import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter

from pipeline import assign_speakers


MODEL = Path('parakeet_test/parakeet-tdt-0.6b-v3.nemo')
OUTPUT = Path('parakeet_test/ldr_season_v3')
EPISODES = Path('out/ldr_nemotron_gpu')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    partial = path.with_suffix('.json.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    partial.replace(path)


def file_identity(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def load_model():
    import torch
    from nemo.collections.asr.models import ASRModel

    print(f'Загружаю {MODEL.name} на CUDA', flush=True)
    model = ASRModel.restore_from(str(MODEL), map_location='cpu')
    model = model.eval().to(device='cuda', dtype=torch.bfloat16)
    model.decoder.float()
    model.joint.float()
    model.change_attention_model('rel_pos_local_attn', [256, 256])
    model.change_subsampling_conv_chunking_factor(1)
    return model


def transcribe(model, audio, windows, output):
    from nemo_stage import transcribe as mediadub_transcribe

    with TemporaryDirectory(prefix='asr-', dir=output) as temporary:
        work = Path(temporary)
        (work / 'asr_input.wav').symlink_to(audio.resolve())
        (work / 'nemotron_w60.json').symlink_to(windows.resolve())
        return mediadub_transcribe(model, work)


def prepare(episode, input_dir):
    if input_dir is None:
        pointer = EPISODES / episode / 'episode.json'
        audio = Path(load_json(pointer)['source'])
    else:
        matches = [path for path in input_dir.glob('*/asr_input.wav')
                   if re.search(rf'\b{episode}\b', path.parent.name, re.IGNORECASE)]
        if len(matches) != 1:
            raise ValueError(f'{episode}: ожидается один asr_input.wav в {input_dir}, найдено {len(matches)}')
        audio = matches[0]
    baseline = audio.parent / 'words.json'
    windows = audio.parent / 'nemotron_w60.json'
    turns = audio.parent / 'turns.json'
    files = (audio, baseline, windows, turns, MODEL)
    job = {'source': str(audio.resolve()), 'inputs': [file_identity(path) for path in files]}
    output = OUTPUT / episode
    metadata = output / 'source.json'
    if metadata.exists() and load_json(metadata) != job:
        raise ValueError(f'{episode}: входы изменились. Перемести папку {output} и повтори запуск.')
    return episode, audio, baseline, windows, turns, output, job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', choices=[f'S01E{i:02d}' for i in range(1, 19)],
                        help='Run just one episode')
    parser.add_argument('--input-dir', type=Path, help='MediaDub season output with cached Parakeet v2 words')
    args = parser.parse_args()
    episodes = [args.episode] if args.episode else [f'S01E{i:02d}' for i in range(1, 19)]
    try:
        jobs = [prepare(episode, args.input_dir) for episode in episodes]
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Не удалось подготовить сезон: {error}\n')

    model = None
    for episode, audio, baseline, windows, turns, output, job in jobs:
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / 'source.json', job)
        if (output / 'v2.json').is_file() and (output / 'v3.json').is_file():
            print(f'{episode}: готово, пропускаю', flush=True)
            continue
        intervals = load_json(turns)
        if not intervals:
            raise SystemExit(f'{episode}: пустой файл интервалов {turns}')
        save_json(output / 'v2.json', assign_speakers(load_json(baseline), intervals))
        if model is None:
            model = load_model()
        started = perf_counter()
        print(f'{episode}: распознаю всю серию', flush=True)
        words = transcribe(model, audio, windows, output)
        save_json(output / 'v3.json', assign_speakers(words, intervals))
        print(f'{episode}: {len(words)} слов за {perf_counter() - started:.1f} с', flush=True)
    print(f'Результаты: {OUTPUT.resolve()}', flush=True)


if __name__ == '__main__':
    main()
