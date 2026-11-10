"""Compare two Parakeet passes on LDR audio in a local browser."""

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from html import escape
import json
from pathlib import Path
import os
import re
from urllib.parse import quote


PAGE = '''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · __EPISODE__</title>
<style>
:root{color-scheme:light dark;--line:color-mix(in srgb,CanvasText 20%,Canvas);--dim:color-mix(in srgb,CanvasText 68%,Canvas)}
body{background:Canvas;color:CanvasText;font:16px/1.55 system-ui;margin:0}
main{max-width:1100px;margin:auto;padding:0 16px 28px}
header{position:sticky;top:0;background:Canvas;padding:12px 0;border-bottom:1px solid var(--line)}
h1{font-size:1.3rem;margin:0 0 4px}h2{font-size:1rem;margin:0 0 6px}p{margin:0 0 8px}
audio{width:min(100%,420px);height:38px}button{font:inherit;cursor:pointer}
.controls,.meta{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.meta{margin-bottom:8px}
.row{margin:14px 0;padding:12px;border:1px solid var(--line);border-radius:10px}
.cols{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.cols>div{padding:10px;background:color-mix(in srgb,CanvasText 5%,Canvas);border-radius:6px;overflow-wrap:anywhere}
.muted,small{color:var(--dim)}mark{background:#f6d681;color:#242019;border-radius:2px}
.speaker{display:block;font-size:.8em;font-weight:700;margin-top:6px}.id-diff{outline:2px solid #b263d7;border-radius:2px}
.speaker-smoothed{text-decoration:underline dotted #009985;text-underline-offset:4px}
@media(max-width:580px){.cols{grid-template-columns:1fr}}
</style>
<main><header><h1>__TITLE__ · __EPISODE__</h1>
<p class="muted">__DESCRIPTION__</p>
<div class="controls"><audio controls preload="metadata" src="__AUDIO__"></audio>
__TEXT_FILTER____ID_FILTER__</div>
<small id="status" role="status"></small></header>
__SEASON_LINK__
__ROWS__
</main>
<script>
const player = document.querySelector('audio');
const status = document.getElementById('status');
let stopAt = null;
const textFilter = document.getElementById('filter');
const idFilter = document.getElementById('id-filter');
function filterRows() {
  document.querySelectorAll('.row').forEach(row => {
    row.hidden = (textFilter?.checked && row.dataset.same === '1') ||
                 (idFilter?.checked && row.dataset.idDiff === '0');
  });
}
textFilter?.addEventListener('change', filterRows);
idFilter?.addEventListener('change', filterRows);
document.querySelectorAll('button[data-start]').forEach(button => button.addEventListener('click', () => {
  stopAt = Number(button.dataset.end);
  player.currentTime = Number(button.dataset.start);
  status.textContent = button.dataset.label;
  player.play().catch(() => { status.textContent = 'Не удалось открыть WAV.'; });
}));
player.addEventListener('timeupdate', () => {
  if (stopAt !== null && player.currentTime >= stopAt) {
    player.pause();
    stopAt = null;
  }
});
</script></html>
'''


def read_groups(path, window):
    groups = defaultdict(list)
    for word in sorted(json.loads(path.read_text(encoding='utf-8')), key=lambda w: w['start']):
        middle = (word['start'] + word['end']) / 2
        groups[int(middle // window)].append(word)
    return groups


def diff_columns(old, new):
    def normalized(words):
        return [re.sub(r'[^\w]+', '', word['w'].casefold()) for word in words]

    left_text, right_text = set(), set()
    left_ids, right_ids = set(), set()
    for kind, a, b, c, d in SequenceMatcher(None, normalized(old), normalized(new), autojunk=False).get_opcodes():
        if kind != 'equal':
            left_text.update(range(a, b))
            right_text.update(range(c, d))
            continue
        for i, j in zip(range(a, b), range(c, d)):
            left, right = old[i], new[j]
            midpoint_delta = abs(left['start'] + left['end'] - right['start'] - right['end']) / 2
            if (midpoint_delta <= .75 and left.get('speaker') and right.get('speaker') and
                    left['speaker'] != right['speaker']):
                left_ids.add(i)
                right_ids.add(j)
    return (word_column(old, left_text, left_ids), word_column(new, right_text, right_ids),
            bool(left_text or right_text), len(left_ids))


def word_column(words, text_diff, id_diff):
    empty = '<span class="muted">Нет слов на этом интервале</span>'
    parts = []
    previous = None
    for i, word in enumerate(words):
        speaker = word.get('speaker')
        if speaker and speaker != previous:
            parts.append(f'<span class="speaker">{escape(str(speaker))} · {word["start"]:.2f} с</span>')
        previous = speaker
        text = escape(word['w'])
        if i in text_diff:
            text = f'<mark>{text}</mark>'
        if i in id_diff:
            text = f'<span class="id-diff" title="Другой ID у совпадающего слова">{text}</span>'
        if 'speaker_before_smoothing' in word:
            previous_id = escape(str(word['speaker_before_smoothing']))
            text = f'<span class="speaker-smoothed" title="ID до сглаживания: {previous_id}">{text}</span>'
        parts.append(text)
    return ' '.join(parts) or empty


def timestamp(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f'{minutes:02d}:{seconds:02d}'


def write_report(results, audio, window, episode, season=False, *, diarization=None):
    suffix = f'_{diarization}' if diarization in ('regular', 'smoothed') else ''
    filenames = ('mediadub_words.json', f'pyannote_words{suffix}.json') if diarization else ('v2.json', 'v3.json')
    mode = {'regular': 'с наложениями', 'smoothed': 'со сглаживанием'}.get(diarization, 'один голос')
    labels = ('MediaDub', f'Community-1 · {mode}') if diarization else ('Parakeet v2', 'Parakeet v3')
    old = read_groups(results / filenames[0], window)
    new = read_groups(results / filenames[1], window)
    if not audio.is_file():
        raise FileNotFoundError(f'{audio}. Укажи точный путь к WAV через --audio.')
    rows = []
    total_id_diffs = 0
    for index in sorted(old.keys() | new.keys()):
        start, end = index * window, (index + 1) * window
        left, right, differs, id_diffs = diff_columns(old[index], new[index])
        total_id_diffs += id_diffs
        label = f'{timestamp(start)} – {timestamp(end)}'
        id_note = f'<small>Слов с другим ID: {id_diffs}</small>' if id_diffs else ''
        rows.append(f'<section class="row" data-same="{int(not differs)}" '
                    f'data-id-diff="{id_diffs}"><div class="meta">'
                    f'<strong>{label}</strong><button type="button" data-start="{start}" '
                    f'data-end="{end}" data-label="{label}">▶ Слушать</button>{id_note}</div>'
                    f'<div class="cols"><div><h2>{labels[0]}</h2>{left}</div>'
                    f'<div><h2>{labels[1]}</h2>{right}</div></div></section>')
    report = results / f'comparison{suffix}.html'
    page = PAGE.replace('__AUDIO__', quote(Path(os.path.relpath(audio, report.parent)).as_posix()))
    page = page.replace('__EPISODE__', escape(episode))
    page = page.replace('__TITLE__', 'MediaDub и Community-1' if diarization else 'Parakeet v2 и v3')
    description = ('Слова и таймкоды взяты из одного прогона Parakeet v3. Слева показаны исходные ID MediaDub. '
                   'Справа ID Community-1 сопоставлены с ними по серии. Рамка отмечает расхождения. '
                   '«?» означает, что слово не привязалось к голосу.'
                   if diarization else 'Выделены различающиеся слова. Плеер использует дорожку, которую получили обе модели.')
    if diarization == 'smoothed':
        description += ' Пунктир снизу отмечает слова, чей ID изменило сглаживание.'
    page = page.replace('__DESCRIPTION__', description)
    page = page.replace('__TEXT_FILTER__', '' if diarization else '<label><input id="filter" type="checkbox"> Только различия в тексте</label>')
    page = page.replace('__ID_FILTER__', '<label><input id="id-filter" type="checkbox"> Только различия в ID</label>' if season or diarization else '')
    navigation = ''
    if season:
        navigation = '<p><a href="../index.html">Все серии</a> · Рамка отмечает совпавшее слово с другим ID.</p>'
    if diarization == 'regular':
        navigation = '<p><a href="comparison.html">Открыть вариант Community-1 с одним голосом</a></p>'
        if (results / 'comparison_smoothed.html').is_file():
            navigation += '<p><a href="comparison_smoothed.html">Открыть вариант со сглаживанием</a></p>'
    if diarization == 'exclusive':
        navigation = '<p><a href="comparison_regular.html">Открыть вариант Community-1 с наложениями</a></p>'
    if diarization == 'smoothed':
        navigation = '<p><a href="comparison_regular.html">Открыть Community-1 без сглаживания</a></p>'
    page = page.replace('__SEASON_LINK__', navigation)
    report.write_text(page.replace('__ROWS__', '\n'.join(rows) or '<p>Нет слов с таймкодами.</p>'), encoding='utf-8')
    print(report.resolve())
    return total_id_diffs


def write_season(folder, window):
    links = []
    for episode in sorted(folder.glob('S01E[0-9][0-9]')):
        if not all((episode / name).is_file() for name in ('source.json', 'v2.json', 'v3.json')):
            continue
        audio = Path(json.loads((episode / 'source.json').read_text(encoding='utf-8'))['source'])
        changes = write_report(episode, audio, window, episode.name, season=True)
        links.append(f'<li><a href="{quote(episode.name)}/comparison.html">{escape(episode.name)}</a>'
                     f' · слов с другим ID: {changes}</li>')
    if not links:
        raise ValueError(f'Нет завершённых серий в {folder}')
    index = folder / 'index.html'
    index.write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'
                     '<title>Parakeet v2 и v3 · сезон</title>'
                     '<style>:root{color-scheme:light dark}body{font:16px/1.6 system-ui;max-width:760px;margin:2em auto;padding:0 16px}</style>'
                     f'<h1>Parakeet v2 и v3 · {len(links)} из 18 серий</h1>'
                     '<p>Обе версии используют одну диаризацию MediaDub. Смена ID отмечена у совпадающих слов на близких таймкодах.</p>'
                     '<ul>' + ''.join(links) + '</ul></html>', encoding='utf-8')
    print(index.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('parakeet_test/ldr_s01e01_full'))
    parser.add_argument('--audio', type=Path, help='Override the original ASR input WAV')
    parser.add_argument('--season', type=Path, help='Generate reports for a batch output folder')
    parser.add_argument('--window', type=int, default=15, help='Seconds per comparison row')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    try:
        if args.season:
            write_season(args.season, args.window)
        else:
            audio = args.audio
            if audio is None:
                episode = Path('out/ldr_nemotron_gpu/S01E01/episode.json')
                audio = Path(json.loads(episode.read_text(encoding='utf-8'))['source'])
            write_report(args.results, audio, args.window, 'S01E01')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Не удалось открыть результат или WAV: {error}\n')


if __name__ == '__main__':
    main()
