"""Put MediaDub and Nemotron words on the same episode timeline for review."""

import argparse
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
from html import escape
import json
import os
from pathlib import Path
import re
from urllib.parse import quote


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def baseline_words(markup):
    for segment in markup['segments']:
        for word in segment['words']:
            yield {**word, 'word': word['w'], 'speaker': segment['speaker']}


def group_by_time(words, seconds):
    groups = {}
    for word in sorted(words, key=lambda item: item['start']):
        index = int((word['start'] + word['end']) / 2 // seconds)
        groups.setdefault(index, []).append(word)
    return groups


def show_words(words):
    parts = []
    previous = None
    for word in words:
        speaker = word.get('speaker') or '?'
        if speaker != previous:
            parts.append(f'[{speaker} @ {word["start"]:.1f}s]')
            previous = speaker
        parts.append(' '.join(word['word'].split()))
    return ' '.join(parts)


def normalized_text(words):
    return re.findall(r'\w+', ' '.join(word['word'] for word in words).casefold())


def clock(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f'{minutes:02d}:{seconds:02d}'


def word_diff(old, new):
    def tokens(words):
        return [''.join(re.findall(r'\w+', word['word'].casefold())) for word in words]

    left, right = set(), set()
    for tag, a, b, c, d in SequenceMatcher(None, tokens(old), tokens(new), autojunk=False).get_opcodes():
        if tag != 'equal':
            left.update(range(a, b))
            right.update(range(c, d))
    return left, right


def speaker_intervals(words):
    intervals = defaultdict(list)
    for word in sorted(words, key=lambda item: item['start']):
        speaker = word.get('speaker')
        start, end = word['start'], word['end']
        if not speaker or end <= start:
            continue
        spans = intervals[str(speaker)]
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return intervals


def hybrid_words(words, turns):
    spans = speaker_intervals({'start': start, 'end': end, 'speaker': speaker}
                              for start, end, speaker in turns)
    result = []
    for word in words:
        start, end = word['start'], word['end']
        scores = Counter()
        for speaker, intervals in spans.items():
            for left, right in intervals:
                if left >= end:
                    break
                overlap = min(end, right) - max(start, left)
                if overlap > 0:
                    scores[speaker] += overlap
        ranked = scores.most_common(2)
        speaker = None
        if ranked:
            best, overlap = ranked[0]
            second = ranked[1][1] if len(ranked) == 2 else 0
            duration = end - start
            if overlap >= .5 * duration and overlap - second >= .2 * duration:
                speaker = best
        result.append({**word, 'nemotron_speaker': word.get('speaker'), 'speaker': speaker})
    return result


def missing_word_runs(old, new):
    # A timing drift of up to a quarter of a second is not a missing-word candidate.
    covered = speaker_intervals({'speaker': 'coverage', 'start': word['start'] - .25,
                                 'end': word['end'] + .25} for word in new)['coverage']
    runs = []
    cursor = 0
    current = []
    for word in sorted(old, key=lambda item: item['start']):
        start, end = word['start'], word['end']
        while cursor < len(covered) and covered[cursor][1] <= start:
            cursor += 1
        missing = end > start and (cursor == len(covered) or covered[cursor][0] >= end)
        if current and (not missing or start - current[-1]['end'] > .8 or
                        current[-1].get('speaker') != word.get('speaker')):
            runs.append(current)
            current = []
        if missing:
            current.append(word)
    if current:
        runs.append(current)
    return [{'start': run[0]['start'], 'end': max(word['end'] for word in run),
             'speaker': run[0].get('speaker'), 'text': ' '.join(word['word'] for word in run)}
            for run in runs if len(run) >= 2 or run[0]['end'] - run[0]['start'] >= .75]


def overlaps_by_row(old_words, new_words, window):
    rows = defaultdict(Counter)
    for left, old_spans in speaker_intervals(old_words).items():
        for right, new_spans in speaker_intervals(new_words).items():
            i = j = 0
            while i < len(old_spans) and j < len(new_spans):
                start = max(old_spans[i][0], new_spans[j][0])
                end = min(old_spans[i][1], new_spans[j][1])
                while start < end:
                    row = int(start // window)
                    stop = min(end, (row + 1) * window)
                    rows[row][left, right] += stop - start
                    start = stop
                if old_spans[i][1] <= new_spans[j][1]:
                    i += 1
                else:
                    j += 1
    return rows


def match_speakers(overlaps):
    pairs = Counter()
    for row in overlaps.values():
        pairs.update(row)
    by_old, by_new = defaultdict(Counter), defaultdict(Counter)
    for (left, right), seconds in pairs.items():
        by_old[left][right] = seconds
        by_new[right][left] = seconds

    matches = []
    alternatives = []
    for (left, right), seconds in pairs.most_common():
        share_old = seconds / by_old[left].total()
        share_new = seconds / by_new[right].total()
        if seconds >= 1 and min(share_old, share_new) >= .60:
            matches.append((left, right, seconds))
        elif seconds >= 1 and max(share_old, share_new) >= .20:
            alternatives.append((left, right, seconds))
    return matches, alternatives


def word_column(words, changed, speakers):
    if not words:
        return '<span class="quiet">Нет слов в этом интервале</span>'
    runs = []
    for index, word in enumerate(words):
        speaker = word.get('speaker') or '?'
        if not runs or runs[-1][0] != speaker:
            runs.append((speaker, word['start'], []))
        token = escape(word['word'])
        runs[-1][2].append(f'<mark>{token}</mark>' if index in changed else token)
    return ''.join(f'<p><span class="speaker">{escape(str(speaker))}'
                   f'{(" · голос " + str(speakers[str(speaker)])) if str(speaker) in speakers else ""}'
                   f' · {start:.1f} с</span>'
                   f'{" ".join(tokens)}</p>' for speaker, start, tokens in runs)


PAGE = '''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · MediaDub / Nemotron</title>
<style>
:root{color-scheme:light dark;--line:color-mix(in srgb,CanvasText 18%,Canvas);--muted:color-mix(in srgb,CanvasText 70%,Canvas)}
body{font:16px/1.5 system-ui,sans-serif;margin:0;background:Canvas;color:CanvasText}main{max-width:1150px;margin:auto;padding:0 16px 30px}
header{position:sticky;top:0;background:Canvas;padding:10px 0;border-bottom:1px solid var(--line);z-index:1}
h1{font-size:1.3rem;margin:0 0 4px}h2{font-size:.9rem;margin:0 0 6px}p{margin:0 0 8px}.quiet,small{color:var(--muted)}
audio{width:min(100%,480px);height:40px;vertical-align:middle}button{font:inherit;cursor:pointer}
.controls,.meta{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.meta{margin-bottom:8px}
.row{margin:14px 0;border:1px solid var(--line);border-radius:10px;padding:12px;scroll-margin-top:150px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}.cols>div{min-width:0;overflow-wrap:anywhere;padding:10px;background:color-mix(in srgb,CanvasText 4%,Canvas);border-radius:7px}
.speaker{display:block;font-size:.76rem;font-weight:700;color:var(--muted)}mark{background:#f6d681;color:#242019;border-radius:2px}
details{margin:12px 0;padding:8px 12px;border:1px solid var(--line);border-radius:7px}summary{cursor:pointer;font-weight:600}
.speaker-alert{color:CanvasText;font-weight:600}table{border-collapse:collapse;width:100%;text-align:left}th,td{padding:5px 8px;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
@media(max-width:620px){.cols{grid-template-columns:1fr}}
</style>
<main><header><h1>__TITLE__ · сравнение распознавания</h1>
<p class="quiet">Выделены разные слова. «Голос 1», «голос 2» связывают вероятные пары ID внутри серии.</p>
<div class="controls">__AUDIO__ <label><input id="filter" type="checkbox"> Только расхождения в тексте</label>
<label><input id="speaker-filter" type="checkbox"> Только спорные ID</label></div>
<small id="status" role="status"></small></header>
<p><a href="../__INDEX__">Все серии</a></p>
__NOTES__
__SPEAKERS__
__ROWS__
</main>
<script>
const player = document.querySelector('audio');
const status = document.getElementById('status');
let stopAt = null;
const textFilter = document.getElementById('filter');
const speakerFilter = document.getElementById('speaker-filter');
function filterRows() {
  document.querySelectorAll('.row').forEach(row => {
    row.hidden = (textFilter.checked && row.dataset.same === '1') ||
                 (speakerFilter.checked && row.dataset.speakerIssue !== '1');
  });
}
textFilter.addEventListener('change', filterRows);
speakerFilter.addEventListener('change', filterRows);
if (player) {
  document.querySelectorAll('button[data-start]').forEach(button => button.addEventListener('click', () => {
    stopAt = Number(button.dataset.end);
    player.currentTime = Number(button.dataset.start);
    status.textContent = button.dataset.label;
    player.play().catch(() => { status.textContent = 'Не удалось открыть локальный WAV.'; });
  }));
  player.addEventListener('timeupdate', () => {
    if (stopAt !== null && player.currentTime >= stopAt) {
      player.pause();
      stopAt = null;
    }
  });
}
</script></html>
'''


def write_browser_report(path, title, rows, audio, notes, overlaps, window, gaps=None):
    hybrid = gaps is not None
    audio_html = '<span class="quiet">Аудио не найдено</span>'
    if audio.is_file():
        url = quote(Path(os.path.relpath(audio, path.parent)).as_posix())
        audio_html = f'<audio controls preload="metadata" src="{url}"></audio>'
    matches, alternatives = match_speakers(overlaps)
    old_ids = {old: index for index, (old, _, _) in enumerate(matches, 1)}
    new_ids = {new: index for index, (_, new, _) in enumerate(matches, 1)}
    review_pairs = {(old, new) for old, new, _ in alternatives}
    pair_rows = []
    for label, (old, new, seconds) in [(f'Голос {i}', pair) for i, pair in enumerate(matches, 1)] + [
        ('Проверить', pair) for pair in alternatives
    ]:
        pair_rows.append(f'<tr><td>{escape(label)}</td><td>{escape(old)}</td>'
                         f'<td>{escape(new)}</td><td>{seconds:.1f} с</td></tr>')
    speaker_panel = ('<details><summary>Сопоставление ID говорящих</summary>'
                     '<p>Пары основаны на пересечении таймкодов слов. Общий номер выдаётся, '
                     'если пара занимает от 60% пересечения для каждого ID и пересекается от 1 секунды. '
                     'Повтор ID в нескольких строках стоит сверить на слух.</p>'
                     '<table><thead><tr><th>Пара</th><th>MediaDub</th><th>Nemotron</th>'
                     '<th>Пересечение</th></tr></thead><tbody>' + ''.join(pair_rows) + '</tbody></table></details>') if pair_rows else (
        '<p class="quiet">Для сопоставления ID недостаточно слов с общими таймкодами.</p>')
    cards = []
    text_issues = speaker_issues = 0
    for start, end, status, old, new in rows:
        left, right = word_diff(old, new)
        text_issues += bool(left or right)
        disputed = [(a, b, seconds) for (a, b), seconds in overlaps.get(start // window, Counter()).most_common()
                    if seconds >= .4 and ((a, b) in review_pairs or
                       (a in old_ids and b in new_ids and old_ids[a] != new_ids[b]))]
        speaker_issues += bool(disputed)
        speaker_note = ''.join(f'<p class="speaker-alert">Проверить ID: {escape(a)} ↔ {escape(b)} '
                               f'({seconds:.1f} с)</p>' for a, b, seconds in disputed)
        gap_note = ''.join(f'<p class="speaker-alert">Возможный пропуск · {gap["start"]:.1f} с · '
                           f'{escape(gap["text"])}</p>' for gap in (gaps or [])
                           if start <= gap['start'] < end)
        label = f'{clock(start)} – {clock(end)}'
        button = (f'<button type="button" data-start="{start}" data-end="{end}" '
                  f'data-label="{label}">▶ Слушать</button>') if audio.is_file() else ''
        cards.append(f'<section class="row" data-same="{int(not left and not right)}" '
                     f'data-speaker-issue="{int(bool(disputed))}">'
                     f'<div class="meta"><strong>{label}</strong>{button}<small>{escape(status)}</small></div>'
                     f'{speaker_note}{gap_note}<div class="cols"><div><h2>MediaDub</h2>{word_column(old, left, old_ids)}</div>'
                     f'<div><h2>{"Nemotron + ID MediaDub" if hybrid else "Nemotron"}</h2>'
                     f'{word_column(new, right, new_ids)}</div></div></section>')
    values = {'TITLE': escape(title), 'AUDIO': audio_html, 'INDEX': 'hybrid_index.html' if hybrid else 'index.html',
              'SPEAKERS': speaker_panel,
              'NOTES': ''.join(f'<p class="quiet">{escape(note)}</p>' for note in notes),
              'ROWS': '\n'.join(cards) or '<p>Нет слов с таймкодами для сравнения.</p>'}
    path.write_text(re.sub(r'__(TITLE|AUDIO|INDEX|NOTES|SPEAKERS|ROWS)__', lambda m: values[m[1]], PAGE), encoding='utf-8')
    return text_issues, speaker_issues


def compare(episode_file, baseline_dir, window, hybrid=False):
    episode = load_json(episode_file)
    source = Path(episode['source'])
    source_dir = baseline_dir / source.parent.name if baseline_dir else source.parent
    markup_file = source_dir / 'markup.json'
    if not markup_file.is_file():
        raise ValueError(f'MediaDub markup.json not found: {markup_file}. Use --baseline-dir.')

    old_words = list(baseline_words(load_json(markup_file)))
    new_words = episode['words']
    gaps = None
    if hybrid:
        gaps = missing_word_runs(old_words, new_words)
        new_words = hybrid_words(new_words, load_json(source_dir / 'turns.json'))
        (episode_file.parent / 'hybrid.json').write_text(json.dumps({
            **episode, 'words': new_words, 'speaker_source': 'MediaDub turns.json',
            'possible_missing_runs': gaps,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    old_groups = group_by_time(old_words, window)
    new_groups = group_by_time(new_words, window)
    overlaps = overlaps_by_row(old_words, new_words, window)
    rows = []
    for index in sorted(old_groups.keys() | new_groups.keys()):
        old = old_groups.get(index, [])
        new = new_groups.get(index, [])
        if not old:
            status = 'Слова только у Nemotron'
        elif not new:
            status = 'Слова только у MediaDub'
        elif normalized_text(old) == normalized_text(new):
            status = 'Совпадает без учёта пунктуации'
        else:
            status = 'Различается'
        rows.append((index * window, (index + 1) * window, status, old, new))
    report = episode_file.parent / 'comparison.tsv'
    if not hybrid and not report.exists():
        with report.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['От', 'До', 'Текст', 'MediaDub', 'Nemotron', 'Ручная проверка'])
            for start, end, status, old, new in rows:
                writer.writerow([clock(start), clock(end), status, show_words(old), show_words(new), ''])

    notes = []
    if hybrid:
        unassigned = sum(word['speaker'] is None for word in new_words)
        notes.append(f'Гибрид: текст Nemotron и ID из MediaDub turns.json. Без уверенного ID: '
                     f'{unassigned} из {len(new_words)} слов. Возможных пропусков: {len(gaps)}.')
    for chunk in episode.get('chunks', []):
        if chunk['text'].strip() and not chunk['word_count']:
            notes.append(f'Nemotron {chunk["file"]}: текст без таймкодов, см. transcript.txt.')
    audio = source_dir / 'mix.wav'
    if not audio.is_file():
        audio = source_dir / 'asr_input.wav'
    html = report.with_name('hybrid_comparison.html') if hybrid else report.with_suffix('.html')
    counts = write_browser_report(html, episode_file.parent.name, rows, audio, notes, overlaps, window, gaps)
    print(html.resolve())
    return html, counts


def write_index(folder, reports, hybrid=False):
    rows = []
    for report, (text_issues, speaker_issues) in reports:
        link = quote(report.relative_to(folder).as_posix())
        rows.append(f'<tr><td><a href="{link}">{escape(report.parent.name)}</a></td>'
                    f'<td>{text_issues}</td><td>{speaker_issues}</td></tr>')
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>MediaDub и Nemotron · все серии</title>'
            '<style>:root{color-scheme:light dark}body{font:16px/1.5 system-ui;max-width:720px;margin:2em auto;'
            'padding:0 16px}table{width:100%;text-align:left}td,th{padding:6px}</style>'
            f'<h1>MediaDub и Nemotron · {len(rows)} серий</h1>'
            '<p>В таблице число интервалов для просмотра. Спорные ID нужно проверить на слух.</p>'
            '<table><tr><th>Серия</th><th>Расхождения в тексте</th><th>Спорные ID</th></tr>'
            + ''.join(rows) + '</table></html>')
    index = folder / ('hybrid_index.html' if hybrid else 'index.html')
    index.write_text(page, encoding='utf-8')
    print(index.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', help='Only this episode, e.g. S01E01')
    parser.add_argument('--baseline-dir', type=Path, help='MediaDub season output if it was moved')
    parser.add_argument('--nemotron-dir', type=Path, default=Path('out/ldr_nemotron_gpu'))
    parser.add_argument('--window', type=int, default=15, help='Seconds per row (default 15)')
    parser.add_argument('--hybrid', action='store_true', help='Keep Nemotron text and assign MediaDub speaker turns')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be at least 1 second')
    episodes = sorted(args.nemotron_dir.glob('*/episode.json'))
    if args.episode:
        episodes = [file for file in episodes if file.parent.name == args.episode.upper()]
    if not episodes:
        parser.error(f'No completed episode.json files found in {args.nemotron_dir}')
    reports = []
    for file in episodes:
        try:
            reports.append(compare(file, args.baseline_dir, args.window, args.hybrid))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SystemExit(f'{file}: {error}') from error
    if not args.episode:
        write_index(args.nemotron_dir, reports, args.hybrid)


if __name__ == '__main__':
    main()
