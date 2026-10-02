# Разметка фильма для дубляжа

Оригинальная аудиодорожка (стерео wav/flac/видео, EN) → папки по персонажам с репликами, транскрипт с таймингами,
пол и возраст говорящего. Это вход для следующих этапов: перевода и TTS с клонированием голоса. Сам дубляж здесь не делается.

## Выход

```
out/<произведение>/
  vocals.wav, background.wav   стемы (голос / всё остальное), стерео 44.1 кГц
  lines.tsv                    все реплики: id, speaker, gender, start, end, duration, text, file
  speakers.tsv                 speaker, gender (female/male/child), p_gender, age, lines, speech_sec
  markup.json                  то же + пословные тайминги
  SPEAKER_00/
    0002.wav, 0004.wav, …      реплики из стема голоса (моно 44.1 кГц, запас 0.1 с), имя = id в lines.tsv
    metadata.csv               "0002|текст" (формат LJSpeech)
```

## Установка

```bash
uv sync                                         # основное окружение (.venv)
uv venv -p 3.12 .envs/nemo                      # NeMo — отдельно, свой torch
VIRTUAL_ENV=.envs/nemo uv pip install Cython packaging 'nemo-toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'
# видео (необязательно): лица + LR-ASD
git submodule update --init                     # third_party/LR-ASD с весами
uv venv -p 3.12 .envs/av
VIRTUAL_ENV=.envs/av uv pip install torch torchvision torchaudio insightface onnxruntime-gpu opencv-python \
    scenedetect python-speech-features scipy tqdm
```

NeMo нужен из main: Nemotron 3 Diarization использует RoPE, релиз 3.0 его не поддерживает. Модели скачиваются
с HF при первом запуске: RoFormer, Parakeet, Nemotron, ReDimNet2, audEERING age-gender; insightface buffalo_l — в
`~/.insightface`. Нужна GPU с 12 ГБ.

## Запуск

```bash
uv run pipeline.py <файл> [-o out/] [--speakers N] [--video <видео>]
```

Если на входе видеофайл, этап 7b (видео) включается сам. Для wav можно передать видео отдельно через `--video`
(та же шкала времени). Нужно окружение `.envs/av`.

Каждый этап кэшируется файлом в папке произведения. Чтобы этап пересчитался, удали его файл.

## Этапы

| # | Этап | Что делает | Файл |
|---|---|---|---|
| 1 | демукс | Mel-RoFormer (becruily instrumental), маскирование без генерации | vocals.wav, background.wav |
| 2 | вход ASR | голос + 10% фона + `dynaudnorm` (фон маскирует артефакты демукса, AGC вытягивает шёпот и крик) | asr_input.wav |
| 3 | ASR | Parakeet TDT 0.6B v2: куски речи ≤30 с (WhisperX cut & merge) + дораспознавание дыр по речи Nemotron (`nemo_stage.py`, в `.envs/nemo`) | words_full.json, words.json |
| 4 | локальная диаризация | Nemotron 3 Diarization по окнам 60 с (≤8 спикеров на окно) | nemotron_w60.json |
| 5 | связывание окон | ReDimNet2 на речь локального спикера → AHC с запретом слияния внутри окна, k по силуэту | local_emb.npz, turns.json |
| 6 | реплики | слово → спикер по перекрытию; реплика = слова одного спикера без паузы > 0.6 с | — |
| 7 | перепроверка | каждая реплика → ближайший по голосу персонаж (ReDimNet2 реплики против центра персонажа) | — |
| 7b | видео (вход-видео или `--video`) | лица (SCRFD) → треки → LR-ASD; реплика, где уверенно говорит лицо, уходит к персонажу его других реплик (`av_stage.py` в `.envs/av`) | av_tracks.json, av_emb.npy |
| 7c | края | края реплик до речи Nemotron (≤0.25 с) | — |
| 8 | пол | audEERING wav2vec2 age-gender по ≤10 самым длинным репликам спикера | speakers.tsv |
| 9 | экспорт | папки персонажей, lines.tsv, metadata.csv | см. «Выход» |

## Качество

Оценка на AVA-AVD (фильмы, только английские: dev 37 фильмов, test 4) и на Tears of Steel (CC-BY, сценарий и
субтитры в `data/tos`). Допуск на границах 0.25 с.

| | AVA dev SER | AVA dev DER | AVA test SER | AVA test DER |
|---|---|---|---|---|
| Whisper + эмбеддинги по предложениям (v1) | 0.227 | 0.609 | 0.359 | 0.712 |
| **текущий пайплайн** (склейка ASR + дораспознавание дыр) | **0.135** | **0.408** | **0.181** | **0.448** |
| то же + видео (10 фильмов вне обучения LR-ASD, перемерено 2026-10-01) | SER 0.185 → 0.176, DER 0.475 → 0.468 | | | |

Определения метрик:
- **SER** — доля путаницы спикеров на речи, найденной и системой, и эталоном: confusion / (total − miss).
- **DER** — стандартная метрика: miss + FA + confusion.

Проверены и не взяты:
- ASR: Whisper, Qwen3-ASR;
- диаризация: pyannote community-1, DiariZen (WavLM, весь фильм и окна 60 с), MOSS-Transcribe-Diarize (весь фильм), Nemotron дообученный на фильмах AVA, Nemotron на всём файле и окнах 120/300 с, HDBSCAN, UMAP, спектральная кластеризация;
- коррекция: LLM qwen3:8b, запрет слияния по полу, избыток кластеров.


## Оценка

```bash
uv run eval_der.py /mnt/e/ava-avd out/ava $(cat data/ava_dev.list)                 # DER/SER по markup.json
uv run eval_der.py /mnt/e/ava-avd out/ava --hyp <name> $(cat data/ava_dev.list)     # другая система: <name>.json
uv run eval_asr.py /mnt/e/ava-avd out/ava words.json $(cat data/ava_dev.list)       # пропуски и выдумки ASR
uv run eval_wer.py out/TOS_DVDSTEREOMIX/words.json data/tos/TOS-en.srt              # WER против субтитров
.venv/bin/python eval_diar.py out/TOS_DVDSTEREOMIX/markup.json data/tos/script.txt  # WDER против сценария
.venv/bin/python test_pipeline.py
```

Подготовка AVA-AVD:
1. Разметка и сплиты: https://github.com/zcxu-eric/AVA-AVD.
2. Аудио 15–30 мин каждого фильма: `tools/ava_download_audio.sh` (видео для `--video`: `tools/ava_download_video.sh`).
   Скрипт вырезает нужный кусок по HTTP-range, фильм целиком не качается.
3. Английские фильмы уже разбиты: `data/ava_dev.list`, `data/ava_test.list`; `data/ava_av_clean.list` — фильмы вне обучения LR-ASD.

Test (`data/ava_test.list`) используется только для итогового замера.

## Ограничения

- Одинаковые по полу и тембру голоса иногда сливаются в одного персонажа (на AVA около 12% путаницы). Перед
  синтезом стоит прослушать папки: объединить две папки проще, чем разделить одну.
- Одновременная речь: в каждый момент времени назначается один спикер.
- ASR пропускает около 12% эталонной речи (крики, одновременная речь, неразборчивое); найденная «трудная» речь чаще уходит не тому персонажу.
- `--video` на мультфильмах, скорее всего, бесполезен (детектор лиц insightface на анимации не работает); веса insightface и LR-ASD — только некоммерческие/исследовательские.
- Лицензия модели пола — CC-BY-NC-SA, только некоммерческое использование.
