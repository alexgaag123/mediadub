# Диаризация MediaDub · чекпоинт

Диаризация определяет, кто и когда говорит. Для дубляжа её результат — реплики с таймкодами и стабильными ID голосов внутри каждой серии.

## Результаты

| Что оценили | Данные и эталон | Результат |
| --- | --- | ---: |
| CER Parakeet v3 в пайплайне MediaDub | Friends S01E01–S01E12, английские субтитры SDH | **16,54%** |
| Точность назначения говорящего | Friends S01E01–S01E12, отобранные реплики MELD | **71,75%** (447 из 623) |
| DER и SER на Friends | Четыре сцены, 5 мин 18 с, черновая разметка на основе MELD | Запланированы на следующий чекпоинт после ручной проверки |

Из 987 реплик MELD в оценку вошли 623. У 320 текст не подтвердился по субтитрам этой копии сериала, ещё 44 исключены из-за существенного пересечения с репликой другого говорящего в разметке MELD. В оценённой выборке у **131** реплики назначен другой говорящий, у **43** не найдено достаточного пересечения с речью, у **2** назначение неоднозначно. Детализация по сериям есть в [сводном CSV](results/meld_speaker_summary.csv).

Оценка по MELD измеряет долю реплик с правильно назначенным голосом. Каждая реплика имеет одинаковый вес. Сопоставление ID MediaDub и имён персонажей выполняется отдельно для каждой серии, одно имя на один ID. Для DER и SER нужны длительности речи и проверенные границы, включая речь между репликами и перебивания. [Данные MELD](https://github.com/declare-lab/MELD/tree/e8cedf27b5d2877e198332c957127e16eb214afe/data/MELD) · [Статья MELD](https://aclanthology.org/P19-1050/)

### DER и SER на AVA-AVD

Это ранее зафиксированные результаты **базового пайплайна из README MediaDub**, который использовал Parakeet v2. Эталон AVA-AVD, допуск на границах 0,25 с, наложения учитываются. Числа приведены как ориентир для метода.

| Пайплайн | Dev SER (37 фильмов) | Dev DER | Test SER (4 фильма) | Test DER |
| --- | ---: | ---: | ---: | ---: |
| Whisper + эмбеддинги по предложениям, v1 | 22,7% | 60,9% | 35,9% | 71,2% |
| Базовый MediaDub | **13,5%** | **40,8%** | **18,1%** | **44,8%** |

- **DER** — сумма времени пропущенной речи, ложной речи и путаницы говорящих, делённая на эталонное время речи.
- **SER** в этом проекте — время путаницы говорящих, делённое на эталонное время речи за вычетом пропущенной речи.

[Исходная таблица MediaDub](../README.md#качество) · [Скрипт оценки AVA-AVD](../eval_der.py)

## Что сделали

1. **Подготовили вход.** Проверили дорожки 24 MKV первого сезона Friends. Оригинал лежит в потоке `0:2`, AC-3 5.1. В `S01E17` у этого потока ошибочно стоит `rus`, английскую речь подтвердили на аудиофрагменте. Для прогона использовали стереосведение оригинала и штатное отделение голоса от фона.
2. **Прогнали первые 12 серий.** MediaDub выделяет голос через Mel-RoFormer. Nemotron 3 Diarization на 16 кГц размечает говорящих внутри 60-секундных окон. ReDimNet2 и кластеризация связывают локальные ID по всей серии. Parakeet v3 распознаёт слова и таймкоды, MediaDub назначает ID и формирует реплики.
3. **Подготовили оценку.** CER считаем по `words.json` и английским SDH. Для оценки говорящего сверили интервалы и текст MELD с субтитрами используемого BDRip и сравнили их с `turns.json`. Четыре сцены с черновой разметкой оставили для ручного эталона на следующий чекпоинт.

На отдельном эксперименте с «Любовь, смерть и роботы» сравнили Community‑1 и MediaDub на одних словах Parakeet v3. В браузерном отчёте проверили вариант с наложениями, вариант с одним голосом и сглаживание коротких переключений. По прослушиванию первой серии для дальнейшего прогона оставили диаризацию MediaDub.

Конфигурация ASR здесь относится к **нашему прогону в составе MediaDub**. Отдельный [эксперимент Speech2Text с центральным каналом](../speech2text/README.md) использует другой вход, блоки, правила отбора и все 24 серии.

## Файлы

| Файл | Назначение |
| --- | --- |
| [portable/run_friends_v3.py](portable/run_friends_v3.py) | Проверка английского потока и последовательный запуск серий |
| [portable/nemo-stage-local.py](portable/nemo-stage-local.py) | Подключение Parakeet v3 к локальному NeMo в переносимой сборке |
| [evaluation/eval_friends.py](evaluation/eval_friends.py) | CER по SDH, DER и SER по ручному эталону, CER по ручному тексту при его наличии |
| [evaluation/eval_friends_meld.py](evaluation/eval_friends_meld.py) | Подготовка MELD и оценка назначения говорящего |
| [results/meld_speaker_summary.csv](results/meld_speaker_summary.csv) | Числа завершённого прогона по 12 сериям |

## Как воспроизвести

### Прогон MediaDub

Перенеси два файла из `portable/` в корень установленной переносимой сборки MediaDub с `run-local.sh` и окружением `.envs/nemo`. Положи веса NVIDIA `parakeet-tdt-0.6b-v3.nemo` в `parakeet_test/` или укажи путь через `--model`. Папка с MKV должна содержать использованный BDRip. [Веса NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/main)

```bash
friends='/путь/к/Friends S01 (1994-1995) BDRip-AVC'
python3 run_friends_v3.py "$friends" --episode S01E01
# Убери --episode, чтобы продолжить сезон. Готовые серии пропускаются.
```

На выходе каждой серии в `out/friends_s01_v3/Friends.S01E**/` появятся `words.json`, `turns.json`, `lines.tsv`, `speakers.tsv`, `markup.json` и WAV отдельных реплик. Для передачи в перевод и озвучку используй `lines.tsv` и `markup.json`.

### CER и оценка на MELD

Из корня ветки укажи путь к исходным MKV и к выходам переносимой сборки. `uv` установит зависимости оценочных скриптов в отдельные окружения. Для оценки GPU не требуется.

```bash
friends='/путь/к/Friends S01 (1994-1995) BDRip-AVC'
results='/путь/к/MediaDub-RTX3060/out/friends_s01_v3'
uv run --script diarization/evaluation/eval_friends.py --results "$results" subtitles "$friends"

mkdir -p out/friends_eval/meld_csv
for split in train dev test; do
  curl --fail --location \
    "https://raw.githubusercontent.com/declare-lab/MELD/e8cedf27b5d2877e198332c957127e16eb214afe/data/MELD/${split}_sent_emo.csv" \
    -o "out/friends_eval/meld_csv/${split}_sent_emo.csv" || break
done

uv run --script diarization/evaluation/eval_friends_meld.py prepare \
  "$friends" out/friends_eval/meld_csv --last-episode 12 --out out/friends_eval/reference

uv run --script diarization/evaluation/eval_friends_meld.py score \
  out/friends_eval/reference/meld_reference.json --results "$results"
```

Оценка сохраняет сводку и детали по репликам в `out/friends_eval/`. Подготовка эталона локально создаёт WAV и черновые TSV, которые можно открыть в Audacity и табличном редакторе. В Git опубликованы код и агрегаты.

### Следующий чекпоинт · DER/SER на Friends

Прослушать подготовленные сцены, проверить таймкоды и голоса, добавить пропущенные реплики и наложения. Сохранить проверенные интервалы в `speakers_reviewed.tsv`. Колонки `start` и `end` задаются в секундах от начала серии. Затем выполнить:

```bash
uv run --script diarization/evaluation/eval_friends.py --results "$results" manual \
  out/friends_eval/reference/regions_draft.tsv \
  out/friends_eval/reference/speakers_reviewed.tsv
```

Будут рассчитаны DER и SER с допусками 0 и 0,25 с по `turns.json`. Для проверки итоговых реплик, которые уходят в озвучку, можно добавить `--hyp markup`.
