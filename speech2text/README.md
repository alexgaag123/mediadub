# Speech2text: Parakeet v3 и центральный канал

Сохранён лучший проверенный вариант распознавания на первом сезоне Friends:
**Parakeet TDT 0.6B v3**, оригинальная английская дорожка, канал **FC** из 5.1,
исключение заставок и 585 зафиксированных аудиоблоков до 60 секунд.

На всех 24 сериях, 56 649 словах SDH: **WER 14,625%, CER 10,982%**.
Пропуски слов: 1 547, замены: 2 352, вставки: 4 386. Сбоев нет.
Это сравнение с SDH, содержащими неточности; дословность эталона на слух не подтверждена.

На выходе — полный текст и исходные таймкоды аудиоблоков. Модель получает только звук.
Тексты субтитров используются отдельно для оценки.

## Окружение и веса

Python 3.12, системные FFmpeg и ffprobe, NVIDIA GPU. В WSL используется пользовательский systemd.
Проверенные версии Python-зависимостей закреплены в [requirements.txt](requirements.txt).

```bash
rtk proxy uv venv .envs/nemo --python 3.12
rtk proxy uv pip install --python .envs/nemo/bin/python -r speech2text/requirements.txt
```

Веса: `nvidia/parakeet-tdt-0.6b-v3`, ревизия
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`.
Каталог: `data/models/parakeet-tdt-0.6b-v3/`.
Веса не входят в Git; существующее локальное окружение и модель можно использовать повторно.

```bash
rtk proxy .envs/nemo/bin/hf download nvidia/parakeet-tdt-0.6b-v3 \
  --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b \
  --include '*.json' '*.safetensors' \
  --local-dir data/models/parakeet-tdt-0.6b-v3
```

Перед оценкой проверяется SHA256 весов:
`3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11`.

## Подготовка Friends S01

Нужны все 24 MKV из использованного BDRip. Укажите свой каталог с сериями:

```bash
rtk proxy systemd-run --user --scope --quiet --unit=voice-friends-prepare \
  -p MemoryMax=8G -p MemorySwapMax=0 -p CPUQuota=200% \
  env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  nice -n 10 .envs/nemo/bin/python -m speech2text prepare \
  '/mnt/c/Users/Galex/Downloads/Friends S01 (1994-1995) BDRip-AVC' \
  data/friends_s01_fc_20261110 --center-channel
```

Код выбирает английскую дорожку. У S01E17 неверную языковую метку компенсирует явное
название оригинальной дорожки. Для FC необходима раскладка 5.1 или 5.1(side);
используется `pan=mono|c0=FC`, без усиления и шумоподавления.

Серии обрабатываются последовательно: mono float32 WAV 16 kHz, копия SDH,
очищенный текст и manifest с контрольными суммами. Видео не изменяются.
Существующая подготовка используется повторно только при совпадении проверок.

## Воспроизведение лучшего результата

```bash
rtk proxy systemd-run --user --scope --quiet --unit=voice-friends-evaluate \
  -p MemoryMax=8G -p MemorySwapMax=0 -p CPUQuota=200% \
  env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  nice -n 10 .envs/nemo/bin/python -m speech2text evaluate \
  data/friends_s01_fc_20261110 out/friends_best \
  --exclusions speech2text/friends_s01_exclusions.json \
  --boundaries speech2text/friends_s01_boundaries.json
```

Выходной каталог должен быть новым. В каждом каталоге серии сохраняются
`transcript.txt`, `transcript.json`, `reference.txt` и `alignment.txt`;
общие метрики находятся в `report.json`, параметры — в `protocol.json`.

Конфигурации содержат только интервалы и идентификаторы входных файлов, без текста реплик.
Проверки привязаны к аудио и субтитрам, а не к абсолютному пути проекта:
перенос каталога разрешён, подмена материала вызывает ошибку.
Для другого издания нужны свои интервалы — совпадение имени серии недостаточно.

Границы были выбраны по минимуму энергии 200-миллисекундного окна между 50-й и 60-й
секундами исходных блоков сведения в моно. Затем те же 585 границ использованы для FC.
Именно этот вариант дал указанные метрики. При запуске без `--boundaries` паузы ищутся
заново в поданном звуке, поэтому результат может отличаться.

Исключения заставок приблизительные, финальные сцены с диалогами сохранены.
Все 56 649 слов эталона и каждый сохранённый аудиосэмпл включены ровно один раз.

## Ресурсы и проверки

RAM ограничена 8 ГБ, CPU — двумя ядрами, swap отключён; FFmpeg работает в один поток.
Аллокатор GPU ограничен 50%, перед загрузкой требуется не менее 4 ГиБ свободной VRAM.
Проверяется запас памяти и диска: Windows C от 20 ГиБ, выходной диск от 8 ГиБ,
доступная RAM от 4 ГиБ. Для WSL это обязательные условия запуска.

```bash
rtk proxy .envs/nemo/bin/python -m unittest discover -s speech2text -p 'test_*.py' -v
```

Проверяются выбор FC, полный охват звука, корректность границ и таймкодов,
защита от другого набора данных, очистка SDH и сохранение результатов без перезаписи.
Вычисления модели: FP32, штатное greedy-декодирование, batch 1.
