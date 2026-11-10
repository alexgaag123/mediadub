# NVIDIA Nemotron 3.5 Transcribe

[Все эксперименты](../README.md)

**Задача.** Проверить отдельный пайплайн, в котором Nemotron 3.5 Transcribe 30B‑A4B выдаёт текст, таймкоды и ID говорящих. В базовом MediaDub распознавание делает Parakeet, а диаризацию — Nemotron 3 Diarization и ReDimNet2. [Карточка Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)

## Методика

1. На первых трёх минутах S01E02 «Чернобыля» проверили русский текст на CPU. Модель распознала русский фрагмент, которого не было в прежней транскрипции.
2. На RTX 3060 с 12 ГБ запустили первый сезон «Любовь, смерть и роботы» на подготовленных MediaDub дорожках `asr_input.wav`. Использовали NeMo‑Speech.cpp с CUDA, декодер GGUF IQ4_NL и перенос экспертов первых 40 блоков декодера в системную память. Скрипт подаёт серии до 20 минут целиком, более длинное аудио делит на части. ID частей сохраняются раздельно.
3. В HTML сравнивали текст и голоса с базовым MediaDub. Вариант `--hybrid` назначает словам Nemotron ID по интервалам MediaDub и отмечает возможные пропуски. Он также показывает случаи, когда модель вернула текст без таймкодов слов.

**Наблюдение.** Nemotron лучше распознавал отдельные слова, но местами пропускал часть разговора. При прослушивании ID оказались менее стабильными, чем в базовом MediaDub. Причину пропусков не установили. Гибрид оставили в эксперименте.

## Запуск

Нужны FFmpeg, сборка CUDA `nemotron_test/nemo-cuda/bin/nemo-speech` с поддержкой `--asr.salm.n_cpu_moe` и два GGUF в `nemotron_test/models/`. Имена файлов зафиксированы в скрипте. Старая установленная сборка возвращала `unknown option` на флаге CPU-offload. [Профиль 12 ГБ в документации NeMo‑Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/536b17a41bef62d2a67a40ed3df61df26a4389fe/docs/salm/configuration.md)

```bash
baseline='/путь/к/выходам/сезона/MediaDub'
python3 -m experiments.nemotron35.run_nemotron_ldr --input-dir "$baseline"
python3 -m experiments.nemotron35.compare_ldr_outputs
python3 -m experiments.nemotron35.compare_ldr_outputs --hybrid
xdg-open out/ldr_nemotron_gpu/index.html
```

Скрипт по «Чернобылю» рассчитан на пять подготовленных серий в `out/ldr_season3/` и использует блоки по 10 минут. Подтверждённое наблюдение о русском тексте относится к трёхминутному фрагменту.

[Прогон «Любовь, смерть и роботы»](run_nemotron_ldr.py) · [Скрипт для «Чернобыля»](run_nemotron_chernobyl.py) · [Сравнение с MediaDub](compare_ldr_outputs.py)
