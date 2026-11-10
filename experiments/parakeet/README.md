# Parakeet v2 и v3

[Все эксперименты](../README.md)

**Задача.** Проверить, как замена английской модели Parakeet TDT 0.6B v2 на многоязычную v3 влияет на текст и таймкоды. [Карточка v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) · [Карточка v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

## Методика

1. Берём готовый выход MediaDub с Parakeet v2 для первого сезона «Любовь, смерть и роботы». Нужны `asr_input.wav`, `words.json`, `nemotron_w60.json` и `turns.json` каждой серии.
2. Повторяем ASR-рецепт MediaDub на том же аудио с v3. Сохраняем `v2.json` и `v3.json`, назначая словам обеих версий ID из одного `turns.json`.
3. Открываем HTML и слушаем спорные места. Отдельно показываем текстовые различия и смену ID у совпавших слов с близкими таймкодами. При общей диаризации разный ID может появиться из-за сдвига таймкода слова у границы реплики.

**Наблюдение.** По просмотренным сценам v3 лучше распознавал слова. Смена модели ASR не устранила ошибки в интервалах говорящих. Для прогона «Друзей» использовали v3 вместе с диаризацией MediaDub. CER **16,54%** относится к первым 12 сериям «Друзей» и английским SDH. [Протокол и результаты](../../diarization/README.md)

## Запуск

Нужны окружение NeMo MediaDub, CUDA и файл `parakeet-tdt-0.6b-v3.nemo` в `parakeet_test/`. Передай в `--input-dir` папку с результатами базового прогона v2. Запуск без `--episode` ожидает все 18 серий. Вариант без `--input-dir` читает пути к уже подготовленному аудио из файлов предыдущего опыта с Nemotron.

```bash
baseline='/путь/к/выходам/сезона/MediaDub'
.envs/nemo/bin/python -m experiments.parakeet.run_parakeet_v3_season --input-dir "$baseline"
python3 -m experiments.parakeet.compare_parakeet --season parakeet_test/ldr_season_v3
xdg-open parakeet_test/ldr_season_v3/index.html
```

[Прогон сезона](run_parakeet_v3_season.py) · [Генератор HTML](compare_parakeet.py)
