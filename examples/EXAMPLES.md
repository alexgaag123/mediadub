# Примеры
Запускать из корня проекта, .env файл лежит там же.

Для всех примеров нужен
```bash
uv sync
```


### Пример работы с базой данных.
Описание таблиц и типов можно найти  scripts/init.sql
```bash
uv run --env-file .env python ./examples/db_test.py
```


### Пример работы с s3 хранилищем.
```bash
uv run --env-file .env python ./examples/s3_upload_download.py
uv run --env-file .env python ./examples/s3_upload_download_as_string.py
```

## Примеры запуска этапов пайплайна
После сабмита в консоли будет напечатана ссылка. После завершения обработки, по ней будет доступен артефакт.
### Перевод 

Группа зависимостей
```bash
uv sync --group translation
```

Сначала необходимо запустить celery воркера, который будет слушать очередь задач на перевод, читать файл из хранилища, засылать его строки в модель на перевод, сохранять новый файл. В процессе также обновлять статусы бд. Делается это следующей командой.
```bash
WORKER_TYPE="translation" uv run --group translation --env-file .env  celery -A media_dub.pipeline.celery_app worker -Q translation_queue -P solo -l info
``` 
  Теперь, в другом терминале можно отправить задачу на первод.
```bash
uv run --env-file .env python ./examples/translation.py
```

### Озвучка
Сначала положите абсолютно лбой .wav файл с речью в examples/sample_audio.wav
Группа зависимостей
```bash
uv sync --group voice-over
```
Аналогично прошлому пункту - сначала запускаем воркера.
```bash
WORKER_TYPE="voice-over" uv run --group voice-over --env-file .env  celery -A media_dub.pipeline.celery_app worker -Q voice_over_queue -P solo -l info
```
Теперь, в другом терминале можно отправить задачу на озвучку.
```bash
uv run --env-file .env python ./examples/voice_over.py
```
