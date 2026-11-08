# Примеры
Запускать из корня проекта, .env файл лежит там же.

Для всех примеров кроме последнего достаточно обычного
```bash
uv sync
```

Для последнего уже нужна новая группа зависимостей
```bash
uv sync --group translation
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

### Пример запуска этапа пайплайна
Сначала необходимо запустить celery воркера, который будет слушать очередь задач на перевод, читать файл из хранилища, засылать его строки в модель на перевод, сохранять новый файл. В процессе также обновлять статусы бд. Делается это следующей командой.
```bash
uv run --group translation --env-file .env celery -A media_dub.pipeline.celery_app worker -Q translation_queue -P solo -l info
``` 
  Теперь, в другом терминале можно отправить задачу на первод.
```bash
uv run --env-file .env python ./examples/translation.py``` 
```
Результат будет напечатан в консоль + будет напечатана врменный url, по которому файл можно посмотреть в браузере