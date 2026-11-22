# Домашнее задание № 1

## Загрузка данных

```bash
uv run python -m src.get_data
```

В рабочем проекте проверен сохранённый CSV. Для контрольного запуска загрузчика
сетевой ответ заменили ZIP-архивом с этим же CSV. Скрипт проверил схему и число
строк, сохранил файл и вывел:

```text
Источник: https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction?datasetVersionNumber=1
Файл: data/raw/heart.csv (918 строк, 12 столбцов)
```

## Проверка кода

`uv lock --check` подтвердил соответствие `uv.lock` и `pyproject.toml`.
Команда `uv run pre-commit run --all-files` завершилась успешно:

```text
ruff check...............................................................Passed
ruff format..............................................................Passed
```

## Обучение

```bash
uv run python -m src.train
```

LightGBM обучен на 734 строках и оценён на 184 строках. Скрипт создал
`artifacts/model.pkl` и `artifacts/metrics.json`. Содержимое `metrics.json`:

```json
{
  "accuracy": 0.8695652173913043,
  "roc_auc": 0.9253945480631277,
  "train_rows": 734,
  "test_rows": 184,
  "random_state": 42
}
```
