# Бинарная классификация Heart Disease

Учебный проект с базовой моделью LightGBM. Целевая переменная
`HeartDisease` принимает значения 0 и 1 и обозначает отсутствие или
наличие заболевания сердца.

## Источник данных

[Heart Failure Prediction Dataset, fedesoriano, Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
Скрипт загружает `heart.csv` из версии 1. В таблице 918 строк, 11 признаков и одна
целевая переменная. На Kaggle указано **Database: Open Database, Contents: © Original Authors**.
Лицензия базы данных — [Open Database License 1.0 (ODbL)](https://opendatacommons.org/licenses/odbl/1.0/).

## Установка

Понадобятся Git и [uv](https://docs.astral.sh/uv/getting-started/installation/).
Запускайте команды из корня проекта. После распаковки ZIP выполните `git init` один
раз, чтобы pre-commit мог работать с Git. В клонированном репозитории этот шаг уже выполнен.

На macOS установите системную зависимость LightGBM командой `brew install libomp`.
На Linux, где нет `libgomp.so.1`, установите пакет `libgomp1` или его аналог.

```bash
uv sync --locked
```

При необходимости uv скачает Python 3.12. Зависимости устанавливаются из `uv.lock`.

## Загрузка данных

```bash
uv run python -m src.get_data
```

Скрипт скачивает публичный архив Kaggle и сохраняет CSV в `data/raw/heart.csv`.
Скрипт может начать загрузку без токена. Если Kaggle требует авторизацию, передайте API-токен через
переменную окружения `KAGGLE_API_TOKEN`. Скрипт не записывает токен в файлы.

## Проверка кода

```bash
git add .
uv run pre-commit run --all-files
uv run pre-commit install
```

Первые две команды запускают Ruff для файлов проекта. Последняя устанавливает
проверки перед будущими коммитами.

## Обучение

```bash
uv run python -m src.train
```

Данные делятся на обучающую и тестовую выборки в пропорции 80/20 со
стратификацией по целевой переменной и `random_state=42`. One-hot-кодировщик и
LightGBM обучаются на обучающей выборке. Accuracy считается по предсказанным
классам, ROC-AUC — по вероятностям класса 1 на отложенной выборке.

- `artifacts/model.pkl` — обученный пайплайн с кодировщиком и классификатором.
- `artifacts/metrics.json` — метрики и размеры выборок.

[Отчёт о выполнении ДЗ](reports/hw-1.md)
