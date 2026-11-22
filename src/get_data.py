import csv
import io
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction"
    "?datasetVersionNumber=1"
)
COLUMNS = (
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
    "HeartDisease",
)


def main() -> None:
    request = Request(DATASET_URL, headers={"User-Agent": "heart-disease-baseline/0.1"})
    if token := os.environ.get("KAGGLE_API_TOKEN"):
        # Авторизация нужна только при запросе к Kaggle, перед перенаправлением на хранилище.
        request.add_unredirected_header("Authorization", f"Bearer {token}")

    try:
        with urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
    except HTTPError as error:
        if error.code in (401, 403):
            raise SystemExit(
                f"Kaggle вернул HTTP {error.code}. Проверьте переменную KAGGLE_API_TOKEN."
            ) from error
        raise SystemExit(f"Kaggle вернул HTTP {error.code}.") from error
    except (URLError, TimeoutError) as error:
        raise SystemExit(f"Не удалось скачать архив Kaggle: {error}") from error

    try:
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            csv_bytes = archive.read("heart.csv")
    except (BadZipFile, KeyError) as error:
        raise SystemExit("Ответ Kaggle не содержит ожидаемый ZIP-архив с heart.csv.") from error

    with io.TextIOWrapper(io.BytesIO(csv_bytes), encoding="utf-8-sig", newline="") as source:
        rows = csv.reader(source)
        if tuple(next(rows, [])) != COLUMNS:
            raise SystemExit("Столбцы heart.csv отличаются от ожидаемой схемы.")
        row_count = sum(bool(row) for row in rows)
        if row_count != 918:
            raise SystemExit(f"Ожидалось 918 строк, получено {row_count}.")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=DATA_PATH.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(csv_bytes)
            temporary.close()
            temporary_path.replace(DATA_PATH)
        finally:
            temporary_path.unlink(missing_ok=True)

    relative_path = DATA_PATH.relative_to(PROJECT_ROOT)
    print(f"Источник: {DATASET_URL}")
    print(f"Файл: {relative_path} ({row_count} строк, {len(COLUMNS)} столбцов)")


if __name__ == "__main__":
    main()
