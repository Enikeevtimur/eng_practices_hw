import json
import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.get_data import DATA_PATH, PROJECT_ROOT

CATEGORICAL_COLUMNS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET = "HeartDisease"
RANDOM_STATE = 42


def main() -> None:
    if not DATA_PATH.is_file():
        raise SystemExit("Сначала скачайте данные: uv run python -m src.get_data")

    data = pd.read_csv(DATA_PATH)
    features = data.drop(columns=TARGET)
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )

    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            )
        ],
        remainder="passthrough",
    ).set_output(transform="pandas")
    model = make_pipeline(
        preprocessing,
        LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbosity=-1),
    )
    model.fit(x_train, y_train)

    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(x_test)),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]),
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "random_state": RANDOM_STATE,
    }

    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with (artifacts / "model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    metrics_json = json.dumps(metrics, indent=2)
    (artifacts / "metrics.json").write_text(f"{metrics_json}\n", encoding="utf-8")
    print(metrics_json)
    print("Сохранены artifacts/model.pkl и artifacts/metrics.json")


if __name__ == "__main__":
    main()
