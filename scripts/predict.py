"""Инференс churn-модели на новых клиентах.

Запуск из корня репозитория: ``python scripts/predict.py``
Читает data/new_customers.csv (та же схема, без Churn), пишет
results/predictions.csv с колонками customerID, churn_pred, churn_proba.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Импорт нужен, чтобы распиклить кастомные трансформеры внутри сохранённого пайплайна.
from scripts.preprocessing import FeatureEngineer, NumericCoercer  # noqa: E402,F401

RESULTS = ROOT / "results"
NEW_CUSTOMERS = ROOT / "data" / "new_customers.csv"


def predict_new(pipe, df, threshold):
    """Возвращает DataFrame [customerID, churn_pred, churn_proba].

    customerID откладывается и дропается перед predict_proba (это идентификатор,
    не фича), затем приклеивается обратно по индексу.
    """
    ids = df["customerID"].reset_index(drop=True)
    X = df.drop(columns=["customerID"])
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pd.DataFrame({
        "customerID": ids,
        "churn_pred": pred,
        "churn_proba": proba.round(4),
    })


def main():
    pipe = joblib.load(RESULTS / "churn_pipeline.pkl")
    threshold = json.loads((RESULTS / "threshold.json").read_text())["threshold"]
    df = pd.read_csv(NEW_CUSTOMERS)
    out = predict_new(pipe, df, threshold)
    out.to_csv(RESULTS / "predictions.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
