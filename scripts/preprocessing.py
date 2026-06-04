"""Кастомные трансформеры, фабрика Pipeline и выбор порога для churn-модели.

Все классы определены здесь, чтобы быть импортируемыми при распиклинге
(`joblib.load`) обученного пайплайна в `predict.py`.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class NumericCoercer(BaseEstimator, TransformerMixin):
    """Приводит указанные колонки к числу; нечисловые/пустые значения → NaN."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X


SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
# Значения, означающие, что сервис НЕ активен (ловушка: InternetService=No, аддоны=No internet service)
INACTIVE_VALUES = {"No", "No internet service", "No phone service"}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Добавляет tenure_bucket, charges_per_month_of_tenure, n_services. Stateless."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["tenure_bucket"] = pd.cut(
            X["tenure"], bins=[-1, 12, 24, 48, np.inf],
            labels=["0-12", "13-24", "25-48", "49+"],
        ).astype(object)
        denom = X["tenure"].clip(lower=1)  # защита от деления на 0
        X["charges_per_month_of_tenure"] = X["TotalCharges"] / denom
        active = ~X[SERVICE_COLS].isin(INACTIVE_VALUES)
        X["n_services"] = active.sum(axis=1).astype(int)
        return X


# Колонки после FeatureEngineer (включают сгенерированные фичи)
NUM_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "charges_per_month_of_tenure", "n_services",
]
CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_bucket",
]


def build_pipeline(model):
    """Единый leakage-free Pipeline: coerce → engineer → ColumnTransformer → model.

    Весь фиттинг препроцессоров происходит ВНУТРИ пайплайна, на train-фолде.
    """
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([
        ("coerce", NumericCoercer(["TotalCharges"])),
        ("engineer", FeatureEngineer()),
        ("prep", preprocessor),
        ("model", model),
    ])


def select_threshold(y_true, y_proba, min_recall=0.80):
    """Порог с recall >= min_recall и максимальной при этом precision.

    Возвращает float. Если целевой recall недостижим — порог с максимальным recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall на 1 длиннее thresholds; последняя точка (recall=0) без порога
    precision, recall = precision[:-1], recall[:-1]
    mask = recall >= min_recall
    if not mask.any():
        return float(thresholds[int(np.argmax(recall))])
    candidates = np.where(mask)[0]
    best = candidates[int(np.argmax(precision[candidates]))]
    return float(thresholds[best])
  
