"""Обучение churn-модели.

Пайплайн этапов:
  1. Stratified train/test split (тест трогаем один раз — в самом конце).
  2. Baselines (DummyClassifier: most_frequent = zero-rule, stratified = random).
  3. Сравнение трёх моделей по 5-fold StratifiedKFold (mean±std метрик).
  4. Лёгкий тюнинг лучшей модели (GridSearchCV, scoring=average_precision).
  5. Выбор порога на OOF-вероятностях train (recall >= 0.80, max precision).
  6. Оценка на тесте ОДИН РАЗ при замороженном пороге + гейты приёмки.
  7. Сохранение артефактов (pipeline.pkl, threshold.json, metrics.json, графики).

Запуск из корня репозитория: ``python scripts/train.py``
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import (CalibratedClassifierCV,  # noqa: E402
                                 calibration_curve)
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import (GradientBoostingClassifier,  # noqa: E402
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score,  # noqa: E402
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV,  # noqa: E402
                                     StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preprocessing import build_pipeline, select_threshold  # noqa: E402

DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
RANDOM_STATE = 42
SCORING = "average_precision"  # PR-AUC — основная метрика отбора/тюнинга (Huyen ch.4)
MIN_RECALL = 0.80
GATE_RECALL, GATE_PRECISION = 0.75, 0.45

CV_SCORING = {"recall": "recall", "precision": "precision", "f1": "f1",
              "roc_auc": "roc_auc", "ap": "average_precision"}


def load_xy():
    df = pd.read_csv(DATA)
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=["Churn", "customerID"])
    return X, y


def cv_report(model, X, y, cv):
    """mean±std по recall/precision/f1/roc_auc/ap для пайплайна с данной моделью."""
    pipe = build_pipeline(model)
    res = cross_validate(pipe, X, y, cv=cv, scoring=CV_SCORING)
    return {m: (round(float(res[f"test_{m}"].mean()), 4),
                round(float(res[f"test_{m}"].std()), 4))
            for m in CV_SCORING}


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    X, y = load_xy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    metrics = {"class_balance": {str(k): round(float(v), 3)
                                 for k, v in y.value_counts(normalize=True).items()}}

    # --- Baselines (zero-rule + random) ---
    baselines = {}
    for strat in ["most_frequent", "stratified"]:
        dummy = build_pipeline(DummyClassifier(strategy=strat, random_state=RANDOM_STATE))
        res = cross_validate(dummy, X_tr, y_tr, cv=cv,
                             scoring={"recall": "recall", "roc_auc": "roc_auc"})
        baselines[strat] = {"recall": round(float(res["test_recall"].mean()), 4),
                            "roc_auc": round(float(res["test_roc_auc"].mean()), 4)}
    metrics["baselines"] = baselines

    # --- Сравнение трёх моделей ---
    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),  # без class_weight
    }
    comparison = {name: cv_report(m, X_tr, y_tr, cv) for name, m in models.items()}
    metrics["cv_comparison"] = comparison

    # лучшая по PR-AUC (ap), затем recall
    best_name = max(comparison,
                    key=lambda n: (comparison[n]["ap"][0], comparison[n]["recall"][0]))
    metrics["best_model"] = best_name
    print(f"Best model by PR-AUC: {best_name}")

    # --- Лёгкий тюнинг лучшей модели (>=2 гиперпараметра, scoring=PR-AUC) ---
    grids = {
        "logreg": {"model__C": [0.05, 0.1, 0.5, 1.0],
                   "model__class_weight": ["balanced", {0: 1, 1: 5}]},
        "rf": {"model__n_estimators": [200, 400],
               "model__max_depth": [None, 8, 16],
               "model__class_weight": ["balanced", {0: 1, 1: 5}]},
        "gb": {"model__n_estimators": [100, 200],
               "model__learning_rate": [0.05, 0.1],
               "model__max_depth": [2, 3]},
    }
    search = GridSearchCV(build_pipeline(models[best_name]), grids[best_name],
                          scoring=SCORING, cv=cv, n_jobs=-1)
    search.fit(X_tr, y_tr)
    metrics["best_params"] = {k: str(v) for k, v in search.best_params_.items()}
    metrics["best_cv_ap"] = round(float(search.best_score_), 4)
    print(f"Best params: {metrics['best_params']}")

    # --- Калибровка вероятностей (Huyen ch.6): сравниваем до/после по Brier на OOF ---
    # Режем по predict_proba, поэтому калибровка важна; GB калибрована хуже LogReg.
    best_model_step = search.best_estimator_.named_steps["model"]
    candidates = {
        "uncalibrated": build_pipeline(clone(best_model_step)),
        "calibrated": build_pipeline(
            CalibratedClassifierCV(clone(best_model_step), method="sigmoid", cv=5)),
    }
    oof_probs, brier = {}, {}
    for name, cand in candidates.items():
        p = cross_val_predict(cand, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        oof_probs[name] = p
        brier[name] = round(float(brier_score_loss(y_tr, p)), 5)
    metrics["brier_oof"] = brier

    plt.figure()
    for name in candidates:
        frac_pos, mean_pred = calibration_curve(y_tr, oof_probs[name], n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label=f"{name} (Brier={brier[name]})")
    plt.plot([0, 1], [0, 1], ls="--", color="grey", label="идеально калибровано")
    plt.xlabel("Средняя предсказанная вероятность")
    plt.ylabel("Доля реальных churn")
    plt.title("Калибровочные кривые (OOF, train)")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "calibration.png", dpi=120); plt.close()

    # финальная модель — с лучшей (меньшей) Brier
    final_name = min(brier, key=brier.get)
    metrics["calibration_choice"] = final_name
    print(f"Calibration: {brier} -> chosen '{final_name}'")
    best_pipe = candidates[final_name]
    oof = oof_probs[final_name]

    # --- Порог на OOF-вероятностях train ---
    threshold = select_threshold(y_tr.to_numpy(), oof, min_recall=MIN_RECALL)
    print(f"Frozen threshold: {threshold:.4f}")

    precision, recall, _ = precision_recall_curve(y_tr, oof)
    pred_oof = (oof >= threshold).astype(int)
    r_pt, p_pt = recall_score(y_tr, pred_oof), precision_score(y_tr, pred_oof)
    plt.figure()
    plt.plot(recall, precision, label="PR curve")
    plt.axvline(MIN_RECALL, ls="--", color="grey", label=f"target recall={MIN_RECALL}")
    plt.scatter([r_pt], [p_pt], color="red", zorder=5,
                label=f"chosen thr={threshold:.3f}\n(recall={r_pt:.2f}, prec={p_pt:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall curve (OOF, train)")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "pr_curve.png", dpi=120); plt.close()

    # --- Рефит на полном train, тест ОДИН РАЗ ---
    best_pipe.fit(X_tr, y_tr)
    proba_te = best_pipe.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= threshold).astype(int)

    test_metrics = {
        "threshold": round(threshold, 4),
        "recall": round(float(recall_score(y_te, pred_te)), 4),
        "precision": round(float(precision_score(y_te, pred_te)), 4),
        "f1": round(float(f1_score(y_te, pred_te)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba_te)), 4),
        "average_precision": round(float(average_precision_score(y_te, proba_te)), 4),
    }
    metrics["test"] = test_metrics

    cm = confusion_matrix(y_te, pred_te)
    cm_df = pd.DataFrame(cm, index=["True:No", "True:Yes"],
                         columns=["Pred:No", "Pred:Yes"])
    metrics["confusion_matrix"] = {"index": list(cm_df.index),
                                   "columns": list(cm_df.columns),
                                   "values": cm.tolist()}

    # --- Slice-based recall по типу контракта (Huyen ch.6) ---
    te = X_te.copy()
    te["_y"], te["_pred"] = y_te.to_numpy(), pred_te
    slices = {}
    for val, grp in te.groupby("Contract"):
        churners = grp["_y"] == 1
        if int(churners.sum()) > 0:
            slices[val] = {
                "churners": int(churners.sum()),
                "recall": round(float((grp.loc[churners, "_pred"] == 1).mean()), 4),
            }
    metrics["slice_recall_by_contract"] = slices

    # --- Бизнес-перевод (на 1000 клиентов) ---
    n = len(y_te)
    flagged = int(pred_te.sum())
    tp = int(((pred_te == 1) & (y_te.to_numpy() == 1)).sum())
    fn = int(((pred_te == 0) & (y_te.to_numpy() == 1)).sum())
    metrics["business"] = {
        "test_size": n,
        "flagged": flagged,
        "flagged_per_1000": round(flagged / n * 1000),
        "true_churners_caught": tp,
        "caught_per_1000": round(tp / n * 1000),
        "missed_per_1000": round(fn / n * 1000),
    }

    # confusion matrix как картинка
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], cm_df.columns); ax.set_yticks([0, 1], cm_df.index)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title(f"Confusion matrix (test, thr={threshold:.3f})")
    plt.tight_layout(); plt.savefig(PLOTS / "confusion_matrix.png", dpi=120); plt.close()

    print("\nConfusion matrix (test):")
    print(cm_df)
    print("\nTest metrics:", json.dumps(test_metrics, indent=2))

    # --- Сохранение артефактов ---
    joblib.dump(best_pipe, RESULTS / "churn_pipeline.pkl")
    (RESULTS / "threshold.json").write_text(
        json.dumps({"threshold": threshold, "min_recall": MIN_RECALL}, indent=2))
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # --- Гейты приёмки ---
    assert test_metrics["recall"] >= GATE_RECALL, \
        f"recall {test_metrics['recall']} < {GATE_RECALL}"
    assert test_metrics["precision"] >= GATE_PRECISION, \
        f"precision {test_metrics['precision']} < {GATE_PRECISION}"
    print(f"\nGATES OK: recall={test_metrics['recall']} >= {GATE_RECALL}, "
          f"precision={test_metrics['precision']} >= {GATE_PRECISION}")


if __name__ == "__main__":
    main()
