import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.predict import predict_new
from scripts.preprocessing import build_pipeline


def test_predict_new_schema_and_id_reattach(raw_rows):
    train = pd.concat([raw_rows] * 6, ignore_index=True)
    y = np.array([0, 1, 0] * 6)
    pipe = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    pipe.fit(train.drop(columns=["customerID"]), y)

    new = raw_rows.copy()  # содержит customerID, без Churn
    out = predict_new(pipe, new, threshold=0.5)
    assert list(out.columns) == ["customerID", "churn_pred", "churn_proba"]
    assert list(out["customerID"]) == list(new["customerID"])
    assert set(out["churn_pred"].unique()).issubset({0, 1})
    assert ((out["churn_proba"] >= 0) & (out["churn_proba"] <= 1)).all()
