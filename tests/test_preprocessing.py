import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.preprocessing import (CAT_COLS, NUM_COLS, FeatureEngineer,
                                    NumericCoercer, build_pipeline)


def test_numeric_coercer_blanks_to_nan_and_float(raw_rows):
    out = NumericCoercer(["TotalCharges"]).fit_transform(raw_rows)
    assert out["TotalCharges"].dtype.kind == "f"  # float
    assert np.isnan(out.loc[1, "TotalCharges"])    # NaN на строке с пустой строкой
    assert out.loc[0, "TotalCharges"] == 1320.0
    assert len(out) == len(raw_rows)


def test_n_services_counts_active_with_traps(raw_rows):
    df = NumericCoercer(["TotalCharges"]).fit_transform(raw_rows)
    out = FeatureEngineer().fit_transform(df)
    # Row A: PhoneService=Yes(1) + InternetService=DSL(1) + OnlineSecurity=Yes(1) => 3
    assert out.loc[0, "n_services"] == 3
    # Row B: только PhoneService=Yes => 1 (интернет = "No internet service", MultipleLines="No phone service")
    assert out.loc[1, "n_services"] == 1
    # Row C: все 9 сервисов активны => 9
    assert out.loc[2, "n_services"] == 9
    assert out["n_services"].dtype.kind in "iu"


def test_tenure_bucket_edges(raw_rows):
    df = NumericCoercer(["TotalCharges"]).fit_transform(raw_rows)
    out = FeatureEngineer().fit_transform(df)
    assert out.loc[0, "tenure_bucket"] == "13-24"  # tenure 24
    assert out.loc[1, "tenure_bucket"] == "0-12"   # tenure 0
    assert out.loc[2, "tenure_bucket"] == "49+"    # tenure 60


def test_charges_per_month_handles_zero_tenure(raw_rows):
    df = NumericCoercer(["TotalCharges"]).fit_transform(raw_rows)
    out = FeatureEngineer().fit_transform(df)
    assert out.loc[0, "charges_per_month_of_tenure"] == 55.0  # 1320 / 24
    # tenure=0 -> делим на max(0,1)=1; TotalCharges NaN -> NaN (импутируется позже)
    assert np.isnan(out.loc[1, "charges_per_month_of_tenure"])


def test_build_pipeline_fits_and_predicts_proba(raw_rows):
    df = pd.concat([raw_rows] * 6, ignore_index=True)  # 18 строк
    y = np.array([0, 1, 0] * 6)
    pipe = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    pipe.fit(df.drop(columns=["customerID"]), y)
    proba = pipe.predict_proba(df.drop(columns=["customerID"]))[:, 1]
    assert proba.shape == (18,)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert "n_services" in NUM_COLS and "tenure_bucket" in CAT_COLS
