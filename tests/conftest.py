import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def raw_rows():
    """Минимальный raw-фрейм со всеми колонками Telco (значения-ловушки включены)."""
    return pd.DataFrame([
        # активный телефон + интернет(DSL) + часть аддонов; tenure>0; TotalCharges число
        {"customerID": "A", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
         "Dependents": "No", "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No",
         "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
         "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
         "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
         "PaymentMethod": "Mailed check", "MonthlyCharges": 55.0, "TotalCharges": "1320.0"},
        # без интернета: все интернет-аддоны "No internet service"; телефон есть; tenure=0; пустой TotalCharges
        {"customerID": "B", "gender": "Male", "SeniorCitizen": 1, "Partner": "No",
         "Dependents": "No", "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No phone service",
         "InternetService": "No", "OnlineSecurity": "No internet service",
         "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
         "TechSupport": "No internet service", "StreamingTV": "No internet service",
         "StreamingMovies": "No internet service", "Contract": "Month-to-month",
         "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
         "MonthlyCharges": 20.0, "TotalCharges": " "},
        # fiber + все 9 сервисов активны
        {"customerID": "C", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
         "Dependents": "Yes", "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
         "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
         "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
         "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
         "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 110.0,
         "TotalCharges": "6600.0"},
    ])
