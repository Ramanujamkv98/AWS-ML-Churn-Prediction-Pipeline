import os

import boto3
import joblib
import pandas as pd

BUCKET = "my-llm-churn-bucket"
PREFIX = "models"

# Uses /tmp/model on Streamlit Cloud
LOCAL_MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/model")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

s3 = boto3.client("s3")


def download_if_missing(filename: str) -> str:
    local_path = os.path.join(LOCAL_MODEL_DIR, filename)

    if not os.path.exists(local_path):
        s3_key = f"{PREFIX}/{filename}"
        s3.download_file(BUCKET, s3_key, local_path)

    return local_path


FEATURES = joblib.load(
    download_if_missing("feature_columns.joblib")
)

XGB = joblib.load(
    download_if_missing("churn_xgb.joblib")
)

LOGREG_PIPE = joblib.load(
    download_if_missing("churn_logreg_pipeline.joblib")
)


def _prep(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])

    for column in FEATURES:
        if column not in df.columns:
            df[column] = 0

    return df[FEATURES]


def predict(
    payload: dict,
    model_choice: str = "xgb",
    threshold: float = 0.35
) -> dict:
    X = _prep(payload)

    if model_choice.lower() in ["xgb", "xgboost"]:
        probability = float(XGB.predict_proba(X)[0, 1])

    elif model_choice.lower() in ["logreg", "logistic"]:
        probability = float(LOGREG_PIPE.predict_proba(X)[0, 1])

    else:
        raise ValueError("model_choice must be 'xgb' or 'logreg'")

    prediction = int(probability >= threshold)

    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": prediction,
        "threshold": float(threshold),
    }
