import os
import joblib
import pandas as pd
import boto3

BUCKET = "my-llm-churn-bucket"
PREFIX = "models"

LOCAL_MODEL_DIR = "/home/ec2-user/churn_pipeline/models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

s3 = boto3.client("s3")

def download_if_missing(filename: str) -> str:
    local_path = os.path.join(LOCAL_MODEL_DIR, filename)
    if not os.path.exists(local_path):
        s3.download_file(BUCKET, f"{PREFIX}/{filename}", local_path)
    return local_path

FEATURES = joblib.load(download_if_missing("feature_columns.joblib"))
XGB = joblib.load(download_if_missing("churn_xgb.joblib"))
LOGREG_PIPE = joblib.load(download_if_missing("churn_logreg_pipeline.joblib"))

def _prep(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0
    return df[FEATURES]

def predict(payload: dict, model_choice: str = "xgb", threshold: float = 0.35) -> dict:
    X = _prep(payload)

    if model_choice.lower() in ["xgb", "xgboost"]:
        prob = float(XGB.predict_proba(X)[0, 1])
    elif model_choice.lower() in ["logreg", "logistic"]:
        prob = float(LOGREG_PIPE.predict_proba(X)[0, 1])
    else:
        raise ValueError("model_choice must be 'xgb' or 'logreg'")

    pred = int(prob >= threshold)
    return {"churn_probability": prob, "churn_prediction": pred, "threshold": float(threshold)}
