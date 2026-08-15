from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# Load model artifacts from the repository
# --------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"


def load_artifact(filename: str):
    artifact_path = MODEL_DIR / filename

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {artifact_path}. "
            f"Make sure {filename} is uploaded to the models/ folder."
        )

    return joblib.load(artifact_path)


FEATURES = load_artifact("feature_columns.joblib")
XGB = load_artifact("churn_xgb.joblib")
LOGREG_PIPE = load_artifact("churn_logreg_pipeline.joblib")


# --------------------------------------------------
# Prepare one observation for prediction
# --------------------------------------------------
def _prep(payload: dict) -> pd.DataFrame:
    payload = payload.copy()

    # Convert the UI model selection into the same one-hot columns
    # created by processing.py during model training.
    selected_model = (
        str(payload.pop("primary_model_7d", "gpt_3.5"))
        .strip()
        .replace("-", "_")
    )

    payload["primary_model_7d_gpt_4.1"] = int(
        selected_model == "gpt_4.1"
    )

    payload["primary_model_7d_gpt_4o"] = int(
        selected_model == "gpt_4o"
    )

    # gpt_4o_mini and any unknown model map to "other".
    payload["primary_model_7d_other"] = int(
        selected_model not in {
            "gpt_3.5",
            "gpt_4.1",
            "gpt_4o",
        }
    )

    df = pd.DataFrame([payload])

    # Add missing features, remove extras and preserve training order.
    X = df.reindex(
        columns=FEATURES,
        fill_value=0,
    )

    return X


# --------------------------------------------------
# Generate churn prediction
# --------------------------------------------------
def predict(
    payload: dict,
    model_choice: str = "xgb",
    threshold: float = 0.35,
) -> dict:
    X = _prep(payload)

    model_name = model_choice.lower().strip()

    if model_name in {"xgb", "xgboost"}:
        probability = float(
            XGB.predict_proba(X)[0, 1]
        )

    elif model_name in {"logreg", "logistic"}:
        probability = float(
            LOGREG_PIPE.predict_proba(X)[0, 1]
        )

    else:
        raise ValueError(
            "model_choice must be 'xgb' or 'logreg'"
        )

    prediction = int(probability >= threshold)

    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": prediction,
        "threshold": float(threshold),
        "model": model_name,
    }
